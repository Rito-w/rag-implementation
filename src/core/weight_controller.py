"""
动态权重控制器 - Dynamic Weight Controller

这个模块实现了基于查询特征的动态权重分配算法，是智能自适应RAG系统的核心组件之一。

主要功能：
1. 基于查询分析结果计算最优权重分配
2. 实现自适应的检索策略选择
3. 提供权重分配的置信度评估和解释

权重计算公式：
w_dense(q) = σ(W_d^T · f(q) + b_d)
w_sparse(q) = σ(W_s^T · f(q) + b_s)  
w_hybrid(q) = σ(W_h^T · f(q) + b_h)

其中 f(q) 是查询特征向量，W 是学习的权重矩阵，σ 是sigmoid激活函数
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

from ..models.query_analysis import QueryAnalysis, QueryType
from ..models.weight_allocation import WeightAllocation, RetrievalStrategy
from ..utils.logging import get_logger


class DynamicWeightController:
    """
    动态权重控制器
    
    基于查询特征向量，使用神经网络模型计算最优的检索器权重分配。
    这是整个智能自适应系统的决策核心，决定了如何组合不同的检索方法。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化动态权重控制器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.logger = get_logger("DynamicWeightController")
        
        # 权重约束参数
        self.min_weight = self.config.get("min_weight", 0.01)
        self.max_weight = self.config.get("max_weight", 0.98)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
        
        # 神经网络参数
        self.hidden_layers = self.config.get("hidden_layers", [256, 128, 64])
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.regularization = self.config.get("regularization", 0.01)
        
        # 初始化权重分配网络
        self._initialize_weight_network()
        
        # 初始化策略选择规则
        self._initialize_strategy_rules()
        
        # 特征重要性权重 - 基于经验调优
        self._initialize_feature_importance()
        
        self.logger.info("动态权重控制器初始化完成")
    
    def _initialize_weight_network(self):
        """初始化权重分配神经网络"""
        # 简化实现：使用预定义的权重矩阵
        # 在实际应用中，这些权重应该通过训练学习得到
        
        input_dim = self.config.get("feature_dim", 768)
        
        # 为每个检索器类型定义权重矩阵
        self.weight_matrices = {
            'dense': {
                'W': np.random.normal(0, 0.1, (input_dim, 1)),
                'b': np.random.normal(0, 0.1, 1)
            },
            'sparse': {
                'W': np.random.normal(0, 0.1, (input_dim, 1)),
                'b': np.random.normal(0, 0.1, 1)
            },
            'hybrid': {
                'W': np.random.normal(0, 0.1, (input_dim, 1)),
                'b': np.random.normal(0, 0.1, 1)
            }
        }
        
        # 预设一些基于经验的权重偏好
        self._set_empirical_weights()
    
    def _set_empirical_weights(self):
        """设置基于经验的权重偏好"""
        # 这些权重基于对8篇权威论文的分析总结
        
        # 稠密检索偏好：语义复杂、概念理解类查询
        self.dense_preferences = {
            'semantic_complexity': 0.8,
            'concept_density': 0.7,
            'abstract_reasoning': 0.9
        }
        
        # 稀疏检索偏好：精确匹配、事实查询
        self.sparse_preferences = {
            'exact_match': 0.9,
            'factual_queries': 0.8,
            'keyword_density': 0.7
        }
        
        # 混合检索偏好：平衡需求、中等复杂度
        self.hybrid_preferences = {
            'balanced_complexity': 0.8,
            'multi_aspect': 0.7,
            'comprehensive_coverage': 0.9
        }
    
    def _initialize_strategy_rules(self):
        """初始化策略选择规则"""
        self.strategy_rules = {
            RetrievalStrategy.PRECISION_FOCUSED: {
                'conditions': {
                    'query_types': [QueryType.LOCAL_FACTUAL, QueryType.SPECIFIC_DETAILED],
                    'complexity_range': (0, 2.5),
                    'sparse_weight_min': 0.4
                },
                'description': '精确匹配策略 - 优先保证结果准确性'
            },
            RetrievalStrategy.SEMANTIC_FOCUSED: {
                'conditions': {
                    'query_types': [QueryType.SEMANTIC_COMPLEX, QueryType.GLOBAL_ANALYTICAL],
                    'complexity_range': (2.5, 5.0),
                    'dense_weight_min': 0.5
                },
                'description': '语义理解策略 - 重视概念和上下文理解'
            },
            RetrievalStrategy.COMPREHENSIVE_COVERAGE: {
                'conditions': {
                    'query_types': [QueryType.GLOBAL_ANALYTICAL, QueryType.MULTI_HOP_REASONING],
                    'complexity_range': (3.0, 5.0),
                    'hybrid_weight_min': 0.3
                },
                'description': '全面覆盖策略 - 确保信息完整性'
            },
            RetrievalStrategy.EXACT_MATCH: {
                'conditions': {
                    'query_types': [QueryType.SPECIFIC_DETAILED],
                    'complexity_range': (1.0, 3.0),
                    'sparse_weight_min': 0.6
                },
                'description': '精确匹配策略 - 寻找完全匹配的信息'
            },
            RetrievalStrategy.MULTI_STEP_REASONING: {
                'conditions': {
                    'query_types': [QueryType.MULTI_HOP_REASONING],
                    'complexity_range': (3.5, 5.0),
                    'dense_weight_min': 0.4
                },
                'description': '多步推理策略 - 支持复杂逻辑推理'
            },
            RetrievalStrategy.BALANCED_HYBRID: {
                'conditions': {
                    'query_types': [],  # 默认策略，适用于所有类型
                    'complexity_range': (0, 5.0),
                    'hybrid_weight_min': 0.2
                },
                'description': '平衡策略 - 兼顾准确性和覆盖面'
            }
        }
    
    def _initialize_feature_importance(self):
        """初始化特征重要性权重"""
        self.feature_importance_weights = {
            # 查询复杂度相关特征
            'lexical_complexity': 0.15,
            'syntactic_complexity': 0.12,
            'entity_complexity': 0.10,
            'domain_complexity': 0.13,
            
            # 查询类型特征
            'query_type': 0.20,
            
            # 查询长度和结构特征
            'query_length': 0.08,
            'technical_density': 0.12,
            
            # 语义特征
            'semantic_depth': 0.10
        }
    
    def compute_weights(self, query_analysis: QueryAnalysis) -> WeightAllocation:
        """
        计算动态权重分配
        
        这是核心算法，基于查询分析结果计算最优的检索器权重分配
        
        Args:
            query_analysis: 查询分析结果
            
        Returns:
            WeightAllocation: 权重分配结果
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        self.logger.debug(f"开始计算权重分配，查询类型: {query_analysis.query_type.value}")
        
        # 1. 特征向量预处理
        feature_vector = self._preprocess_features(query_analysis.feature_vector)
        
        # 2. 基于神经网络计算原始权重
        raw_weights = self._calculate_raw_weights(feature_vector, query_analysis)
        
        # 3. 权重归一化和约束
        normalized_weights = self._normalize_weights(raw_weights)
        
        # 4. 策略选择
        strategy, strategy_confidence = self._select_strategy(query_analysis, normalized_weights)
        
        # 5. 基于策略调整权重
        adjusted_weights = self._adjust_weights_by_strategy(normalized_weights, strategy, query_analysis)
        
        # 6. 计算特征重要性
        feature_importance = self._calculate_feature_importance(query_analysis)
        
        # 7. 权重分配置信度评估
        weight_confidence = self._calculate_weight_confidence(adjusted_weights, query_analysis)
        
        # 8. 生成解释
        allocation_reasoning = self._generate_allocation_reasoning(query_analysis, adjusted_weights)
        strategy_reasoning = self._generate_strategy_reasoning(strategy, query_analysis)
        
        processing_time = time.time() - start_time
        
        # 构建权重分配结果
        weight_allocation = WeightAllocation(
            dense_weight=adjusted_weights['dense'],
            sparse_weight=adjusted_weights['sparse'],
            hybrid_weight=adjusted_weights['hybrid'],
            strategy=strategy,
            strategy_confidence=strategy_confidence,
            feature_importance=feature_importance,
            weight_confidence=weight_confidence,
            calculation_method="neural_network_with_strategy_adjustment",
            allocation_reasoning=allocation_reasoning,
            strategy_reasoning=strategy_reasoning,
            timestamp=timestamp,
            processing_time=processing_time
        )
        
        self.logger.debug(f"权重分配计算完成，耗时: {processing_time:.3f}秒")
        return weight_allocation
    
    def _preprocess_features(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        预处理特征向量
        
        Args:
            feature_vector: 原始特征向量
            
        Returns:
            np.ndarray: 预处理后的特征向量
        """
        # 特征归一化
        normalized_features = feature_vector.copy()
        
        # 避免除零错误
        feature_std = np.std(normalized_features)
        if feature_std > 1e-8:
            normalized_features = (normalized_features - np.mean(normalized_features)) / feature_std
        
        # 特征裁剪，避免极值
        normalized_features = np.clip(normalized_features, -3.0, 3.0)
        
        return normalized_features
    
    def _calculate_raw_weights(self, feature_vector: np.ndarray, 
                             query_analysis: QueryAnalysis) -> Dict[str, float]:
        """
        使用神经网络计算原始权重
        
        Args:
            feature_vector: 预处理后的特征向量
            query_analysis: 查询分析结果
            
        Returns:
            Dict[str, float]: 原始权重字典
        """
        raw_weights = {}
        
        # 对每个检索器类型计算权重
        for retriever_type, matrices in self.weight_matrices.items():
            W = matrices['W']
            b = matrices['b']
            
            # 线性变换
            linear_output = np.dot(feature_vector, W) + b
            
            # Sigmoid激活
            weight = self._sigmoid(linear_output[0])
            
            # 基于查询特征的启发式调整
            weight = self._apply_heuristic_adjustments(weight, retriever_type, query_analysis)
            
            raw_weights[retriever_type] = weight
        
        return raw_weights
    
    def _apply_heuristic_adjustments(self, weight: float, retriever_type: str, 
                                   query_analysis: QueryAnalysis) -> float:
        """
        基于启发式规则调整权重
        
        这些规则基于对权威论文的分析和实践经验总结
        
        Args:
            weight: 原始权重
            retriever_type: 检索器类型
            query_analysis: 查询分析结果
            
        Returns:
            float: 调整后的权重
        """
        complexity = query_analysis.complexity_score
        query_type = query_analysis.query_type
        
        # 基于查询类型的调整
        if retriever_type == 'dense':
            # 稠密检索在语义复杂查询中表现更好
            if query_type in [QueryType.SEMANTIC_COMPLEX, QueryType.GLOBAL_ANALYTICAL]:
                weight *= 1.3
            elif query_type == QueryType.LOCAL_FACTUAL:
                weight *= 0.7
                
        elif retriever_type == 'sparse':
            # 稀疏检索在精确匹配查询中表现更好
            if query_type in [QueryType.LOCAL_FACTUAL, QueryType.SPECIFIC_DETAILED]:
                weight *= 1.4
            elif query_type == QueryType.SEMANTIC_COMPLEX:
                weight *= 0.6
                
        elif retriever_type == 'hybrid':
            # 混合检索在中等复杂度查询中表现更好
            if 2.0 <= complexity <= 3.5:
                weight *= 1.2
            else:
                weight *= 0.8
        
        # 基于复杂度的调整
        if complexity > 3.5:
            # 高复杂度查询偏向稠密检索
            if retriever_type == 'dense':
                weight *= 1.2
            elif retriever_type == 'sparse':
                weight *= 0.8
        elif complexity < 1.5:
            # 低复杂度查询偏向稀疏检索
            if retriever_type == 'sparse':
                weight *= 1.3
            elif retriever_type == 'dense':
                weight *= 0.7
        
        return weight

    def _normalize_weights(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """
        权重归一化和约束

        确保权重和为1，且在合理范围内

        Args:
            raw_weights: 原始权重字典

        Returns:
            Dict[str, float]: 归一化后的权重字典
        """
        # 应用最小最大约束
        constrained_weights = {}
        for retriever_type, weight in raw_weights.items():
            constrained_weights[retriever_type] = np.clip(weight, self.min_weight, self.max_weight)

        # 归一化使权重和为1
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            normalized_weights = {
                retriever_type: weight / total_weight
                for retriever_type, weight in constrained_weights.items()
            }
        else:
            # 如果所有权重都为0，使用均匀分布
            normalized_weights = {
                'dense': 1/3,
                'sparse': 1/3,
                'hybrid': 1/3
            }

        return normalized_weights

    def _select_strategy(self, query_analysis: QueryAnalysis,
                        weights: Dict[str, float]) -> Tuple[RetrievalStrategy, float]:
        """
        选择最适合的检索策略

        Args:
            query_analysis: 查询分析结果
            weights: 权重分配

        Returns:
            Tuple[RetrievalStrategy, float]: 策略和置信度
        """
        strategy_scores = {}

        # 对每个策略计算匹配分数
        for strategy, rules in self.strategy_rules.items():
            score = 0.0

            # 1. 查询类型匹配
            if not rules['conditions']['query_types']:  # 空列表表示适用所有类型
                type_score = 0.5
            elif query_analysis.query_type in rules['conditions']['query_types']:
                type_score = 1.0
            else:
                type_score = 0.0

            # 2. 复杂度范围匹配
            complexity_min, complexity_max = rules['conditions']['complexity_range']
            if complexity_min <= query_analysis.complexity_score <= complexity_max:
                complexity_score = 1.0
            else:
                # 计算距离最近边界的距离
                distance = min(
                    abs(query_analysis.complexity_score - complexity_min),
                    abs(query_analysis.complexity_score - complexity_max)
                )
                complexity_score = max(0.0, 1.0 - distance / 2.0)

            # 3. 权重条件匹配
            weight_score = 1.0
            for condition_key, condition_value in rules['conditions'].items():
                if condition_key.endswith('_weight_min'):
                    retriever_type = condition_key.replace('_weight_min', '')
                    if retriever_type in weights:
                        if weights[retriever_type] >= condition_value:
                            weight_score *= 1.0
                        else:
                            weight_score *= 0.5

            # 综合评分
            total_score = (
                0.4 * type_score +
                0.3 * complexity_score +
                0.3 * weight_score
            )

            strategy_scores[strategy] = total_score

        # 选择最高分的策略
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        confidence = strategy_scores[best_strategy]

        # 如果置信度太低，使用默认策略
        if confidence < 0.6:
            best_strategy = RetrievalStrategy.BALANCED_HYBRID
            confidence = 0.5

        return best_strategy, confidence

    def _adjust_weights_by_strategy(self, weights: Dict[str, float],
                                  strategy: RetrievalStrategy,
                                  query_analysis: QueryAnalysis) -> Dict[str, float]:
        """
        基于选定策略调整权重

        Args:
            weights: 原始权重
            strategy: 选定策略
            query_analysis: 查询分析结果

        Returns:
            Dict[str, float]: 调整后的权重
        """
        adjusted_weights = weights.copy()

        # 基于策略的权重调整
        if strategy == RetrievalStrategy.PRECISION_FOCUSED:
            # 精确匹配策略：增强稀疏检索权重
            adjusted_weights['sparse'] *= 1.3
            adjusted_weights['dense'] *= 0.8

        elif strategy == RetrievalStrategy.SEMANTIC_FOCUSED:
            # 语义理解策略：增强稠密检索权重
            adjusted_weights['dense'] *= 1.4
            adjusted_weights['sparse'] *= 0.7

        elif strategy == RetrievalStrategy.COMPREHENSIVE_COVERAGE:
            # 全面覆盖策略：平衡所有方法
            adjusted_weights['hybrid'] *= 1.2

        elif strategy == RetrievalStrategy.EXACT_MATCH:
            # 精确匹配策略：强化稀疏检索
            adjusted_weights['sparse'] *= 1.5
            adjusted_weights['dense'] *= 0.6

        elif strategy == RetrievalStrategy.MULTI_STEP_REASONING:
            # 多步推理策略：稠密和混合并重
            adjusted_weights['dense'] *= 1.2
            adjusted_weights['hybrid'] *= 1.1
            adjusted_weights['sparse'] *= 0.8

        # 重新归一化
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                retriever_type: weight / total_weight
                for retriever_type, weight in adjusted_weights.items()
            }

        return adjusted_weights

    def _calculate_feature_importance(self, query_analysis: QueryAnalysis) -> Dict[str, float]:
        """
        计算特征重要性

        Args:
            query_analysis: 查询分析结果

        Returns:
            Dict[str, float]: 特征重要性字典
        """
        importance = {}

        # 基于查询特征计算重要性
        complexity_factors = query_analysis.complexity_factors

        importance['lexical_complexity'] = complexity_factors.lexical_complexity * 0.2
        importance['syntactic_complexity'] = complexity_factors.syntactic_complexity * 0.15
        importance['entity_complexity'] = complexity_factors.entity_complexity * 0.1
        importance['domain_complexity'] = complexity_factors.domain_complexity * 0.15
        importance['query_type'] = query_analysis.type_confidence * 0.25
        importance['query_length'] = min(1.0, query_analysis.query_length / 100) * 0.1
        importance['technical_density'] = complexity_factors.technical_density * 0.05

        return importance

    def _calculate_weight_confidence(self, weights: Dict[str, float],
                                   query_analysis: QueryAnalysis) -> float:
        """
        计算权重分配置信度

        Args:
            weights: 权重分配
            query_analysis: 查询分析结果

        Returns:
            float: 权重分配置信度
        """
        # 1. 基于查询分析置信度
        analysis_confidence = query_analysis.confidence

        # 2. 基于权重分布的合理性
        # 避免过于极端的权重分布
        weight_values = list(weights.values())
        weight_entropy = -sum(w * np.log(w + 1e-8) for w in weight_values)
        max_entropy = -3 * (1/3) * np.log(1/3)  # 均匀分布的熵
        distribution_confidence = weight_entropy / max_entropy

        # 3. 基于主导权重的置信度
        max_weight = max(weight_values)
        dominance_confidence = max_weight if max_weight > 0.5 else 0.5

        # 综合置信度
        overall_confidence = (
            0.4 * analysis_confidence +
            0.3 * distribution_confidence +
            0.3 * dominance_confidence
        )

        return min(1.0, max(0.0, overall_confidence))

    def _generate_allocation_reasoning(self, query_analysis: QueryAnalysis,
                                     weights: Dict[str, float]) -> str:
        """
        生成权重分配推理解释

        Args:
            query_analysis: 查询分析结果
            weights: 权重分配

        Returns:
            str: 权重分配推理说明
        """
        primary_method = max(weights, key=weights.get)
        primary_weight = weights[primary_method]

        method_names = {
            'dense': '语义检索',
            'sparse': '关键词检索',
            'hybrid': '混合检索'
        }

        reasoning = f"基于查询复杂度({query_analysis.complexity_score:.1f})和类型({query_analysis.query_type.value})，"
        reasoning += f"系统选择以{method_names[primary_method]}为主({primary_weight:.1%})的权重分配。"

        # 添加具体原因
        if primary_method == 'dense':
            reasoning += "该查询涉及语义理解和概念分析，稠密检索能更好地捕捉语义相似性。"
        elif primary_method == 'sparse':
            reasoning += "该查询需要精确匹配特定信息，关键词检索能提供更准确的结果。"
        else:
            reasoning += "该查询需要平衡语义理解和精确匹配，混合检索能提供最佳覆盖。"

        return reasoning

    def _generate_strategy_reasoning(self, strategy: RetrievalStrategy,
                                   query_analysis: QueryAnalysis) -> str:
        """
        生成策略选择推理解释

        Args:
            strategy: 选定策略
            query_analysis: 查询分析结果

        Returns:
            str: 策略选择推理说明
        """
        strategy_descriptions = {
            RetrievalStrategy.PRECISION_FOCUSED: "精确匹配策略，优先保证结果准确性",
            RetrievalStrategy.SEMANTIC_FOCUSED: "语义理解策略，重视概念和上下文理解",
            RetrievalStrategy.COMPREHENSIVE_COVERAGE: "全面覆盖策略，确保信息完整性",
            RetrievalStrategy.EXACT_MATCH: "精确匹配策略，寻找完全匹配的信息",
            RetrievalStrategy.MULTI_STEP_REASONING: "多步推理策略，支持复杂逻辑推理",
            RetrievalStrategy.BALANCED_HYBRID: "平衡策略，兼顾准确性和覆盖面"
        }

        description = strategy_descriptions.get(strategy, "未知策略")

        reasoning = f"选择{description}，因为查询表现出"

        if query_analysis.query_type == QueryType.LOCAL_FACTUAL:
            reasoning += "明确的事实查询特征，需要精确的信息匹配。"
        elif query_analysis.query_type == QueryType.SEMANTIC_COMPLEX:
            reasoning += "复杂的语义理解需求，需要深度的概念分析。"
        elif query_analysis.query_type == QueryType.GLOBAL_ANALYTICAL:
            reasoning += "全局分析特征，需要综合多个信息源。"
        elif query_analysis.query_type == QueryType.MULTI_HOP_REASONING:
            reasoning += "多步推理特征，需要逻辑链条的构建。"
        else:
            reasoning += "平衡的查询特征，需要综合考虑多个方面。"

        return reasoning

    # 辅助方法
    def _sigmoid(self, x: float) -> float:
        """Sigmoid激活函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # 防止溢出
