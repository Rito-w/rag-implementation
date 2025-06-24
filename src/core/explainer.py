"""
决策解释器 - Decision Explainer

这个模块实现了智能自适应RAG系统的可解释性功能，是系统透明度的核心组件。

主要功能：
1. 查询理解过程解释
2. 权重分配决策解释  
3. 检索过程透明化
4. 答案来源追踪
5. 多语言解释支持

这是我们相比其他RAG系统的重要差异化优势，提供端到端的决策透明度。
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

from ..models.query_analysis import QueryAnalysis, QueryType
from ..models.weight_allocation import WeightAllocation, RetrievalStrategy
from ..models.retrieval_result import DocumentScore
from ..utils.logging import get_logger


class DecisionExplainer:
    """
    决策解释器
    
    负责生成系统决策过程的详细解释，包括查询理解、权重分配、
    检索策略选择和答案生成的全过程透明化。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化决策解释器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.logger = get_logger("DecisionExplainer")
        
        # 解释配置
        self.explanation_level = self.config.get("explanation_level", "detailed")
        self.include_confidence = self.config.get("include_confidence", True)
        self.max_explanation_length = self.config.get("max_explanation_length", 500)
        self.language = self.config.get("language", "zh")
        
        # 解释组件开关
        self.components = self.config.get("components", {
            "query_analysis": True,
            "weight_allocation": True,
            "retrieval_process": True,
            "answer_source": True,
            "confidence_assessment": True
        })
        
        # 初始化解释模板
        self._initialize_explanation_templates()
        
        # 初始化多语言支持
        self._initialize_multilingual_support()
        
        self.logger.info("决策解释器初始化完成")
    
    def _initialize_explanation_templates(self):
        """初始化解释模板"""
        self.templates = {
            'zh': {
                'query_complexity': "查询复杂度: {level} ({score:.2f}/5.0)",
                'query_type': "查询类型: {type_name} (置信度: {confidence:.1%})",
                'weight_distribution': "权重分配: {primary_method} {weight:.1%}为主",
                'strategy_selection': "检索策略: {strategy} (置信度: {confidence:.1%})",
                'document_source': "第{rank}个结果来自{method}检索 (相关度: {score:.1%})",
                'confidence_level': "系统置信度: {confidence:.1%}",
                'processing_time': "处理耗时: {time:.2f}秒"
            },
            'en': {
                'query_complexity': "Query complexity: {level} ({score:.2f}/5.0)",
                'query_type': "Query type: {type_name} (confidence: {confidence:.1%})",
                'weight_distribution': "Weight allocation: {primary_method} {weight:.1%} primary",
                'strategy_selection': "Retrieval strategy: {strategy} (confidence: {confidence:.1%})",
                'document_source': "Result #{rank} from {method} retrieval (relevance: {score:.1%})",
                'confidence_level': "System confidence: {confidence:.1%}",
                'processing_time': "Processing time: {time:.2f}s"
            }
        }
    
    def _initialize_multilingual_support(self):
        """初始化多语言支持"""
        self.translations = {
            'zh': {
                'query_types': {
                    'local_factual': '局部事实查询',
                    'global_analytical': '全局分析查询',
                    'semantic_complex': '语义复杂查询',
                    'specific_detailed': '具体详细查询',
                    'multi_hop_reasoning': '多跳推理查询'
                },
                'complexity_levels': {
                    'simple': '简单',
                    'moderate': '中等',
                    'complex': '复杂',
                    'very_complex': '非常复杂'
                },
                'retrieval_methods': {
                    'dense': '语义检索',
                    'sparse': '关键词检索',
                    'hybrid': '混合检索'
                },
                'strategies': {
                    'precision_focused': '精确匹配策略',
                    'semantic_focused': '语义理解策略',
                    'comprehensive_coverage': '全面覆盖策略',
                    'exact_match': '精确匹配策略',
                    'multi_step_reasoning': '多步推理策略',
                    'balanced_hybrid': '平衡策略'
                }
            },
            'en': {
                'query_types': {
                    'local_factual': 'Local Factual Query',
                    'global_analytical': 'Global Analytical Query',
                    'semantic_complex': 'Semantic Complex Query',
                    'specific_detailed': 'Specific Detailed Query',
                    'multi_hop_reasoning': 'Multi-hop Reasoning Query'
                },
                'complexity_levels': {
                    'simple': 'Simple',
                    'moderate': 'Moderate',
                    'complex': 'Complex',
                    'very_complex': 'Very Complex'
                },
                'retrieval_methods': {
                    'dense': 'Semantic Retrieval',
                    'sparse': 'Keyword Retrieval',
                    'hybrid': 'Hybrid Retrieval'
                },
                'strategies': {
                    'precision_focused': 'Precision-Focused Strategy',
                    'semantic_focused': 'Semantic-Focused Strategy',
                    'comprehensive_coverage': 'Comprehensive Coverage Strategy',
                    'exact_match': 'Exact Match Strategy',
                    'multi_step_reasoning': 'Multi-step Reasoning Strategy',
                    'balanced_hybrid': 'Balanced Hybrid Strategy'
                }
            }
        }
    
    def generate_explanations(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        weight_allocation: WeightAllocation,
        retrieval_results: List[DocumentScore],
        answer: str
    ) -> Dict[str, str]:
        """
        生成完整的决策解释
        
        Args:
            query: 原始查询
            query_analysis: 查询分析结果
            weight_allocation: 权重分配结果
            retrieval_results: 检索结果
            answer: 生成的答案
            
        Returns:
            Dict[str, str]: 各组件的解释字典
        """
        start_time = time.time()
        
        self.logger.debug("开始生成决策解释")
        
        explanations = {}
        
        # 1. 查询分析解释
        if self.components.get("query_analysis", True):
            explanations['query_analysis'] = self._explain_query_analysis(query_analysis)
        
        # 2. 权重分配解释
        if self.components.get("weight_allocation", True):
            explanations['weight_allocation'] = self._explain_weight_allocation(
                weight_allocation, query_analysis
            )
        
        # 3. 检索过程解释
        if self.components.get("retrieval_process", True):
            explanations['retrieval_process'] = self._explain_retrieval_process(
                retrieval_results, weight_allocation
            )
        
        # 4. 答案来源解释
        if self.components.get("answer_source", True):
            explanations['answer_source'] = self._explain_answer_source(
                answer, retrieval_results, query_analysis
            )
        
        # 5. 置信度评估解释
        if self.components.get("confidence_assessment", True):
            explanations['confidence_assessment'] = self._explain_confidence_assessment(
                query_analysis, weight_allocation, retrieval_results
            )
        
        processing_time = time.time() - start_time
        self.logger.debug(f"解释生成完成，耗时: {processing_time:.3f}秒")
        
        return explanations
    
    def _explain_query_analysis(self, query_analysis: QueryAnalysis) -> str:
        """
        解释查询分析过程
        
        Args:
            query_analysis: 查询分析结果
            
        Returns:
            str: 查询分析解释
        """
        lang = self.language
        templates = self.templates.get(lang, self.templates['zh'])
        translations = self.translations.get(lang, self.translations['zh'])
        
        # 复杂度等级
        complexity_score = query_analysis.complexity_score
        if complexity_score <= 1.5:
            complexity_level = translations['complexity_levels']['simple']
        elif complexity_score <= 2.5:
            complexity_level = translations['complexity_levels']['moderate']
        elif complexity_score <= 4.0:
            complexity_level = translations['complexity_levels']['complex']
        else:
            complexity_level = translations['complexity_levels']['very_complex']
        
        # 查询类型翻译
        query_type_name = translations['query_types'].get(
            query_analysis.query_type.value, 
            query_analysis.query_type.value
        )
        
        explanation_parts = []
        
        # 复杂度解释
        complexity_text = templates['query_complexity'].format(
            level=complexity_level,
            score=complexity_score
        )
        explanation_parts.append(complexity_text)
        
        # 类型解释
        type_text = templates['query_type'].format(
            type_name=query_type_name,
            confidence=query_analysis.type_confidence
        )
        explanation_parts.append(type_text)
        
        # 详细分析（如果是详细模式）
        if self.explanation_level == "detailed":
            if lang == 'zh':
                detail_text = f"关键特征包括{len(query_analysis.key_terms)}个关键词"
                if query_analysis.named_entities:
                    detail_text += f"和{len(query_analysis.named_entities)}个实体"
                detail_text += f"，整体分析置信度为{query_analysis.confidence:.1%}。"
            else:
                detail_text = f"Key features include {len(query_analysis.key_terms)} keywords"
                if query_analysis.named_entities:
                    detail_text += f" and {len(query_analysis.named_entities)} entities"
                detail_text += f", with overall analysis confidence of {query_analysis.confidence:.1%}."
            
            explanation_parts.append(detail_text)
        
        return " ".join(explanation_parts)
    
    def _explain_weight_allocation(self, weight_allocation: WeightAllocation, 
                                 query_analysis: QueryAnalysis) -> str:
        """
        解释权重分配决策
        
        Args:
            weight_allocation: 权重分配结果
            query_analysis: 查询分析结果
            
        Returns:
            str: 权重分配解释
        """
        lang = self.language
        templates = self.templates.get(lang, self.templates['zh'])
        translations = self.translations.get(lang, self.translations['zh'])
        
        # 主要方法
        primary_method = weight_allocation.get_primary_method()
        primary_weight = getattr(weight_allocation, f"{primary_method}_weight")
        
        # 方法名翻译
        method_name = translations['retrieval_methods'].get(primary_method, primary_method)
        
        # 策略名翻译
        strategy_name = translations['strategies'].get(
            weight_allocation.strategy.value, 
            weight_allocation.strategy.value
        )
        
        explanation_parts = []
        
        # 权重分配解释
        weight_text = templates['weight_distribution'].format(
            primary_method=method_name,
            weight=primary_weight
        )
        explanation_parts.append(weight_text)
        
        # 策略选择解释
        strategy_text = templates['strategy_selection'].format(
            strategy=strategy_name,
            confidence=weight_allocation.strategy_confidence
        )
        explanation_parts.append(strategy_text)
        
        # 详细推理（如果是详细模式）
        if self.explanation_level == "detailed":
            reasoning = weight_allocation.allocation_reasoning
            if reasoning:
                explanation_parts.append(reasoning)
        
        return " ".join(explanation_parts)
    
    def _explain_retrieval_process(self, retrieval_results: List[DocumentScore],
                                 weight_allocation: WeightAllocation) -> str:
        """
        解释检索过程
        
        Args:
            retrieval_results: 检索结果
            weight_allocation: 权重分配结果
            
        Returns:
            str: 检索过程解释
        """
        lang = self.language
        translations = self.translations.get(lang, self.translations['zh'])
        
        if not retrieval_results:
            if lang == 'zh':
                return "未找到相关文档。"
            else:
                return "No relevant documents found."
        
        explanation_parts = []
        
        # 总体统计
        total_docs = len(retrieval_results)
        avg_score = sum(doc.final_score for doc in retrieval_results) / total_docs
        
        if lang == 'zh':
            stats_text = f"检索到{total_docs}个相关文档，平均相关度为{avg_score:.1%}。"
        else:
            stats_text = f"Retrieved {total_docs} relevant documents with average relevance of {avg_score:.1%}."
        
        explanation_parts.append(stats_text)
        
        # 方法分布
        method_counts = {}
        for doc in retrieval_results:
            method = doc.retrieval_method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            if lang == 'zh':
                method_text = "来源分布: "
                method_parts = []
                for method, count in method_counts.items():
                    method_name = translations['retrieval_methods'].get(method, method)
                    method_parts.append(f"{method_name}{count}个")
                method_text += "、".join(method_parts) + "。"
            else:
                method_text = "Source distribution: "
                method_parts = []
                for method, count in method_counts.items():
                    method_name = translations['retrieval_methods'].get(method, method)
                    method_parts.append(f"{count} from {method_name}")
                method_text += ", ".join(method_parts) + "."
            
            explanation_parts.append(method_text)
        
        return " ".join(explanation_parts)
    
    def _explain_answer_source(self, answer: str, retrieval_results: List[DocumentScore],
                             query_analysis: QueryAnalysis) -> str:
        """
        解释答案来源
        
        Args:
            answer: 生成的答案
            retrieval_results: 检索结果
            query_analysis: 查询分析结果
            
        Returns:
            str: 答案来源解释
        """
        lang = self.language
        templates = self.templates.get(lang, self.templates['zh'])
        translations = self.translations.get(lang, self.translations['zh'])
        
        if not retrieval_results:
            if lang == 'zh':
                return "答案基于系统内置知识生成。"
            else:
                return "Answer generated based on built-in system knowledge."
        
        explanation_parts = []
        
        # 主要来源
        top_docs = sorted(retrieval_results, key=lambda x: x.final_score, reverse=True)[:3]
        
        if lang == 'zh':
            source_text = f"答案主要基于前{len(top_docs)}个最相关的文档："
        else:
            source_text = f"Answer primarily based on top {len(top_docs)} most relevant documents:"
        
        explanation_parts.append(source_text)
        
        # 详细来源信息
        for i, doc in enumerate(top_docs, 1):
            method_name = translations['retrieval_methods'].get(doc.retrieval_method, doc.retrieval_method)
            
            source_detail = templates['document_source'].format(
                rank=i,
                method=method_name,
                score=doc.relevance_score
            )
            explanation_parts.append(source_detail)
        
        return " ".join(explanation_parts)
    
    def _explain_confidence_assessment(self, query_analysis: QueryAnalysis,
                                     weight_allocation: WeightAllocation,
                                     retrieval_results: List[DocumentScore]) -> str:
        """
        解释置信度评估
        
        Args:
            query_analysis: 查询分析结果
            weight_allocation: 权重分配结果
            retrieval_results: 检索结果
            
        Returns:
            str: 置信度评估解释
        """
        lang = self.language
        templates = self.templates.get(lang, self.templates['zh'])
        
        # 计算综合置信度
        analysis_conf = query_analysis.confidence
        weight_conf = weight_allocation.weight_confidence
        
        if retrieval_results:
            retrieval_conf = sum(doc.relevance_score for doc in retrieval_results) / len(retrieval_results)
        else:
            retrieval_conf = 0.0
        
        overall_confidence = (analysis_conf + weight_conf + retrieval_conf) / 3
        
        explanation_parts = []
        
        # 总体置信度
        confidence_text = templates['confidence_level'].format(
            confidence=overall_confidence
        )
        explanation_parts.append(confidence_text)
        
        # 详细分解（如果是详细模式）
        if self.explanation_level == "detailed":
            if lang == 'zh':
                detail_text = (f"其中查询理解置信度{analysis_conf:.1%}，"
                             f"权重分配置信度{weight_conf:.1%}，"
                             f"检索质量置信度{retrieval_conf:.1%}。")
            else:
                detail_text = (f"Including query understanding confidence {analysis_conf:.1%}, "
                             f"weight allocation confidence {weight_conf:.1%}, "
                             f"retrieval quality confidence {retrieval_conf:.1%}.")
            
            explanation_parts.append(detail_text)
        
        return " ".join(explanation_parts)
