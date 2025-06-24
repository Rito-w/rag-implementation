"""
查询智能分析器 - Query Intelligence Analyzer

这个模块实现了查询的深度智能分析，包括：
1. 查询复杂度建模：C(q) = α·L(q) + β·S(q) + γ·E(q) + δ·D(q)
2. 查询类型分类：5种不同的查询类型识别
3. 特征提取：生成用于权重分配的特征向量

基于对8篇权威RAG论文的分析，这是整个智能自适应系统的核心大脑。
"""

import re
import time
import spacy
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import logging

from ..models.query_analysis import QueryAnalysis, QueryType, ComplexityFactors
from ..utils.logging import get_logger


class QueryIntelligenceAnalyzer:
    """
    查询智能分析器
    
    这是整个智能自适应RAG系统的核心组件，负责深度理解查询的特征和复杂度。
    通过多维度分析，为后续的动态权重分配提供精确的特征信息。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化查询智能分析器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.logger = get_logger("QueryIntelligenceAnalyzer")
        
        # 复杂度建模权重参数 - 基于论文分析优化
        self.complexity_weights = self.config.get("complexity_weights", {
            "alpha": 0.3,   # 词汇复杂度权重
            "beta": 0.25,   # 语法复杂度权重  
            "gamma": 0.25,  # 实体复杂度权重
            "delta": 0.2    # 领域复杂度权重
        })
        
        # 分类置信度阈值
        self.classification_threshold = self.config.get("classification_threshold", 0.7)
        
        # 特征向量维度
        self.feature_dim = self.config.get("feature_dim", 768)
        
        # 初始化NLP工具
        self._initialize_nlp_tools()
        
        # 预定义的领域关键词库 - 用于领域复杂度分析
        self._initialize_domain_keywords()
        
        # 查询类型分类规则 - 基于论文分析总结
        self._initialize_classification_rules()
        
        self.logger.info("查询智能分析器初始化完成")
    
    def _initialize_nlp_tools(self):
        """初始化NLP处理工具"""
        try:
            # 加载spaCy模型 - 用于语法分析
            model_name = self.config.get("nlp_model", "en_core_web_sm")
            self.nlp = spacy.load(model_name)
            self.logger.info(f"已加载spaCy模型: {model_name}")
            
        except OSError:
            self.logger.warning("spaCy模型未找到，使用基础分析功能")
            self.nlp = None
        
        # 停用词列表
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        ])
    
    def _initialize_domain_keywords(self):
        """初始化领域关键词库"""
        self.domain_keywords = {
            'technical': {
                'ai_ml': ['machine learning', 'deep learning', 'neural network', 'algorithm', 
                         'model', 'training', 'prediction', 'classification', 'regression'],
                'computer_science': ['programming', 'software', 'database', 'system', 
                                   'architecture', 'framework', 'api', 'protocol'],
                'mathematics': ['equation', 'formula', 'theorem', 'proof', 'calculation', 
                              'statistics', 'probability', 'optimization'],
                'science': ['research', 'experiment', 'hypothesis', 'analysis', 'method', 
                           'theory', 'study', 'investigation']
            },
            'general': {
                'factual': ['what', 'who', 'when', 'where', 'which', 'definition', 'meaning'],
                'analytical': ['how', 'why', 'compare', 'analyze', 'evaluate', 'assess', 
                             'relationship', 'impact', 'effect', 'cause'],
                'procedural': ['steps', 'process', 'procedure', 'method', 'way', 'approach']
            }
        }
    
    def _initialize_classification_rules(self):
        """初始化查询类型分类规则"""
        self.classification_rules = {
            QueryType.LOCAL_FACTUAL: {
                'keywords': ['what is', 'who is', 'when', 'where', 'definition', 'meaning'],
                'patterns': [r'\bwhat\s+is\b', r'\bwho\s+is\b', r'\bdefine\b'],
                'complexity_range': (0, 2.5),
                'length_range': (5, 50)
            },
            QueryType.GLOBAL_ANALYTICAL: {
                'keywords': ['compare', 'analyze', 'relationship', 'impact', 'overview', 'summary'],
                'patterns': [r'\bcompare\b', r'\banalyze\b', r'\brelationship\b'],
                'complexity_range': (2.0, 4.5),
                'length_range': (30, 200)
            },
            QueryType.SEMANTIC_COMPLEX: {
                'keywords': ['understand', 'concept', 'theory', 'principle', 'mechanism'],
                'patterns': [r'\bhow\s+does\b', r'\bwhy\s+does\b', r'\bexplain\b'],
                'complexity_range': (2.5, 5.0),
                'length_range': (20, 150)
            },
            QueryType.SPECIFIC_DETAILED: {
                'keywords': ['specific', 'exact', 'precise', 'detailed', 'formula', 'algorithm'],
                'patterns': [r'\bexact\b', r'\bspecific\b', r'\bformula\b'],
                'complexity_range': (1.5, 4.0),
                'length_range': (15, 100)
            },
            QueryType.MULTI_HOP_REASONING: {
                'keywords': ['if', 'then', 'because', 'therefore', 'consequently', 'given that'],
                'patterns': [r'\bif\s+.*\s+then\b', r'\bgiven\s+that\b', r'\bbecause\b'],
                'complexity_range': (3.0, 5.0),
                'length_range': (40, 300)
            }
        }
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        对查询进行全面的智能分析
        
        Args:
            query: 输入查询字符串
            
        Returns:
            QueryAnalysis: 完整的查询分析结果
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        self.logger.debug(f"开始分析查询: {query[:50]}...")
        
        # 1. 查询预处理
        processed_query = self._preprocess_query(query)
        
        # 2. 复杂度分析 - 核心算法
        complexity_factors = self._analyze_complexity(processed_query)
        complexity_score = self._calculate_complexity_score(complexity_factors)
        
        # 3. 查询类型分类
        query_type, type_confidence = self._classify_query_type(processed_query, complexity_score)
        
        # 4. 关键特征提取
        key_terms = self._extract_key_terms(processed_query)
        named_entities = self._extract_named_entities(processed_query)
        semantic_concepts = self._extract_semantic_concepts(processed_query)
        
        # 5. 特征向量生成 - 用于权重分配
        feature_vector = self._generate_feature_vector(
            processed_query, complexity_factors, query_type
        )
        
        # 6. 整体置信度评估
        overall_confidence = self._calculate_overall_confidence(
            complexity_score, type_confidence, len(key_terms)
        )
        
        processing_time = time.time() - start_time
        
        # 构建分析结果
        analysis = QueryAnalysis(
            original_query=query,
            processed_query=processed_query,
            query_length=len(query),
            complexity_score=complexity_score,
            complexity_factors=complexity_factors,
            query_type=query_type,
            type_confidence=type_confidence,
            key_terms=key_terms,
            named_entities=named_entities,
            semantic_concepts=semantic_concepts,
            feature_vector=feature_vector,
            analysis_timestamp=timestamp,
            processing_time=processing_time,
            confidence=overall_confidence
        )
        
        self.logger.debug(f"查询分析完成，耗时: {processing_time:.3f}秒")
        return analysis
    
    def _preprocess_query(self, query: str) -> str:
        """
        查询预处理
        
        Args:
            query: 原始查询
            
        Returns:
            str: 预处理后的查询
        """
        # 基础清理
        processed = query.strip().lower()
        
        # 移除多余空格
        processed = re.sub(r'\s+', ' ', processed)
        
        # 移除特殊字符（保留基本标点）
        processed = re.sub(r'[^\w\s\?\!\.\,\;\:]', '', processed)
        
        return processed
    
    def _analyze_complexity(self, query: str) -> ComplexityFactors:
        """
        分析查询复杂度的各个维度
        
        这是核心算法，实现论文中的复杂度建模：
        C(q) = α·L(q) + β·S(q) + γ·E(q) + δ·D(q)
        
        Args:
            query: 预处理后的查询
            
        Returns:
            ComplexityFactors: 复杂度分析结果
        """
        # 1. 词汇复杂度分析 L(q)
        lexical_complexity = self._analyze_lexical_complexity(query)
        
        # 2. 语法复杂度分析 S(q)  
        syntactic_complexity = self._analyze_syntactic_complexity(query)
        
        # 3. 实体复杂度分析 E(q)
        entity_complexity = self._analyze_entity_complexity(query)
        
        # 4. 领域复杂度分析 D(q)
        domain_complexity = self._analyze_domain_complexity(query)
        
        return ComplexityFactors(
            lexical_complexity=lexical_complexity,
            syntactic_complexity=syntactic_complexity,
            entity_complexity=entity_complexity,
            domain_complexity=domain_complexity,
            word_frequency_score=self._calculate_word_frequency_score(query),
            semantic_depth_score=self._calculate_semantic_depth_score(query),
            technical_density=self._calculate_technical_density(query),
            dependency_tree_depth=self._calculate_dependency_depth(query),
            named_entity_count=len(self._extract_named_entities(query)),
            domain_specificity=self._calculate_domain_specificity(query)
        )
    
    def _analyze_lexical_complexity(self, query: str) -> float:
        """
        分析词汇复杂度 L(q)
        
        考虑因素：
        - 词汇多样性
        - 平均词长
        - 罕见词比例
        - 专业术语密度
        """
        words = query.split()
        if not words:
            return 0.0
        
        # 词汇多样性 (Type-Token Ratio)
        unique_words = set(words)
        diversity_score = len(unique_words) / len(words)
        
        # 平均词长
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_score = min(1.0, avg_word_length / 8.0)  # 归一化到[0,1]
        
        # 罕见词比例 (简化实现)
        rare_words = [word for word in words if len(word) > 7 and word not in self.stop_words]
        rare_word_ratio = len(rare_words) / len(words)
        
        # 专业术语密度
        technical_terms = self._count_technical_terms(query)
        technical_density = technical_terms / len(words)
        
        # 综合词汇复杂度
        lexical_complexity = (
            0.3 * diversity_score +
            0.25 * length_score +
            0.25 * rare_word_ratio +
            0.2 * technical_density
        )
        
        return min(1.0, lexical_complexity)
    
    def _analyze_syntactic_complexity(self, query: str) -> float:
        """
        分析语法复杂度 S(q)
        
        考虑因素：
        - 句子长度
        - 从句数量
        - 依存关系深度
        - 语法结构复杂性
        """
        if not self.nlp:
            # 简化的语法复杂度分析
            words = query.split()
            sentence_length_score = min(1.0, len(words) / 30.0)
            
            # 简单的从句检测
            subordinate_markers = ['that', 'which', 'who', 'when', 'where', 'because', 'if']
            subordinate_count = sum(1 for marker in subordinate_markers if marker in query)
            subordinate_score = min(1.0, subordinate_count / 3.0)
            
            return (sentence_length_score + subordinate_score) / 2
        
        # 使用spaCy进行详细语法分析
        doc = self.nlp(query)
        
        # 依存关系深度
        max_depth = 0
        for token in doc:
            depth = self._calculate_token_depth(token)
            max_depth = max(max_depth, depth)
        
        depth_score = min(1.0, max_depth / 8.0)
        
        # 句子复杂性
        sentence_count = len(list(doc.sents))
        avg_sentence_length = len(doc) / max(1, sentence_count)
        length_score = min(1.0, avg_sentence_length / 25.0)
        
        # 语法标签多样性
        pos_tags = set(token.pos_ for token in doc)
        pos_diversity = len(pos_tags) / 17  # 17是常见POS标签数量
        
        syntactic_complexity = (
            0.4 * depth_score +
            0.3 * length_score +
            0.3 * pos_diversity
        )
        
        return min(1.0, syntactic_complexity)
    
    def _analyze_entity_complexity(self, query: str) -> float:
        """
        分析实体复杂度 E(q)
        
        考虑因素：
        - 命名实体数量
        - 实体类型多样性
        - 实体关系复杂性
        """
        entities = self._extract_named_entities(query)
        
        if not entities:
            return 0.0
        
        # 实体数量得分
        entity_count_score = min(1.0, len(entities) / 5.0)
        
        # 实体类型多样性（简化实现）
        entity_types = set()
        for entity in entities:
            if any(keyword in entity.lower() for keyword in ['person', 'organization', 'location']):
                entity_types.add('named_entity')
            elif any(keyword in entity.lower() for keyword in ['algorithm', 'method', 'technique']):
                entity_types.add('technical_concept')
            else:
                entity_types.add('general_concept')
        
        type_diversity_score = len(entity_types) / 3.0
        
        entity_complexity = (entity_count_score + type_diversity_score) / 2
        return min(1.0, entity_complexity)
    
    def _analyze_domain_complexity(self, query: str) -> float:
        """
        分析领域复杂度 D(q)
        
        考虑因素：
        - 领域特异性
        - 专业术语密度
        - 跨领域复杂性
        """
        domain_scores = {}
        
        # 计算各领域的匹配度
        for domain_category, domains in self.domain_keywords.items():
            category_score = 0
            for domain_name, keywords in domains.items():
                keyword_matches = sum(1 for keyword in keywords if keyword in query.lower())
                domain_score = keyword_matches / len(keywords)
                category_score = max(category_score, domain_score)
            domain_scores[domain_category] = category_score
        
        # 领域特异性 - 最高匹配度
        domain_specificity = max(domain_scores.values()) if domain_scores else 0.0
        
        # 跨领域复杂性 - 多个领域的匹配
        cross_domain_score = sum(1 for score in domain_scores.values() if score > 0.1) / len(domain_scores)
        
        domain_complexity = (0.7 * domain_specificity + 0.3 * cross_domain_score)
        return min(1.0, domain_complexity)

    def _calculate_complexity_score(self, factors: ComplexityFactors) -> float:
        """
        计算总体复杂度评分

        使用论文中的公式：C(q) = α·L(q) + β·S(q) + γ·E(q) + δ·D(q)

        Args:
            factors: 复杂度因子

        Returns:
            float: 总体复杂度评分 (0-5)
        """
        alpha = self.complexity_weights["alpha"]
        beta = self.complexity_weights["beta"]
        gamma = self.complexity_weights["gamma"]
        delta = self.complexity_weights["delta"]

        # 计算加权复杂度 (0-1范围)
        weighted_complexity = (
            alpha * factors.lexical_complexity +
            beta * factors.syntactic_complexity +
            gamma * factors.entity_complexity +
            delta * factors.domain_complexity
        )

        # 转换到0-5评分范围
        complexity_score = weighted_complexity * 5.0

        return min(5.0, max(0.0, complexity_score))

    def _classify_query_type(self, query: str, complexity_score: float) -> Tuple[QueryType, float]:
        """
        分类查询类型

        基于规则和复杂度的混合分类方法

        Args:
            query: 预处理后的查询
            complexity_score: 复杂度评分

        Returns:
            Tuple[QueryType, float]: 查询类型和置信度
        """
        type_scores = {}

        # 对每种查询类型计算匹配分数
        for query_type, rules in self.classification_rules.items():
            score = 0.0

            # 1. 关键词匹配
            keyword_matches = sum(1 for keyword in rules['keywords'] if keyword in query)
            keyword_score = keyword_matches / len(rules['keywords'])

            # 2. 正则模式匹配
            pattern_matches = sum(1 for pattern in rules['patterns'] if re.search(pattern, query))
            pattern_score = pattern_matches / len(rules['patterns']) if rules['patterns'] else 0

            # 3. 复杂度范围匹配
            complexity_min, complexity_max = rules['complexity_range']
            if complexity_min <= complexity_score <= complexity_max:
                complexity_score_normalized = 1.0
            else:
                # 计算距离最近边界的距离
                distance = min(abs(complexity_score - complexity_min),
                             abs(complexity_score - complexity_max))
                complexity_score_normalized = max(0.0, 1.0 - distance / 2.0)

            # 4. 长度范围匹配
            length_min, length_max = rules['length_range']
            query_length = len(query)
            if length_min <= query_length <= length_max:
                length_score = 1.0
            else:
                distance = min(abs(query_length - length_min),
                             abs(query_length - length_max))
                length_score = max(0.0, 1.0 - distance / 50.0)

            # 综合评分
            total_score = (
                0.4 * keyword_score +
                0.3 * pattern_score +
                0.2 * complexity_score_normalized +
                0.1 * length_score
            )

            type_scores[query_type] = total_score

        # 选择最高分的类型
        best_type = max(type_scores, key=type_scores.get)
        confidence = type_scores[best_type]

        # 如果置信度太低，使用默认类型
        if confidence < self.classification_threshold:
            best_type = QueryType.BALANCED_HYBRID if hasattr(QueryType, 'BALANCED_HYBRID') else QueryType.LOCAL_FACTUAL
            confidence = 0.5

        return best_type, confidence

    def _extract_key_terms(self, query: str) -> List[str]:
        """
        提取查询中的关键词

        Args:
            query: 预处理后的查询

        Returns:
            List[str]: 关键词列表
        """
        words = query.split()

        # 移除停用词
        content_words = [word for word in words if word not in self.stop_words and len(word) > 2]

        # 词频统计
        word_freq = Counter(content_words)

        # 选择高频词和长词作为关键词
        key_terms = []
        for word, freq in word_freq.most_common():
            if len(word) > 3 or freq > 1:
                key_terms.append(word)
            if len(key_terms) >= 10:  # 限制关键词数量
                break

        return key_terms

    def _extract_named_entities(self, query: str) -> List[str]:
        """
        提取命名实体

        Args:
            query: 预处理后的查询

        Returns:
            List[str]: 命名实体列表
        """
        entities = []

        if self.nlp:
            # 使用spaCy提取命名实体
            doc = self.nlp(query)
            entities = [ent.text for ent in doc.ents]
        else:
            # 简化的实体提取 - 查找大写开头的词组
            words = query.split()
            current_entity = []

            for word in words:
                if word[0].isupper() and word not in self.stop_words:
                    current_entity.append(word)
                else:
                    if current_entity:
                        entities.append(' '.join(current_entity))
                        current_entity = []

            # 添加最后一个实体
            if current_entity:
                entities.append(' '.join(current_entity))

        return entities

    def _extract_semantic_concepts(self, query: str) -> List[str]:
        """
        提取语义概念

        Args:
            query: 预处理后的查询

        Returns:
            List[str]: 语义概念列表
        """
        concepts = []

        # 基于领域关键词提取概念
        for domain_category, domains in self.domain_keywords.items():
            for domain_name, keywords in domains.items():
                for keyword in keywords:
                    if keyword in query.lower():
                        concepts.append(keyword)

        # 去重并限制数量
        concepts = list(set(concepts))[:8]

        return concepts

    def _generate_feature_vector(self, query: str, factors: ComplexityFactors,
                                query_type: QueryType) -> np.ndarray:
        """
        生成特征向量用于权重分配

        Args:
            query: 预处理后的查询
            factors: 复杂度因子
            query_type: 查询类型

        Returns:
            np.ndarray: 特征向量
        """
        # 基础特征
        basic_features = [
            len(query.split()),  # 词数
            len(query),          # 字符数
            factors.lexical_complexity,
            factors.syntactic_complexity,
            factors.entity_complexity,
            factors.domain_complexity,
            factors.word_frequency_score,
            factors.semantic_depth_score,
            factors.technical_density,
            float(factors.dependency_tree_depth),
            float(factors.named_entity_count),
            factors.domain_specificity
        ]

        # 查询类型one-hot编码
        type_features = [0.0] * 5
        type_mapping = {
            QueryType.LOCAL_FACTUAL: 0,
            QueryType.GLOBAL_ANALYTICAL: 1,
            QueryType.SEMANTIC_COMPLEX: 2,
            QueryType.SPECIFIC_DETAILED: 3,
            QueryType.MULTI_HOP_REASONING: 4
        }
        if query_type in type_mapping:
            type_features[type_mapping[query_type]] = 1.0

        # 组合所有特征
        all_features = basic_features + type_features

        # 如果需要更高维度，用零填充
        if len(all_features) < self.feature_dim:
            padding = [0.0] * (self.feature_dim - len(all_features))
            all_features.extend(padding)

        # 截断到指定维度
        feature_vector = np.array(all_features[:self.feature_dim], dtype=np.float32)

        return feature_vector

    def _calculate_overall_confidence(self, complexity_score: float,
                                    type_confidence: float, key_terms_count: int) -> float:
        """
        计算整体分析置信度

        Args:
            complexity_score: 复杂度评分
            type_confidence: 类型分类置信度
            key_terms_count: 关键词数量

        Returns:
            float: 整体置信度
        """
        # 复杂度置信度 - 基于评分的合理性
        complexity_confidence = 1.0 - abs(complexity_score - 2.5) / 2.5

        # 关键词置信度 - 基于关键词数量
        terms_confidence = min(1.0, key_terms_count / 5.0)

        # 综合置信度
        overall_confidence = (
            0.4 * type_confidence +
            0.3 * complexity_confidence +
            0.3 * terms_confidence
        )

        return min(1.0, max(0.0, overall_confidence))

    # 辅助方法
    def _count_technical_terms(self, query: str) -> int:
        """统计技术术语数量"""
        technical_count = 0
        for domain_category, domains in self.domain_keywords.items():
            if domain_category == 'technical':
                for domain_name, keywords in domains.items():
                    technical_count += sum(1 for keyword in keywords if keyword in query.lower())
        return technical_count

    def _calculate_token_depth(self, token) -> int:
        """计算token在依存树中的深度"""
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
            if depth > 20:  # 防止无限循环
                break
        return depth

    def _calculate_word_frequency_score(self, query: str) -> float:
        """计算词频逆向指标"""
        words = query.split()
        if not words:
            return 0.0

        # 简化实现：基于词长估算罕见度
        rare_words = [word for word in words if len(word) > 6]
        return len(rare_words) / len(words)

    def _calculate_semantic_depth_score(self, query: str) -> float:
        """计算语义深度评分"""
        # 简化实现：基于抽象概念词汇
        abstract_indicators = ['concept', 'theory', 'principle', 'mechanism', 'relationship']
        abstract_count = sum(1 for indicator in abstract_indicators if indicator in query.lower())
        return min(1.0, abstract_count / 3.0)

    def _calculate_technical_density(self, query: str) -> float:
        """计算专业术语密度"""
        words = query.split()
        if not words:
            return 0.0

        technical_count = self._count_technical_terms(query)
        return technical_count / len(words)

    def _calculate_dependency_depth(self, query: str) -> int:
        """计算依存关系深度"""
        if not self.nlp:
            # 简化实现：基于句子长度估算
            return min(8, len(query.split()) // 5)

        doc = self.nlp(query)
        max_depth = 0
        for token in doc:
            depth = self._calculate_token_depth(token)
            max_depth = max(max_depth, depth)

        return max_depth

    def _calculate_domain_specificity(self, query: str) -> float:
        """计算领域特异性"""
        domain_matches = 0
        total_keywords = 0

        for domain_category, domains in self.domain_keywords.items():
            for domain_name, keywords in domains.items():
                total_keywords += len(keywords)
                domain_matches += sum(1 for keyword in keywords if keyword in query.lower())

        return domain_matches / max(1, total_keywords)
