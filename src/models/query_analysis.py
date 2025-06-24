"""
Query analysis data models.

This module defines the data structures for representing query analysis results,
including query types, complexity factors, and analysis metadata.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class QueryType(Enum):
    """Query type classification based on analysis of 8 authoritative RAG papers."""
    
    LOCAL_FACTUAL = "local_factual"           # 局部事实查询
    GLOBAL_ANALYTICAL = "global_analytical"   # 全局分析查询  
    SEMANTIC_COMPLEX = "semantic_complex"     # 语义复杂查询
    SPECIFIC_DETAILED = "specific_detailed"   # 具体详细查询
    MULTI_HOP_REASONING = "multi_hop_reasoning"  # 多跳推理查询


@dataclass
class ComplexityFactors:
    """Query complexity analysis factors."""
    
    lexical_complexity: float      # 词汇复杂度 L(q)
    syntactic_complexity: float    # 语法复杂度 S(q)
    entity_complexity: float       # 实体复杂度 E(q)
    domain_complexity: float       # 领域复杂度 D(q)
    
    # 详细分析
    word_frequency_score: float    # 词频逆向指标
    semantic_depth_score: float    # 语义深度
    technical_density: float       # 专业术语密度
    dependency_tree_depth: int     # 依存解析树深度
    named_entity_count: int        # 命名实体数量
    domain_specificity: float      # 领域特异性
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'lexical_complexity': self.lexical_complexity,
            'syntactic_complexity': self.syntactic_complexity,
            'entity_complexity': self.entity_complexity,
            'domain_complexity': self.domain_complexity,
            'word_frequency_score': self.word_frequency_score,
            'semantic_depth_score': self.semantic_depth_score,
            'technical_density': self.technical_density,
            'dependency_tree_depth': self.dependency_tree_depth,
            'named_entity_count': self.named_entity_count,
            'domain_specificity': self.domain_specificity
        }


@dataclass
class QueryAnalysis:
    """
    Comprehensive query analysis result.
    
    This class represents the output of the QueryIntelligenceAnalyzer,
    containing all information needed for adaptive retrieval strategy selection.
    """
    
    # 基础信息
    original_query: str
    processed_query: str
    query_length: int
    
    # 复杂度分析
    complexity_score: float        # 总体复杂度评分 C(q)
    complexity_factors: ComplexityFactors
    
    # 查询分类
    query_type: QueryType
    type_confidence: float         # 分类置信度
    
    # 关键特征
    key_terms: List[str]          # 关键词
    named_entities: List[str]     # 命名实体
    semantic_concepts: List[str]  # 语义概念
    
    # 特征向量 (用于权重计算)
    feature_vector: np.ndarray
    
    # 元数据
    analysis_timestamp: str
    processing_time: float        # 分析耗时 (秒)
    confidence: float            # 整体分析置信度
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not 0 <= self.complexity_score <= 5:
            raise ValueError("Complexity score must be between 0 and 5")
        
        if not 0 <= self.type_confidence <= 1:
            raise ValueError("Type confidence must be between 0 and 1")
            
        if not 0 <= self.confidence <= 1:
            raise ValueError("Overall confidence must be between 0 and 1")
    
    def get_complexity_explanation(self) -> str:
        """Generate human-readable complexity explanation."""
        if self.complexity_score <= 1.5:
            level = "简单"
        elif self.complexity_score <= 3.0:
            level = "中等"
        elif self.complexity_score <= 4.0:
            level = "复杂"
        else:
            level = "非常复杂"
        
        return f"查询复杂度: {level} ({self.complexity_score:.2f}/5.0)"
    
    def get_type_explanation(self) -> str:
        """Generate human-readable query type explanation."""
        type_descriptions = {
            QueryType.LOCAL_FACTUAL: "局部事实查询 - 寻找具体的事实信息",
            QueryType.GLOBAL_ANALYTICAL: "全局分析查询 - 需要综合分析多个信息源",
            QueryType.SEMANTIC_COMPLEX: "语义复杂查询 - 涉及复杂的概念理解",
            QueryType.SPECIFIC_DETAILED: "具体详细查询 - 需要精确的详细信息",
            QueryType.MULTI_HOP_REASONING: "多跳推理查询 - 需要多步逻辑推理"
        }
        
        description = type_descriptions.get(self.query_type, "未知类型")
        return f"查询类型: {description} (置信度: {self.type_confidence:.2f})"
    
    def get_recommended_strategy(self) -> str:
        """Get recommended retrieval strategy based on analysis."""
        if self.query_type == QueryType.LOCAL_FACTUAL:
            return "precision_focused"  # 精确匹配为主
        elif self.query_type == QueryType.GLOBAL_ANALYTICAL:
            return "comprehensive_coverage"  # 全面覆盖
        elif self.query_type == QueryType.SEMANTIC_COMPLEX:
            return "semantic_focused"  # 语义理解为主
        elif self.query_type == QueryType.SPECIFIC_DETAILED:
            return "exact_match"  # 精确匹配
        elif self.query_type == QueryType.MULTI_HOP_REASONING:
            return "multi_step_reasoning"  # 多步推理
        else:
            return "balanced_hybrid"  # 平衡策略
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'original_query': self.original_query,
            'processed_query': self.processed_query,
            'query_length': self.query_length,
            'complexity_score': self.complexity_score,
            'complexity_factors': self.complexity_factors.to_dict(),
            'query_type': self.query_type.value,
            'type_confidence': self.type_confidence,
            'key_terms': self.key_terms,
            'named_entities': self.named_entities,
            'semantic_concepts': self.semantic_concepts,
            'feature_vector': self.feature_vector.tolist(),
            'analysis_timestamp': self.analysis_timestamp,
            'processing_time': self.processing_time,
            'confidence': self.confidence,
            'complexity_explanation': self.get_complexity_explanation(),
            'type_explanation': self.get_type_explanation(),
            'recommended_strategy': self.get_recommended_strategy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryAnalysis':
        """Create instance from dictionary."""
        complexity_factors = ComplexityFactors(**data['complexity_factors'])
        
        return cls(
            original_query=data['original_query'],
            processed_query=data['processed_query'],
            query_length=data['query_length'],
            complexity_score=data['complexity_score'],
            complexity_factors=complexity_factors,
            query_type=QueryType(data['query_type']),
            type_confidence=data['type_confidence'],
            key_terms=data['key_terms'],
            named_entities=data['named_entities'],
            semantic_concepts=data['semantic_concepts'],
            feature_vector=np.array(data['feature_vector']),
            analysis_timestamp=data['analysis_timestamp'],
            processing_time=data['processing_time'],
            confidence=data['confidence']
        )
