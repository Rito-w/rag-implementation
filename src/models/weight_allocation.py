"""
Weight allocation data models.

This module defines the data structures for representing dynamic weight allocation
results and retrieval strategies.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class RetrievalStrategy(Enum):
    """Retrieval strategy types based on query analysis."""
    
    PRECISION_FOCUSED = "precision_focused"           # 精确匹配为主
    SEMANTIC_FOCUSED = "semantic_focused"             # 语义理解为主  
    COMPREHENSIVE_COVERAGE = "comprehensive_coverage" # 全面覆盖
    EXACT_MATCH = "exact_match"                       # 精确匹配
    MULTI_STEP_REASONING = "multi_step_reasoning"     # 多步推理
    BALANCED_HYBRID = "balanced_hybrid"               # 平衡策略


@dataclass
class WeightAllocation:
    """
    Dynamic weight allocation result.
    
    This class represents the output of the DynamicWeightController,
    containing weight assignments for different retrieval methods.
    """
    
    # 基础权重分配
    dense_weight: float      # 稠密检索权重 w_dense
    sparse_weight: float     # 稀疏检索权重 w_sparse  
    hybrid_weight: float     # 混合检索权重 w_hybrid
    
    # 策略信息
    strategy: RetrievalStrategy
    strategy_confidence: float
    
    # 权重计算元数据
    feature_importance: Dict[str, float]  # 特征重要性
    weight_confidence: float              # 权重分配置信度
    calculation_method: str               # 计算方法
    
    # 解释信息
    allocation_reasoning: str             # 权重分配推理
    strategy_reasoning: str               # 策略选择推理
    
    # 元数据
    timestamp: str
    processing_time: float
    
    def __post_init__(self):
        """Post-initialization validation."""
        # 验证权重和为1
        total_weight = self.dense_weight + self.sparse_weight + self.hybrid_weight
        if not abs(total_weight - 1.0) < 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        # 验证权重范围
        for weight in [self.dense_weight, self.sparse_weight, self.hybrid_weight]:
            if not 0 <= weight <= 1:
                raise ValueError("All weights must be between 0 and 1")
        
        # 验证置信度范围
        if not 0 <= self.strategy_confidence <= 1:
            raise ValueError("Strategy confidence must be between 0 and 1")
            
        if not 0 <= self.weight_confidence <= 1:
            raise ValueError("Weight confidence must be between 0 and 1")
    
    def get_primary_method(self) -> str:
        """Get the primary retrieval method based on highest weight."""
        weights = {
            'dense': self.dense_weight,
            'sparse': self.sparse_weight, 
            'hybrid': self.hybrid_weight
        }
        return max(weights, key=weights.get)
    
    def get_weight_distribution_explanation(self) -> str:
        """Generate human-readable weight distribution explanation."""
        primary = self.get_primary_method()
        primary_weight = getattr(self, f"{primary}_weight")
        
        method_names = {
            'dense': '语义检索',
            'sparse': '关键词检索',
            'hybrid': '混合检索'
        }
        
        explanation = f"主要使用{method_names[primary]} ({primary_weight:.1%})"
        
        # 添加次要方法信息
        other_methods = []
        for method, weight in [('dense', self.dense_weight), 
                              ('sparse', self.sparse_weight),
                              ('hybrid', self.hybrid_weight)]:
            if method != primary and weight > 0.1:  # 权重超过10%才显示
                other_methods.append(f"{method_names[method]} ({weight:.1%})")
        
        if other_methods:
            explanation += f"，辅助使用{', '.join(other_methods)}"
        
        return explanation
    
    def get_strategy_explanation(self) -> str:
        """Generate human-readable strategy explanation."""
        strategy_descriptions = {
            RetrievalStrategy.PRECISION_FOCUSED: "精确匹配策略 - 优先保证结果准确性",
            RetrievalStrategy.SEMANTIC_FOCUSED: "语义理解策略 - 重视概念和上下文理解",
            RetrievalStrategy.COMPREHENSIVE_COVERAGE: "全面覆盖策略 - 确保信息完整性",
            RetrievalStrategy.EXACT_MATCH: "精确匹配策略 - 寻找完全匹配的信息",
            RetrievalStrategy.MULTI_STEP_REASONING: "多步推理策略 - 支持复杂逻辑推理",
            RetrievalStrategy.BALANCED_HYBRID: "平衡策略 - 兼顾准确性和覆盖面"
        }
        
        description = strategy_descriptions.get(self.strategy, "未知策略")
        return f"检索策略: {description} (置信度: {self.strategy_confidence:.2f})"
    
    def get_detailed_explanation(self) -> Dict[str, str]:
        """Get detailed explanation for all aspects."""
        return {
            'weight_distribution': self.get_weight_distribution_explanation(),
            'strategy_selection': self.get_strategy_explanation(),
            'allocation_reasoning': self.allocation_reasoning,
            'strategy_reasoning': self.strategy_reasoning,
            'confidence_assessment': f"权重分配置信度: {self.weight_confidence:.2f}",
            'feature_importance': self._format_feature_importance()
        }
    
    def _format_feature_importance(self) -> str:
        """Format feature importance for display."""
        if not self.feature_importance:
            return "特征重要性信息不可用"
        
        # 按重要性排序
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 格式化前3个最重要的特征
        top_features = sorted_features[:3]
        formatted = []
        for feature, importance in top_features:
            formatted.append(f"{feature}: {importance:.3f}")
        
        return "主要影响因素: " + ", ".join(formatted)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dense_weight': self.dense_weight,
            'sparse_weight': self.sparse_weight,
            'hybrid_weight': self.hybrid_weight,
            'strategy': self.strategy.value,
            'strategy_confidence': self.strategy_confidence,
            'feature_importance': self.feature_importance,
            'weight_confidence': self.weight_confidence,
            'calculation_method': self.calculation_method,
            'allocation_reasoning': self.allocation_reasoning,
            'strategy_reasoning': self.strategy_reasoning,
            'timestamp': self.timestamp,
            'processing_time': self.processing_time,
            'primary_method': self.get_primary_method(),
            'weight_distribution_explanation': self.get_weight_distribution_explanation(),
            'strategy_explanation': self.get_strategy_explanation(),
            'detailed_explanation': self.get_detailed_explanation()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeightAllocation':
        """Create instance from dictionary."""
        return cls(
            dense_weight=data['dense_weight'],
            sparse_weight=data['sparse_weight'],
            hybrid_weight=data['hybrid_weight'],
            strategy=RetrievalStrategy(data['strategy']),
            strategy_confidence=data['strategy_confidence'],
            feature_importance=data['feature_importance'],
            weight_confidence=data['weight_confidence'],
            calculation_method=data['calculation_method'],
            allocation_reasoning=data['allocation_reasoning'],
            strategy_reasoning=data['strategy_reasoning'],
            timestamp=data['timestamp'],
            processing_time=data['processing_time']
        )
