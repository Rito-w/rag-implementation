"""
Retrieval result data models.

This module defines the data structures for representing retrieval results,
document scores, and final system outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np


@dataclass
class DocumentScore:
    """Individual document score information."""
    
    document_id: str
    content: str
    title: Optional[str] = None
    
    # 分数信息
    dense_score: float = 0.0      # 稠密检索分数
    sparse_score: float = 0.0     # 稀疏检索分数
    hybrid_score: float = 0.0     # 混合检索分数
    final_score: float = 0.0      # 最终融合分数
    
    # 额外信息
    retrieval_method: str = ""    # 检索方法
    relevance_score: float = 0.0  # 相关性评分
    quality_score: float = 0.0    # 质量评分
    diversity_bonus: float = 0.0  # 多样性奖励
    
    # 元数据
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        # 验证分数范围
        scores = [self.dense_score, self.sparse_score, self.hybrid_score, 
                 self.final_score, self.relevance_score, self.quality_score]
        
        for score in scores:
            if score < 0:
                raise ValueError("Scores cannot be negative")
    
    def get_score_breakdown(self) -> Dict[str, float]:
        """Get detailed score breakdown."""
        return {
            'dense_score': self.dense_score,
            'sparse_score': self.sparse_score,
            'hybrid_score': self.hybrid_score,
            'final_score': self.final_score,
            'relevance_score': self.relevance_score,
            'quality_score': self.quality_score,
            'diversity_bonus': self.diversity_bonus
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'document_id': self.document_id,
            'content': self.content,
            'title': self.title,
            'dense_score': self.dense_score,
            'sparse_score': self.sparse_score,
            'hybrid_score': self.hybrid_score,
            'final_score': self.final_score,
            'retrieval_method': self.retrieval_method,
            'relevance_score': self.relevance_score,
            'quality_score': self.quality_score,
            'diversity_bonus': self.diversity_bonus,
            'source': self.source,
            'metadata': self.metadata,
            'score_breakdown': self.get_score_breakdown()
        }


@dataclass
class RetrievalResult:
    """
    Complete retrieval result from the intelligent adaptive RAG system.
    
    This class represents the final output containing the answer,
    supporting documents, and comprehensive explanations.
    """
    
    # 基础信息
    query: str
    answer: str
    
    # 检索结果
    retrieved_documents: List[DocumentScore]
    total_documents_found: int
    
    # 质量指标
    overall_confidence: float
    answer_quality_score: float
    retrieval_quality_score: float
    
    # 解释信息
    query_analysis_explanation: str
    weight_allocation_explanation: str
    retrieval_process_explanation: str
    answer_source_explanation: str
    
    # 处理元数据
    processing_time: float
    timestamp: str
    system_version: str
    
    # 详细分析结果 (可选)
    query_analysis: Optional[Any] = None      # QueryAnalysis对象
    weight_allocation: Optional[Any] = None   # WeightAllocation对象
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not 0 <= self.overall_confidence <= 1:
            raise ValueError("Overall confidence must be between 0 and 1")
            
        if not 0 <= self.answer_quality_score <= 1:
            raise ValueError("Answer quality score must be between 0 and 1")
            
        if not 0 <= self.retrieval_quality_score <= 1:
            raise ValueError("Retrieval quality score must be between 0 and 1")
    
    def get_top_documents(self, k: int = 5) -> List[DocumentScore]:
        """Get top-k retrieved documents by final score."""
        return sorted(
            self.retrieved_documents, 
            key=lambda x: x.final_score, 
            reverse=True
        )[:k]
    
    def get_documents_by_method(self, method: str) -> List[DocumentScore]:
        """Get documents retrieved by specific method."""
        return [
            doc for doc in self.retrieved_documents 
            if doc.retrieval_method == method
        ]
    
    def get_answer_sources(self) -> List[Dict[str, Any]]:
        """Get source information for answer generation."""
        sources = []
        for i, doc in enumerate(self.get_top_documents()):
            sources.append({
                'rank': i + 1,
                'document_id': doc.document_id,
                'title': doc.title,
                'relevance_score': doc.relevance_score,
                'contribution': f"第{i+1}段信息来源",
                'content_preview': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            })
        return sources
    
    def get_comprehensive_explanation(self) -> Dict[str, Any]:
        """Get comprehensive explanation of the entire process."""
        return {
            'query_understanding': {
                'explanation': self.query_analysis_explanation,
                'details': self.query_analysis.to_dict() if self.query_analysis else None
            },
            'retrieval_strategy': {
                'explanation': self.weight_allocation_explanation,
                'details': self.weight_allocation.to_dict() if self.weight_allocation else None
            },
            'retrieval_process': {
                'explanation': self.retrieval_process_explanation,
                'documents_found': self.total_documents_found,
                'top_documents': [doc.to_dict() for doc in self.get_top_documents()]
            },
            'answer_generation': {
                'explanation': self.answer_source_explanation,
                'sources': self.get_answer_sources(),
                'quality_score': self.answer_quality_score
            },
            'quality_assessment': {
                'overall_confidence': self.overall_confidence,
                'retrieval_quality': self.retrieval_quality_score,
                'answer_quality': self.answer_quality_score
            },
            'performance_metrics': {
                'processing_time': self.processing_time,
                'timestamp': self.timestamp,
                'system_version': self.system_version
            }
        }
    
    def get_user_friendly_summary(self) -> Dict[str, str]:
        """Get user-friendly summary for display."""
        top_doc = self.get_top_documents(1)[0] if self.retrieved_documents else None
        
        return {
            'answer': self.answer,
            'confidence': f"{self.overall_confidence:.1%}",
            'processing_time': f"{self.processing_time:.2f}秒",
            'documents_used': f"基于{len(self.retrieved_documents)}个相关文档",
            'top_source': top_doc.title if top_doc and top_doc.title else "多个来源",
            'query_understanding': self.query_analysis_explanation,
            'retrieval_strategy': self.weight_allocation_explanation
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'answer': self.answer,
            'retrieved_documents': [doc.to_dict() for doc in self.retrieved_documents],
            'total_documents_found': self.total_documents_found,
            'overall_confidence': self.overall_confidence,
            'answer_quality_score': self.answer_quality_score,
            'retrieval_quality_score': self.retrieval_quality_score,
            'query_analysis_explanation': self.query_analysis_explanation,
            'weight_allocation_explanation': self.weight_allocation_explanation,
            'retrieval_process_explanation': self.retrieval_process_explanation,
            'answer_source_explanation': self.answer_source_explanation,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp,
            'system_version': self.system_version,
            'top_documents': [doc.to_dict() for doc in self.get_top_documents()],
            'answer_sources': self.get_answer_sources(),
            'comprehensive_explanation': self.get_comprehensive_explanation(),
            'user_friendly_summary': self.get_user_friendly_summary()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalResult':
        """Create instance from dictionary."""
        retrieved_documents = [
            DocumentScore(**doc_data) 
            for doc_data in data['retrieved_documents']
        ]
        
        return cls(
            query=data['query'],
            answer=data['answer'],
            retrieved_documents=retrieved_documents,
            total_documents_found=data['total_documents_found'],
            overall_confidence=data['overall_confidence'],
            answer_quality_score=data['answer_quality_score'],
            retrieval_quality_score=data['retrieval_quality_score'],
            query_analysis_explanation=data['query_analysis_explanation'],
            weight_allocation_explanation=data['weight_allocation_explanation'],
            retrieval_process_explanation=data['retrieval_process_explanation'],
            answer_source_explanation=data['answer_source_explanation'],
            processing_time=data['processing_time'],
            timestamp=data['timestamp'],
            system_version=data['system_version']
        )
