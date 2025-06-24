"""
Intelligent Adaptive RAG System

A novel RAG system that implements query-aware adaptive retrieval
with comprehensive explainability.
"""

__version__ = "0.1.0"
__author__ = "Intelligent RAG Research Team"

from .core.intelligent_adapter import IntelligentAdaptiveRAG
from .models.query_analysis import QueryAnalysis, QueryType
from .models.weight_allocation import WeightAllocation
from .models.retrieval_result import RetrievalResult

__all__ = [
    "IntelligentAdaptiveRAG",
    "QueryAnalysis", 
    "QueryType",
    "WeightAllocation",
    "RetrievalResult"
]
