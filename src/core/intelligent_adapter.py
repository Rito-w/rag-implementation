"""
Intelligent Adaptive RAG System - Main Entry Point

This module implements the core IntelligentAdaptiveRAG class that orchestrates
the entire intelligent adaptive retrieval process. Inspired by GasketRAG's
modular design but with intelligent query-aware adaptations.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path

from ..models.query_analysis import QueryAnalysis
from ..models.weight_allocation import WeightAllocation
from ..models.retrieval_result import RetrievalResult, DocumentScore
from ..utils.config import Config
from ..utils.logging import setup_logger

from .query_analyzer import QueryIntelligenceAnalyzer
from .weight_controller import DynamicWeightController
from .fusion_engine import IntelligentFusionEngine
from .explainer import DecisionExplainer

# Import retrievers
from ..retrievers.dense_retriever import DenseRetriever
from ..retrievers.sparse_retriever import SparseRetriever
from ..retrievers.hybrid_retriever import HybridRetriever


class IntelligentAdaptiveRAG:
    """
    Intelligent Adaptive RAG System
    
    Main system class that implements query-aware adaptive retrieval with
    comprehensive explainability. Inspired by GasketRAG's modular approach
    but with intelligent adaptations based on query characteristics.
    
    Key Features:
    - Query complexity analysis and type classification
    - Dynamic weight allocation based on query features
    - Intelligent fusion of multiple retrieval methods
    - Comprehensive explainability at every step
    - Modular design for easy extension and maintenance
    """
    
    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize the Intelligent Adaptive RAG system.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional configuration parameters
        """
        # Load configuration
        self.config = Config(config_path, **kwargs)
        
        # Setup logging
        self.logger = setup_logger(
            name="IntelligentAdaptiveRAG",
            level=self.config.get("logging.level", "INFO")
        )
        
        # System metadata
        self.version = "0.1.0"
        self.logger.info(f"Initializing Intelligent Adaptive RAG v{self.version}")
        
        # Initialize core components (Intelligent Adaptation Layer)
        self._initialize_intelligence_layer()
        
        # Initialize retrieval components (existing components, no retraining needed)
        self._initialize_retrieval_components()
        
        # System state
        self.is_initialized = True
        self.stats = {
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        self.logger.info("Intelligent Adaptive RAG system initialized successfully")
    
    def _initialize_intelligence_layer(self):
        """Initialize the intelligent adaptation layer components."""
        self.logger.info("Initializing intelligent adaptation layer...")
        
        # Core intelligence components
        self.query_analyzer = QueryIntelligenceAnalyzer(
            config=self.config.get("query_analyzer", {})
        )
        
        self.weight_controller = DynamicWeightController(
            config=self.config.get("weight_controller", {})
        )
        
        self.fusion_engine = IntelligentFusionEngine(
            config=self.config.get("fusion_engine", {})
        )
        
        self.explainer = DecisionExplainer(
            config=self.config.get("explainer", {})
        )
        
        self.logger.info("Intelligent adaptation layer initialized")
    
    def _initialize_retrieval_components(self):
        """Initialize retrieval components (existing, no retraining needed)."""
        self.logger.info("Initializing retrieval components...")
        
        # Existing retrieval components (inspired by GasketRAG approach)
        self.retrievers = {
            'dense': DenseRetriever(
                config=self.config.get("retrievers.dense", {})
            ),
            'sparse': SparseRetriever(
                config=self.config.get("retrievers.sparse", {})
            ),
            'hybrid': HybridRetriever(
                config=self.config.get("retrievers.hybrid", {})
            )
        }
        
        self.logger.info("Retrieval components initialized")
    
    def process_query(
        self, 
        query: str,
        return_detailed: bool = True,
        **kwargs
    ) -> RetrievalResult:
        """
        Process a query through the intelligent adaptive RAG pipeline.
        
        Args:
            query: Input query string
            return_detailed: Whether to return detailed analysis results
            **kwargs: Additional processing parameters
            
        Returns:
            RetrievalResult: Complete result with answer and explanations
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Step 1: Intelligent Query Analysis
            self.logger.debug("Step 1: Analyzing query...")
            query_analysis = self.query_analyzer.analyze(query)
            
            # Step 2: Dynamic Weight Allocation
            self.logger.debug("Step 2: Computing dynamic weights...")
            weight_allocation = self.weight_controller.compute_weights(query_analysis)
            
            # Step 3: Adaptive Retrieval
            self.logger.debug("Step 3: Performing adaptive retrieval...")
            retrieval_results = self._perform_adaptive_retrieval(
                query, query_analysis, weight_allocation
            )
            
            # Step 4: Intelligent Fusion
            self.logger.debug("Step 4: Fusing retrieval results...")
            fused_results = self.fusion_engine.fuse_results(
                retrieval_results, query_analysis, weight_allocation
            )
            
            # Step 5: Answer Generation
            self.logger.debug("Step 5: Generating answer...")
            answer = self._generate_answer(query, fused_results, query_analysis)
            
            # Step 6: Generate Explanations
            self.logger.debug("Step 6: Generating explanations...")
            explanations = self.explainer.generate_explanations(
                query, query_analysis, weight_allocation, fused_results, answer
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create final result
            result = RetrievalResult(
                query=query,
                answer=answer,
                retrieved_documents=fused_results,
                total_documents_found=len(fused_results),
                overall_confidence=self._calculate_overall_confidence(
                    query_analysis, weight_allocation, fused_results
                ),
                answer_quality_score=self._assess_answer_quality(answer, fused_results),
                retrieval_quality_score=self._assess_retrieval_quality(fused_results),
                query_analysis_explanation=explanations['query_analysis'],
                weight_allocation_explanation=explanations['weight_allocation'],
                retrieval_process_explanation=explanations['retrieval_process'],
                answer_source_explanation=explanations['answer_source'],
                processing_time=processing_time,
                timestamp=timestamp,
                system_version=self.version,
                query_analysis=query_analysis if return_detailed else None,
                weight_allocation=weight_allocation if return_detailed else None
            )
            
            # Update statistics
            self._update_stats(processing_time)
            
            self.logger.info(f"Query processed successfully in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _perform_adaptive_retrieval(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        weight_allocation: WeightAllocation
    ) -> Dict[str, List[DocumentScore]]:
        """Perform adaptive retrieval using multiple methods."""
        retrieval_results = {}
        
        # Retrieve using each method based on allocated weights
        for method, weight in [
            ('dense', weight_allocation.dense_weight),
            ('sparse', weight_allocation.sparse_weight),
            ('hybrid', weight_allocation.hybrid_weight)
        ]:
            if weight > 0.01:  # Only retrieve if weight is significant
                k = max(1, int(weight * self.config.get("retrieval.max_docs", 20)))
                results = self.retrievers[method].retrieve(query, k=k)
                retrieval_results[method] = results
        
        return retrieval_results
    
    def _generate_answer(
        self,
        query: str,
        documents: List[DocumentScore],
        query_analysis: QueryAnalysis
    ) -> str:
        """Generate answer based on retrieved documents."""
        # This is a placeholder - in real implementation, this would use an LLM
        if not documents:
            return "抱歉，没有找到相关信息来回答您的问题。"
        
        # Simple answer generation based on top documents
        top_docs = sorted(documents, key=lambda x: x.final_score, reverse=True)[:3]
        
        answer_parts = []
        for i, doc in enumerate(top_docs):
            # Extract relevant content (simplified)
            content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            answer_parts.append(f"根据相关文档{i+1}：{content_preview}")
        
        return "\n\n".join(answer_parts)
    
    def _calculate_overall_confidence(
        self,
        query_analysis: QueryAnalysis,
        weight_allocation: WeightAllocation,
        documents: List[DocumentScore]
    ) -> float:
        """Calculate overall system confidence."""
        if not documents:
            return 0.0
        
        # Combine various confidence factors
        analysis_confidence = query_analysis.confidence
        weight_confidence = weight_allocation.weight_confidence
        retrieval_confidence = sum(doc.relevance_score for doc in documents) / len(documents)
        
        # Weighted average
        overall_confidence = (
            0.3 * analysis_confidence +
            0.3 * weight_confidence +
            0.4 * retrieval_confidence
        )
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _assess_answer_quality(self, answer: str, documents: List[DocumentScore]) -> float:
        """Assess the quality of generated answer."""
        if not answer or not documents:
            return 0.0
        
        # Simple quality assessment (in real implementation, use more sophisticated metrics)
        length_score = min(1.0, len(answer) / 500)  # Prefer longer, more detailed answers
        coverage_score = min(1.0, len(documents) / 5)  # More documents = better coverage
        
        return (length_score + coverage_score) / 2
    
    def _assess_retrieval_quality(self, documents: List[DocumentScore]) -> float:
        """Assess the quality of retrieval results."""
        if not documents:
            return 0.0
        
        # Average relevance score
        avg_relevance = sum(doc.relevance_score for doc in documents) / len(documents)
        
        # Diversity bonus (simplified)
        diversity_score = min(1.0, len(set(doc.source for doc in documents if doc.source)) / 3)
        
        return (avg_relevance + diversity_score) / 2
    
    def _update_stats(self, processing_time: float):
        """Update system statistics."""
        self.stats['queries_processed'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['average_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['queries_processed']
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        return {
            'version': self.version,
            'is_initialized': self.is_initialized,
            'queries_processed': self.stats['queries_processed'],
            'total_processing_time': self.stats['total_processing_time'],
            'average_processing_time': self.stats['average_processing_time'],
            'components_status': {
                'query_analyzer': hasattr(self, 'query_analyzer'),
                'weight_controller': hasattr(self, 'weight_controller'),
                'fusion_engine': hasattr(self, 'fusion_engine'),
                'explainer': hasattr(self, 'explainer'),
                'retrievers': list(self.retrievers.keys()) if hasattr(self, 'retrievers') else []
            }
        }
    
    def reset_stats(self):
        """Reset system statistics."""
        self.stats = {
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        self.logger.info("System statistics reset")
