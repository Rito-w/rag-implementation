"""
Basic functionality tests for the Intelligent Adaptive RAG System
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.query_analysis import QueryAnalysis, QueryType, ComplexityFactors
from src.models.weight_allocation import WeightAllocation, RetrievalStrategy
from src.models.retrieval_result import RetrievalResult, DocumentScore
from src.utils.config import Config
import numpy as np


class TestDataModels:
    """Test data model classes."""
    
    def test_query_analysis_creation(self):
        """Test QueryAnalysis model creation and validation."""
        complexity_factors = ComplexityFactors(
            lexical_complexity=0.5,
            syntactic_complexity=0.3,
            entity_complexity=0.4,
            domain_complexity=0.6,
            word_frequency_score=0.7,
            semantic_depth_score=0.8,
            technical_density=0.2,
            dependency_tree_depth=5,
            named_entity_count=3,
            domain_specificity=0.4
        )
        
        analysis = QueryAnalysis(
            original_query="What is machine learning?",
            processed_query="what is machine learning",
            query_length=25,
            complexity_score=2.5,
            complexity_factors=complexity_factors,
            query_type=QueryType.LOCAL_FACTUAL,
            type_confidence=0.9,
            key_terms=["machine", "learning"],
            named_entities=["machine learning"],
            semantic_concepts=["artificial intelligence"],
            feature_vector=np.array([0.1, 0.2, 0.3]),
            analysis_timestamp="2024-01-01T00:00:00",
            processing_time=0.1,
            confidence=0.85
        )
        
        assert analysis.complexity_score == 2.5
        assert analysis.query_type == QueryType.LOCAL_FACTUAL
        assert analysis.confidence == 0.85
        assert "简单" in analysis.get_complexity_explanation()
    
    def test_weight_allocation_creation(self):
        """Test WeightAllocation model creation and validation."""
        allocation = WeightAllocation(
            dense_weight=0.5,
            sparse_weight=0.3,
            hybrid_weight=0.2,
            strategy=RetrievalStrategy.BALANCED_HYBRID,
            strategy_confidence=0.8,
            feature_importance={"complexity": 0.6, "type": 0.4},
            weight_confidence=0.9,
            calculation_method="neural_network",
            allocation_reasoning="基于查询复杂度的平衡策略",
            strategy_reasoning="选择平衡策略以兼顾准确性和覆盖面",
            timestamp="2024-01-01T00:00:00",
            processing_time=0.05
        )
        
        assert allocation.dense_weight == 0.5
        assert allocation.get_primary_method() == "dense"
        assert abs(allocation.dense_weight + allocation.sparse_weight + allocation.hybrid_weight - 1.0) < 1e-6
    
    def test_weight_allocation_validation(self):
        """Test WeightAllocation validation."""
        with pytest.raises(ValueError):
            # Weights don't sum to 1
            WeightAllocation(
                dense_weight=0.5,
                sparse_weight=0.3,
                hybrid_weight=0.3,  # Sum = 1.1
                strategy=RetrievalStrategy.BALANCED_HYBRID,
                strategy_confidence=0.8,
                feature_importance={},
                weight_confidence=0.9,
                calculation_method="test",
                allocation_reasoning="test",
                strategy_reasoning="test",
                timestamp="2024-01-01T00:00:00",
                processing_time=0.05
            )
    
    def test_document_score_creation(self):
        """Test DocumentScore model creation."""
        doc = DocumentScore(
            document_id="doc_1",
            content="This is a test document about machine learning.",
            title="Machine Learning Basics",
            dense_score=0.8,
            sparse_score=0.6,
            hybrid_score=0.7,
            final_score=0.75,
            retrieval_method="dense",
            relevance_score=0.9,
            quality_score=0.8,
            diversity_bonus=0.1,
            source="test_corpus"
        )
        
        assert doc.document_id == "doc_1"
        assert doc.final_score == 0.75
        assert doc.retrieval_method == "dense"
        
        breakdown = doc.get_score_breakdown()
        assert "dense_score" in breakdown
        assert breakdown["dense_score"] == 0.8
    
    def test_retrieval_result_creation(self):
        """Test RetrievalResult model creation."""
        docs = [
            DocumentScore(
                document_id="doc_1",
                content="Test content 1",
                final_score=0.9,
                retrieval_method="dense"
            ),
            DocumentScore(
                document_id="doc_2", 
                content="Test content 2",
                final_score=0.7,
                retrieval_method="sparse"
            )
        ]
        
        result = RetrievalResult(
            query="test query",
            answer="test answer",
            retrieved_documents=docs,
            total_documents_found=2,
            overall_confidence=0.85,
            answer_quality_score=0.8,
            retrieval_quality_score=0.9,
            query_analysis_explanation="test analysis",
            weight_allocation_explanation="test allocation",
            retrieval_process_explanation="test retrieval",
            answer_source_explanation="test source",
            processing_time=0.5,
            timestamp="2024-01-01T00:00:00",
            system_version="0.1.0"
        )
        
        assert result.query == "test query"
        assert len(result.retrieved_documents) == 2
        assert result.overall_confidence == 0.85
        
        top_docs = result.get_top_documents(1)
        assert len(top_docs) == 1
        assert top_docs[0].document_id == "doc_1"  # Highest score


class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test Config class creation with defaults."""
        config = Config()
        
        assert config.get("system.version") == "0.1.0"
        assert config.get("logging.level") == "INFO"
        assert config.get("nonexistent.key", "default") == "default"
    
    def test_config_set_get(self):
        """Test setting and getting configuration values."""
        config = Config()
        
        config.set("test.key", "test_value")
        assert config.get("test.key") == "test_value"
        
        config.set("nested.deep.key", 42)
        assert config.get("nested.deep.key") == 42
    
    def test_config_has(self):
        """Test checking if configuration keys exist."""
        config = Config()
        
        assert config.has("system.version")
        assert not config.has("nonexistent.key")
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Should pass with default configuration
        assert config.validate()


class TestUtilities:
    """Test utility functions."""
    
    def test_query_type_enum(self):
        """Test QueryType enum values."""
        assert QueryType.LOCAL_FACTUAL.value == "local_factual"
        assert QueryType.GLOBAL_ANALYTICAL.value == "global_analytical"
        assert QueryType.SEMANTIC_COMPLEX.value == "semantic_complex"
        assert QueryType.SPECIFIC_DETAILED.value == "specific_detailed"
        assert QueryType.MULTI_HOP_REASONING.value == "multi_hop_reasoning"
    
    def test_retrieval_strategy_enum(self):
        """Test RetrievalStrategy enum values."""
        assert RetrievalStrategy.PRECISION_FOCUSED.value == "precision_focused"
        assert RetrievalStrategy.SEMANTIC_FOCUSED.value == "semantic_focused"
        assert RetrievalStrategy.COMPREHENSIVE_COVERAGE.value == "comprehensive_coverage"
        assert RetrievalStrategy.EXACT_MATCH.value == "exact_match"
        assert RetrievalStrategy.MULTI_STEP_REASONING.value == "multi_step_reasoning"
        assert RetrievalStrategy.BALANCED_HYBRID.value == "balanced_hybrid"


if __name__ == "__main__":
    pytest.main([__file__])
