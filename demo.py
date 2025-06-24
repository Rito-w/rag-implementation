#!/usr/bin/env python3
"""
Demo script for Intelligent Adaptive RAG System

This script demonstrates the basic usage of the intelligent adaptive RAG system
with sample queries and explanations.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.intelligent_adapter import IntelligentAdaptiveRAG
    from src.utils.logging import setup_logger
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖包")
    print("运行: pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Main demo function."""
    
    # Setup logging
    logger = setup_logger("Demo", level="INFO")
    logger.info("🎯 Intelligent Adaptive RAG System Demo")
    logger.info("=" * 50)
    
    try:
        # Initialize the system
        logger.info("🔧 Initializing Intelligent Adaptive RAG system...")
        
        # Use default configuration
        rag_system = IntelligentAdaptiveRAG(
            config_path="configs/default.yaml"
        )
        
        logger.info("✅ System initialized successfully!")
        
        # Demo queries with different complexity levels
        demo_queries = [
            {
                "query": "What is machine learning?",
                "description": "简单事实查询",
                "expected_type": "LOCAL_FACTUAL"
            },
            {
                "query": "How do transformer models work and what are their advantages over RNNs in natural language processing tasks?",
                "description": "复杂技术查询",
                "expected_type": "SEMANTIC_COMPLEX"
            },
            {
                "query": "Compare the performance of different retrieval methods in RAG systems and analyze their trade-offs",
                "description": "分析比较查询",
                "expected_type": "GLOBAL_ANALYTICAL"
            },
            {
                "query": "What is the exact formula for calculating BLEU score in machine translation evaluation?",
                "description": "具体详细查询",
                "expected_type": "SPECIFIC_DETAILED"
            },
            {
                "query": "If a neural network has 3 layers with 100, 50, and 10 neurons respectively, and uses ReLU activation, what would be the impact on gradient flow during backpropagation?",
                "description": "多跳推理查询",
                "expected_type": "MULTI_HOP_REASONING"
            }
        ]
        
        # Process each demo query
        for i, demo in enumerate(demo_queries, 1):
            logger.info(f"\n📝 Demo Query {i}: {demo['description']}")
            logger.info(f"Query: {demo['query']}")
            logger.info("-" * 80)
            
            start_time = time.time()
            
            try:
                # Process the query
                result = rag_system.process_query(
                    query=demo['query'],
                    return_detailed=True
                )
                
                processing_time = time.time() - start_time
                
                # Display results
                display_results(result, logger)
                
                logger.info(f"⏱️  Processing time: {processing_time:.3f} seconds")
                
            except Exception as e:
                logger.error(f"❌ Error processing query: {str(e)}")
            
            logger.info("=" * 80)
        
        # Display system statistics
        logger.info("\n📊 System Statistics:")
        stats = rag_system.get_system_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        return 1
    
    return 0


def display_results(result, logger):
    """Display query processing results in a user-friendly format."""
    
    # Basic information
    logger.info(f"🎯 Query: {result.query}")
    logger.info(f"📝 Answer: {result.answer}")
    logger.info(f"🔍 Confidence: {result.overall_confidence:.2%}")
    
    # Query analysis
    if result.query_analysis:
        analysis = result.query_analysis
        logger.info(f"\n🧠 Query Analysis:")
        logger.info(f"  • Complexity Score: {analysis.complexity_score:.2f}/5.0")
        logger.info(f"  • Query Type: {analysis.query_type.value}")
        logger.info(f"  • Type Confidence: {analysis.type_confidence:.2%}")
        logger.info(f"  • Key Terms: {', '.join(analysis.key_terms[:5])}")
    
    # Weight allocation
    if result.weight_allocation:
        weights = result.weight_allocation
        logger.info(f"\n⚖️  Weight Allocation:")
        logger.info(f"  • Dense Weight: {weights.dense_weight:.2%}")
        logger.info(f"  • Sparse Weight: {weights.sparse_weight:.2%}")
        logger.info(f"  • Hybrid Weight: {weights.hybrid_weight:.2%}")
        logger.info(f"  • Strategy: {weights.strategy.value}")
    
    # Retrieval results
    logger.info(f"\n📚 Retrieval Results:")
    logger.info(f"  • Documents Found: {result.total_documents_found}")
    logger.info(f"  • Retrieval Quality: {result.retrieval_quality_score:.2%}")
    logger.info(f"  • Answer Quality: {result.answer_quality_score:.2%}")
    
    # Top documents
    top_docs = result.get_top_documents(3)
    if top_docs:
        logger.info(f"\n📄 Top Documents:")
        for i, doc in enumerate(top_docs, 1):
            logger.info(f"  {i}. Score: {doc.final_score:.3f} | Method: {doc.retrieval_method}")
            logger.info(f"     Content: {doc.content[:100]}...")
    
    # Explanations
    logger.info(f"\n💡 Explanations:")
    logger.info(f"  • Query Understanding: {result.query_analysis_explanation}")
    logger.info(f"  • Weight Allocation: {result.weight_allocation_explanation}")
    logger.info(f"  • Retrieval Process: {result.retrieval_process_explanation}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
