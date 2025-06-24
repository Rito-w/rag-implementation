#!/usr/bin/env python3
"""
核心组件测试脚本

测试智能自适应RAG系统的核心组件功能，包括：
1. 查询智能分析器
2. 动态权重控制器
3. 智能融合引擎
4. 决策解释器
"""

import sys
import numpy as np
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_query_analyzer():
    """测试查询智能分析器"""
    print("🧠 测试查询智能分析器...")
    
    try:
        from src.core.query_analyzer import QueryIntelligenceAnalyzer
        
        # 初始化分析器
        analyzer = QueryIntelligenceAnalyzer()
        
        # 测试不同类型的查询
        test_queries = [
            "What is machine learning?",  # 简单事实查询
            "How do transformer models work and what are their advantages?",  # 复杂查询
            "Compare different RAG approaches",  # 分析查询
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n  测试查询 {i}: {query}")
            
            # 分析查询
            analysis = analyzer.analyze(query)
            
            print(f"    复杂度: {analysis.complexity_score:.2f}/5.0")
            print(f"    类型: {analysis.query_type.value}")
            print(f"    置信度: {analysis.confidence:.2%}")
            print(f"    关键词: {analysis.key_terms[:3]}")
            print(f"    处理时间: {analysis.processing_time:.3f}秒")
        
        print("✅ 查询智能分析器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 查询智能分析器测试失败: {str(e)}")
        return False


def test_weight_controller():
    """测试动态权重控制器"""
    print("\n⚖️  测试动态权重控制器...")
    
    try:
        from src.core.query_analyzer import QueryIntelligenceAnalyzer
        from src.core.weight_controller import DynamicWeightController
        
        # 初始化组件
        analyzer = QueryIntelligenceAnalyzer()
        controller = DynamicWeightController()
        
        # 测试查询
        query = "What are the advantages of semantic search?"
        
        # 分析查询
        analysis = analyzer.analyze(query)
        
        # 计算权重
        weights = controller.compute_weights(analysis)
        
        print(f"    查询: {query}")
        print(f"    稠密权重: {weights.dense_weight:.2%}")
        print(f"    稀疏权重: {weights.sparse_weight:.2%}")
        print(f"    混合权重: {weights.hybrid_weight:.2%}")
        print(f"    策略: {weights.strategy.value}")
        print(f"    权重置信度: {weights.weight_confidence:.2%}")
        print(f"    处理时间: {weights.processing_time:.3f}秒")
        
        # 验证权重和为1
        total_weight = weights.dense_weight + weights.sparse_weight + weights.hybrid_weight
        assert abs(total_weight - 1.0) < 1e-6, f"权重和不为1: {total_weight}"
        
        print("✅ 动态权重控制器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 动态权重控制器测试失败: {str(e)}")
        return False


def test_fusion_engine():
    """测试智能融合引擎"""
    print("\n🔀 测试智能融合引擎...")
    
    try:
        from src.core.query_analyzer import QueryIntelligenceAnalyzer
        from src.core.weight_controller import DynamicWeightController
        from src.core.fusion_engine import IntelligentFusionEngine
        from src.models.retrieval_result import DocumentScore
        
        # 初始化组件
        analyzer = QueryIntelligenceAnalyzer()
        controller = DynamicWeightController()
        fusion_engine = IntelligentFusionEngine()
        
        # 测试查询
        query = "How does machine learning work?"
        
        # 分析查询和计算权重
        analysis = analyzer.analyze(query)
        weights = controller.compute_weights(analysis)
        
        # 创建模拟检索结果
        mock_results = {
            'dense': [
                DocumentScore(
                    document_id="doc_1",
                    content="Machine learning is a method of data analysis...",
                    final_score=0.9,
                    retrieval_method="dense"
                ),
                DocumentScore(
                    document_id="doc_2", 
                    content="Deep learning uses neural networks...",
                    final_score=0.8,
                    retrieval_method="dense"
                )
            ],
            'sparse': [
                DocumentScore(
                    document_id="doc_1",
                    content="Machine learning is a method of data analysis...",
                    final_score=0.7,
                    retrieval_method="sparse"
                ),
                DocumentScore(
                    document_id="doc_3",
                    content="Artificial intelligence encompasses machine learning...",
                    final_score=0.6,
                    retrieval_method="sparse"
                )
            ]
        }
        
        # 执行融合
        fused_results = fusion_engine.fuse_results(mock_results, analysis, weights)
        
        print(f"    查询: {query}")
        print(f"    融合结果数量: {len(fused_results)}")
        
        for i, doc in enumerate(fused_results[:3], 1):
            print(f"    结果 {i}: ID={doc.document_id}, 分数={doc.final_score:.3f}")
        
        print("✅ 智能融合引擎测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 智能融合引擎测试失败: {str(e)}")
        return False


def test_explainer():
    """测试决策解释器"""
    print("\n💡 测试决策解释器...")
    
    try:
        from src.core.query_analyzer import QueryIntelligenceAnalyzer
        from src.core.weight_controller import DynamicWeightController
        from src.core.explainer import DecisionExplainer
        from src.models.retrieval_result import DocumentScore
        
        # 初始化组件
        analyzer = QueryIntelligenceAnalyzer()
        controller = DynamicWeightController()
        explainer = DecisionExplainer()
        
        # 测试查询
        query = "What is the difference between supervised and unsupervised learning?"
        
        # 分析查询和计算权重
        analysis = analyzer.analyze(query)
        weights = controller.compute_weights(analysis)
        
        # 创建模拟检索结果
        mock_results = [
            DocumentScore(
                document_id="doc_1",
                content="Supervised learning uses labeled data...",
                final_score=0.9,
                retrieval_method="dense",
                relevance_score=0.85
            )
        ]
        
        # 模拟答案
        answer = "Supervised learning uses labeled data while unsupervised learning finds patterns in unlabeled data."
        
        # 生成解释
        explanations = explainer.generate_explanations(
            query, analysis, weights, mock_results, answer
        )
        
        print(f"    查询: {query}")
        print(f"    查询分析解释: {explanations.get('query_analysis', 'N/A')[:100]}...")
        print(f"    权重分配解释: {explanations.get('weight_allocation', 'N/A')[:100]}...")
        print(f"    检索过程解释: {explanations.get('retrieval_process', 'N/A')[:100]}...")
        print(f"    答案来源解释: {explanations.get('answer_source', 'N/A')[:100]}...")
        
        print("✅ 决策解释器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 决策解释器测试失败: {str(e)}")
        return False


def test_retrievers():
    """测试检索器组件"""
    print("\n🔍 测试检索器组件...")
    
    try:
        from src.retrievers.dense_retriever import DenseRetriever
        from src.retrievers.sparse_retriever import SparseRetriever
        from src.retrievers.hybrid_retriever import HybridRetriever
        
        # 测试稠密检索器
        print("  测试稠密检索器...")
        dense_retriever = DenseRetriever()
        dense_results = dense_retriever.retrieve("machine learning", k=3)
        print(f"    稠密检索结果: {len(dense_results)}个")
        
        # 测试稀疏检索器
        print("  测试稀疏检索器...")
        sparse_retriever = SparseRetriever()
        sparse_results = sparse_retriever.retrieve("machine learning", k=3)
        print(f"    稀疏检索结果: {len(sparse_results)}个")
        
        # 测试混合检索器
        print("  测试混合检索器...")
        hybrid_retriever = HybridRetriever()
        hybrid_results = hybrid_retriever.retrieve("machine learning", k=3)
        print(f"    混合检索结果: {len(hybrid_results)}个")
        
        print("✅ 检索器组件测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 检索器组件测试失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("🎯 智能自适应RAG系统 - 核心组件测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        test_query_analyzer,
        test_weight_controller,
        test_fusion_engine,
        test_explainer,
        test_retrievers
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 出现异常: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有核心组件测试通过！系统基础功能正常。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关组件。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
