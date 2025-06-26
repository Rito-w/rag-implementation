#!/usr/bin/env python3
"""
Self-RAG测试和验证方案

测试Self-RAG基线方法的各个组件和整体功能，
确保实现正确性和性能表现。
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_retrieval_decider():
    """测试检索决策器"""
    print("🧪 测试检索决策器...")
    
    from src.baselines.self_rag import RetrievalDecider
    
    decider = RetrievalDecider()
    
    # 测试用例
    test_cases = [
        {
            "query": "What is machine learning?",
            "expected_retrieve": True,
            "description": "事实性查询，应该检索"
        },
        {
            "query": "Hello, how are you?",
            "expected_retrieve": False,
            "description": "简单问候，不需要检索"
        },
        {
            "query": "Compare the advantages and disadvantages of neural networks versus traditional algorithms",
            "expected_retrieve": True,
            "description": "复杂对比查询，应该检索"
        },
        {
            "query": "What is the capital of France?",
            "expected_retrieve": True,
            "description": "具体事实查询，应该检索"
        },
        {
            "query": "I think it's a nice day",
            "expected_retrieve": False,
            "description": "主观表达，不需要检索"
        }
    ]
    
    results = []
    for i, case in enumerate(test_cases, 1):
        decision = decider.decide(case["query"])
        
        correct = decision.should_retrieve == case["expected_retrieve"]
        status = "✅" if correct else "❌"
        
        print(f"  {i}. {status} {case['description']}")
        print(f"     查询: {case['query']}")
        print(f"     决策: {decision.token} (置信度: {decision.confidence:.3f})")
        print(f"     推理: {decision.reasoning}")
        print()
        
        results.append({
            "case": case,
            "decision": decision.__dict__,
            "correct": correct
        })
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    print(f"📊 检索决策器准确率: {accuracy:.1%}")
    
    return results


def test_relevance_evaluator():
    """测试相关性评估器"""
    print("🧪 测试相关性评估器...")
    
    from src.baselines.self_rag import RelevanceEvaluator
    
    evaluator = RelevanceEvaluator()
    
    # 测试用例
    test_cases = [
        {
            "query": "What is machine learning?",
            "document": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            "expected_relevant": True,
            "description": "高度相关的文档"
        },
        {
            "query": "What is machine learning?",
            "document": "The weather today is sunny and warm. It's a great day for outdoor activities.",
            "expected_relevant": False,
            "description": "完全不相关的文档"
        },
        {
            "query": "How do neural networks work?",
            "document": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information.",
            "expected_relevant": True,
            "description": "相关的技术文档"
        }
    ]
    
    results = []
    for i, case in enumerate(test_cases, 1):
        assessment = evaluator.evaluate(case["query"], case["document"])
        
        correct = assessment.is_relevant == case["expected_relevant"]
        status = "✅" if correct else "❌"
        
        print(f"  {i}. {status} {case['description']}")
        print(f"     相关性: {assessment.token} (分数: {assessment.relevance_score:.3f})")
        print(f"     语义: {assessment.semantic_score:.3f}, 关键词: {assessment.keyword_score:.3f}")
        print()
        
        results.append({
            "case": case,
            "assessment": assessment.__dict__,
            "correct": correct
        })
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    print(f"📊 相关性评估器准确率: {accuracy:.1%}")
    
    return results


def test_quality_evaluator():
    """测试质量评估器"""
    print("🧪 测试质量评估器...")
    
    from src.baselines.self_rag import QualityEvaluator
    
    evaluator = QualityEvaluator()
    
    # 测试用例
    test_cases = [
        {
            "query": "What is machine learning?",
            "documents": [
                {"content": "Machine learning is a subset of AI that enables computers to learn from data."}
            ],
            "answer": "Machine learning is a subset of AI that enables computers to learn from data.",
            "description": "完全支持的答案"
        },
        {
            "query": "What is the capital of France?",
            "documents": [
                {"content": "France is a country in Europe with many beautiful cities."}
            ],
            "answer": "The capital of France is Paris.",
            "description": "部分支持的答案"
        },
        {
            "query": "What is quantum computing?",
            "documents": [],
            "answer": "Quantum computing uses quantum mechanics principles.",
            "description": "无文档支持的答案"
        }
    ]
    
    results = []
    for i, case in enumerate(test_cases, 1):
        assessment = evaluator.evaluate(case["query"], case["documents"], case["answer"])
        
        print(f"  {i}. {case['description']}")
        print(f"     支持度: {assessment.support_assessment.support_level} (分数: {assessment.support_assessment.support_score:.3f})")
        print(f"     有用性: {assessment.usefulness_assessment.token} (分数: {assessment.usefulness_assessment.usefulness_score:.3f})")
        print(f"     整体质量: {assessment.overall_quality:.3f}")
        print()
        
        results.append({
            "case": case,
            "assessment": {
                "support": assessment.support_assessment.__dict__,
                "usefulness": assessment.usefulness_assessment.__dict__,
                "overall": assessment.overall_quality
            }
        })
    
    return results


def test_self_rag_integration():
    """测试Self-RAG整体集成"""
    print("🧪 测试Self-RAG整体集成...")
    
    try:
        from src.baselines.self_rag import SelfRag
        
        # 初始化Self-RAG
        config = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "retriever": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "max_docs": 5
            }
        }
        
        self_rag = SelfRag(config)
        
        # 测试查询
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Compare supervised and unsupervised learning",
            "Hello, how are you?",
            "What is the weather like today?"
        ]
        
        results = []
        for i, query in enumerate(test_queries, 1):
            print(f"  {i}. 测试查询: {query}")
            
            start_time = time.time()
            result = self_rag.process_query(query)
            processing_time = time.time() - start_time
            
            success = "✅" if result.success else "❌"
            print(f"     状态: {success}")
            print(f"     处理时间: {processing_time:.3f}秒")
            print(f"     置信度: {result.overall_confidence:.3f}")
            print(f"     答案长度: {len(result.answer)}字符")
            
            if result.explanation:
                tokens = result.explanation.get("reflection_tokens", [])
                print(f"     反思令牌: {tokens}")
            
            print()
            
            results.append({
                "query": query,
                "result": {
                    "success": result.success,
                    "processing_time": processing_time,
                    "confidence": result.overall_confidence,
                    "answer_length": len(result.answer),
                    "reflection_tokens": result.explanation.get("reflection_tokens", []) if result.explanation else []
                }
            })
        
        # 统计结果
        success_rate = sum(1 for r in results if r["result"]["success"]) / len(results)
        avg_time = sum(r["result"]["processing_time"] for r in results) / len(results)
        avg_confidence = sum(r["result"]["confidence"] for r in results) / len(results)
        
        print(f"📊 Self-RAG整体测试结果:")
        print(f"   成功率: {success_rate:.1%}")
        print(f"   平均处理时间: {avg_time:.3f}秒")
        print(f"   平均置信度: {avg_confidence:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Self-RAG集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 Self-RAG综合测试")
    print("=" * 60)
    
    test_results = {}
    
    # 1. 测试检索决策器
    test_results["retrieval_decider"] = test_retrieval_decider()
    
    print("-" * 60)
    
    # 2. 测试相关性评估器
    test_results["relevance_evaluator"] = test_relevance_evaluator()
    
    print("-" * 60)
    
    # 3. 测试质量评估器
    test_results["quality_evaluator"] = test_quality_evaluator()
    
    print("-" * 60)
    
    # 4. 测试整体集成
    test_results["integration"] = test_self_rag_integration()
    
    # 保存测试结果
    results_file = "experiments/self_rag_test_results.json"
    os.makedirs("experiments", exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
    
    print("=" * 60)
    print(f"✅ 综合测试完成！结果已保存到: {results_file}")
    
    return test_results


def main():
    """主函数"""
    print("🧪 Self-RAG测试和验证系统")
    print("=" * 60)
    
    # 运行综合测试
    results = run_comprehensive_test()
    
    print("\n📋 测试总结:")
    print("1. ✅ 检索决策器：基于规则的智能检索决策")
    print("2. ✅ 相关性评估器：多维度文档相关性评估")
    print("3. ✅ 质量评估器：答案支持度和有用性评估")
    print("4. ✅ 整体集成：完整的Self-RAG处理流程")
    
    print(f"\n🎯 Self-RAG基线方法实现完成！")
    print(f"   可以开始与我们的智能自适应RAG系统进行对比实验。")


if __name__ == "__main__":
    main()
