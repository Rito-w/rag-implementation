#!/usr/bin/env python3
"""
快速修复测试脚本

修复嵌入模型配置问题并测试系统功能。
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_embedding_model():
    """测试嵌入模型是否正常工作"""
    print("🔧 测试嵌入模型...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 测试加载模型
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"正在加载模型: {model_name}")
        
        model = SentenceTransformer(model_name)
        print("✅ 模型加载成功")
        
        # 测试编码
        test_texts = ["What is machine learning?", "How do neural networks work?"]
        embeddings = model.encode(test_texts)
        print(f"✅ 编码成功，维度: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 嵌入模型测试失败: {str(e)}")
        return False


def test_rag_system():
    """测试RAG系统"""
    print("\n🧪 测试RAG系统...")
    
    try:
        from src.core.intelligent_adapter import IntelligentAdaptiveRAG
        
        # 初始化系统
        print("正在初始化RAG系统...")
        rag = IntelligentAdaptiveRAG()
        print("✅ RAG系统初始化成功")
        
        # 测试查询
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Compare supervised and unsupervised learning"
        ]
        
        results = []
        for i, query in enumerate(test_queries, 1):
            print(f"\n测试查询 {i}: {query}")
            
            start_time = time.time()
            result = rag.process_query(query)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            print(f"  处理时间: {processing_time:.3f}秒")
            print(f"  置信度: {result.overall_confidence:.2%}")
            print(f"  检索文档数: {len(result.retrieved_documents)}")
            print(f"  答案长度: {len(result.answer)}")
            
            results.append({
                "query": query,
                "processing_time": processing_time,
                "confidence": result.overall_confidence,
                "doc_count": len(result.retrieved_documents),
                "success": True
            })
        
        # 统计结果
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        
        print(f"\n📊 测试结果统计:")
        print(f"  成功率: {success_rate:.1%}")
        print(f"  平均处理时间: {avg_time:.3f}秒")
        print(f"  平均置信度: {avg_confidence:.2%}")
        
        if success_rate == 1.0:
            print("✅ RAG系统测试成功！")
            return True
        else:
            print("⚠️  RAG系统部分功能异常")
            return False
        
    except Exception as e:
        print(f"❌ RAG系统测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_small_experiment():
    """运行小规模实验"""
    print("\n🚀 运行小规模实验...")
    
    try:
        from experiments.experiment_runner import ExperimentRunner
        
        # 创建实验配置
        config = {
            "datasets": ["samples"],
            "systems": ["intelligent_rag"],
            "metrics": ["retrieval_quality", "answer_quality", "efficiency"],
            "sample_size": 5,
            "repetitions": 1,
            "timeout": 30
        }
        
        # 保存配置
        config_path = "experiments/quick_fix_config.json"
        os.makedirs("experiments", exist_ok=True)
        
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # 运行实验
        runner = ExperimentRunner(config_path)
        results = runner.run_experiment()
        
        if "error" in results:
            print(f"❌ 实验失败: {results['error']}")
            return False
        
        # 分析结果
        total_queries = 0
        successful_queries = 0
        
        for dataset_name, dataset_results in results["results"].items():
            for system_name, system_result in dataset_results.items():
                if "error" not in system_result:
                    total_queries += system_result.get("total_queries", 0)
                    successful_queries += system_result.get("successful_queries", 0)
        
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        
        print(f"📊 实验结果:")
        print(f"  总查询数: {total_queries}")
        print(f"  成功查询数: {successful_queries}")
        print(f"  成功率: {success_rate:.1%}")
        
        if success_rate > 0.8:
            print("✅ 小规模实验成功！")
            return True
        else:
            print("⚠️  实验成功率较低，需要进一步调试")
            return False
        
    except Exception as e:
        print(f"❌ 实验运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🔧 智能自适应RAG系统 - 快速修复测试")
    print("=" * 60)
    
    # 步骤1: 测试嵌入模型
    embedding_ok = test_embedding_model()
    
    if not embedding_ok:
        print("\n❌ 嵌入模型测试失败，无法继续")
        return 1
    
    # 步骤2: 测试RAG系统
    rag_ok = test_rag_system()
    
    if not rag_ok:
        print("\n❌ RAG系统测试失败，无法继续")
        return 1
    
    # 步骤3: 运行小规模实验
    experiment_ok = run_small_experiment()
    
    # 总结
    print("\n" + "=" * 60)
    if embedding_ok and rag_ok and experiment_ok:
        print("🎉 所有测试通过！系统修复成功！")
        print("\n📋 下一步可以:")
        print("  1. 运行大规模实验: python run_large_scale_experiment.py")
        print("  2. 运行真实数据集实验: python run_real_experiment.py")
        print("  3. 开始撰写技术论文")
        return 0
    else:
        print("⚠️  部分测试失败，需要进一步调试")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
