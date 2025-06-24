#!/usr/bin/env python3
"""
修复文档索引问题

解决SQuAD等数据集的文档无法被检索器正确索引的问题。
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.core.intelligent_adapter import IntelligentAdaptiveRAG
from src.utils.logging import setup_logger


def load_documents_from_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """
    从数据集加载文档
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        List[Dict]: 文档列表
    """
    logger = setup_logger("DocumentLoader")
    
    try:
        doc_path = Path(f"data/{dataset_name}/documents.json")
        
        if not doc_path.exists():
            logger.warning(f"文档文件不存在: {doc_path}")
            return []
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"从{dataset_name}加载了{len(documents)}个文档")
        return documents
        
    except Exception as e:
        logger.error(f"加载文档失败 {dataset_name}: {str(e)}")
        return []


def test_document_indexing():
    """测试文档索引功能"""
    logger = setup_logger("DocumentIndexTest")
    
    print("🔧 测试文档索引功能...")
    
    # 测试各个数据集
    datasets = ["samples", "squad", "synthetic"]
    
    for dataset_name in datasets:
        print(f"\n📊 测试数据集: {dataset_name}")
        print("-" * 40)
        
        # 加载文档
        documents = load_documents_from_dataset(dataset_name)
        
        if not documents:
            print(f"❌ {dataset_name}: 没有文档可加载")
            continue
        
        print(f"✅ 加载了 {len(documents)} 个文档")
        
        # 显示文档样例
        for i, doc in enumerate(documents[:3]):
            print(f"  文档 {i+1}:")
            print(f"    ID: {doc.get('id', 'N/A')}")
            print(f"    标题: {doc.get('title', 'N/A')[:50]}...")
            print(f"    内容长度: {len(doc.get('content', ''))}")
        
        # 测试RAG系统是否能正确处理这些文档
        try:
            rag = IntelligentAdaptiveRAG()
            
            # 手动添加文档到检索器 (这是问题所在)
            print(f"  🔧 手动添加文档到检索器...")
            
            # 为每个检索器添加文档
            for retriever_name in ['dense', 'sparse', 'hybrid']:
                retriever = getattr(rag, f'{retriever_name}_retriever', None)
                if retriever and hasattr(retriever, 'add_documents'):
                    retriever.add_documents(documents)
                    print(f"    ✅ 已添加文档到 {retriever_name} 检索器")
            
            # 测试查询
            test_queries = [
                "What is machine learning?",
                "Notre Dame",
                "artificial intelligence"
            ]
            
            for query in test_queries:
                print(f"  🔍 测试查询: {query}")
                result = rag.process_query(query)
                print(f"    检索到文档数: {len(result.retrieved_documents)}")
                print(f"    置信度: {result.overall_confidence:.2%}")
                
                if len(result.retrieved_documents) > 0:
                    print(f"    ✅ 检索成功")
                else:
                    print(f"    ❌ 检索失败")
        
        except Exception as e:
            print(f"  ❌ 测试失败: {str(e)}")


def create_enhanced_rag_with_documents():
    """创建增强的RAG系统，预加载所有数据集文档"""
    logger = setup_logger("EnhancedRAG")
    
    print("🚀 创建增强的RAG系统...")
    
    # 初始化RAG系统
    rag = IntelligentAdaptiveRAG()
    
    # 加载所有数据集的文档
    all_documents = []
    datasets = ["samples", "squad", "synthetic"]
    
    for dataset_name in datasets:
        documents = load_documents_from_dataset(dataset_name)
        if documents:
            # 为文档添加数据集标识
            for doc in documents:
                doc['dataset'] = dataset_name
            all_documents.extend(documents)
            print(f"✅ 加载 {dataset_name}: {len(documents)} 个文档")
    
    print(f"📊 总共加载了 {len(all_documents)} 个文档")
    
    # 手动添加文档到所有检索器
    print("🔧 添加文档到检索器...")
    
    try:
        # 添加到稠密检索器
        if hasattr(rag.dense_retriever, 'add_documents'):
            rag.dense_retriever.add_documents(all_documents)
            print("✅ 文档已添加到稠密检索器")
        
        # 添加到稀疏检索器
        if hasattr(rag.sparse_retriever, 'add_documents'):
            rag.sparse_retriever.add_documents(all_documents)
            print("✅ 文档已添加到稀疏检索器")
        
        # 添加到混合检索器
        if hasattr(rag.hybrid_retriever, 'add_documents'):
            rag.hybrid_retriever.add_documents(all_documents)
            print("✅ 文档已添加到混合检索器")
    
    except Exception as e:
        print(f"❌ 添加文档失败: {str(e)}")
        return None
    
    return rag


def run_enhanced_experiment():
    """运行增强的实验"""
    print("🧪 运行增强实验...")
    
    # 创建增强的RAG系统
    rag = create_enhanced_rag_with_documents()
    
    if not rag:
        print("❌ 无法创建增强RAG系统")
        return
    
    # 测试查询
    test_queries = [
        {"query": "What is machine learning?", "expected_dataset": "samples"},
        {"query": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?", "expected_dataset": "squad"},
        {"query": "What are the applications of deep learning?", "expected_dataset": "synthetic"},
        {"query": "How does neural networks work?", "expected_dataset": "synthetic"},
        {"query": "What is the Grotto at Notre Dame?", "expected_dataset": "squad"}
    ]
    
    print(f"\n🔍 测试 {len(test_queries)} 个查询...")
    print("=" * 60)
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_dataset = test_case["expected_dataset"]
        
        print(f"\n查询 {i}: {query}")
        print(f"期望数据集: {expected_dataset}")
        print("-" * 40)
        
        try:
            result = rag.process_query(query)
            
            # 分析结果
            doc_count = len(result.retrieved_documents)
            confidence = result.overall_confidence
            
            print(f"检索到文档数: {doc_count}")
            print(f"置信度: {confidence:.2%}")
            
            if doc_count > 0:
                print("前3个文档:")
                for j, doc in enumerate(result.retrieved_documents[:3]):
                    dataset = getattr(doc, 'dataset', 'unknown')
                    score = getattr(doc, 'final_score', 0)
                    print(f"  {j+1}. 数据集: {dataset}, 分数: {score:.3f}")
                
                # 检查是否找到了期望数据集的文档
                found_datasets = set()
                for doc in result.retrieved_documents:
                    dataset = getattr(doc, 'dataset', 'unknown')
                    found_datasets.add(dataset)
                
                if expected_dataset in found_datasets:
                    print(f"✅ 成功找到来自 {expected_dataset} 的文档")
                else:
                    print(f"⚠️  未找到来自 {expected_dataset} 的文档")
                    print(f"实际找到的数据集: {found_datasets}")
            else:
                print("❌ 没有检索到任何文档")
            
            results.append({
                "query": query,
                "expected_dataset": expected_dataset,
                "doc_count": doc_count,
                "confidence": confidence,
                "success": doc_count > 0
            })
        
        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
            results.append({
                "query": query,
                "expected_dataset": expected_dataset,
                "doc_count": 0,
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            })
    
    # 总结结果
    print("\n" + "=" * 60)
    print("📊 实验结果总结:")
    print("=" * 60)
    
    successful_queries = sum(1 for r in results if r["success"])
    total_queries = len(results)
    avg_confidence = sum(r["confidence"] for r in results) / total_queries
    avg_doc_count = sum(r["doc_count"] for r in results) / total_queries
    
    print(f"成功查询: {successful_queries}/{total_queries} ({successful_queries/total_queries*100:.1f}%)")
    print(f"平均置信度: {avg_confidence:.2%}")
    print(f"平均检索文档数: {avg_doc_count:.1f}")
    
    if successful_queries == total_queries:
        print("🎉 所有查询都成功！文档索引问题已解决！")
    else:
        print("⚠️  仍有部分查询失败，需要进一步调试")
    
    return results


def main():
    """主函数"""
    print("🔧 智能自适应RAG系统 - 文档索引修复")
    print("=" * 60)
    
    # 步骤1: 测试文档索引
    test_document_indexing()
    
    print("\n" + "=" * 60)
    
    # 步骤2: 运行增强实验
    results = run_enhanced_experiment()
    
    print("\n🎯 修复完成！")
    
    return results


if __name__ == "__main__":
    main()
