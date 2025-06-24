#!/usr/bin/env python3
"""
HuggingFace数据集下载脚本

使用HuggingFace datasets库下载标准RAG评估数据集，
这是更可靠的数据获取方式。
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger


class HFDatasetDownloader:
    """HuggingFace数据集下载器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据集下载器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger = setup_logger("HFDatasetDownloader")
        
        # 检查datasets库
        try:
            import datasets
            self.datasets = datasets
            self.logger.info("HuggingFace datasets库已加载")
        except ImportError:
            self.logger.error("请安装datasets库: pip install datasets")
            sys.exit(1)
    
    def download_ms_marco(self, subset_size: int = 1000) -> bool:
        """
        下载MS MARCO数据集的子集
        
        Args:
            subset_size: 子集大小
            
        Returns:
            bool: 是否下载成功
        """
        try:
            self.logger.info("开始下载MS MARCO数据集...")
            
            # 创建数据目录
            dataset_dir = self.data_dir / "ms_marco"
            dataset_dir.mkdir(exist_ok=True)
            
            # 下载MS MARCO数据集
            dataset = self.datasets.load_dataset("ms_marco", "v1.1", split="train")
            
            # 取子集
            if len(dataset) > subset_size:
                dataset = dataset.select(range(subset_size))
            
            # 转换为我们的格式
            queries = []
            documents = []
            
            for i, example in enumerate(dataset):
                # 查询
                query = {
                    "id": f"msmarco_{i}",
                    "query": example["query"],
                    "type": "factual"
                }
                queries.append(query)
                
                # 文档 (使用passages)
                for j, passage in enumerate(example["passages"]["passage_text"]):
                    if passage.strip():  # 过滤空文档
                        doc = {
                            "id": f"msmarco_doc_{i}_{j}",
                            "title": f"MS MARCO Document {i}-{j}",
                            "content": passage.strip(),
                            "source": "ms_marco"
                        }
                        documents.append(doc)
            
            # 保存数据
            with open(dataset_dir / "queries.json", 'w', encoding='utf-8') as f:
                json.dump(queries, f, indent=2, ensure_ascii=False)
            
            with open(dataset_dir / "documents.json", 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"MS MARCO数据集下载完成: {len(queries)}个查询, {len(documents)}个文档")
            return True
            
        except Exception as e:
            self.logger.error(f"MS MARCO数据集下载失败: {str(e)}")
            return False
    
    def download_natural_questions(self, subset_size: int = 500) -> bool:
        """
        下载Natural Questions数据集的子集
        
        Args:
            subset_size: 子集大小
            
        Returns:
            bool: 是否下载成功
        """
        try:
            self.logger.info("开始下载Natural Questions数据集...")
            
            # 创建数据目录
            dataset_dir = self.data_dir / "natural_questions"
            dataset_dir.mkdir(exist_ok=True)
            
            # 下载Natural Questions数据集
            dataset = self.datasets.load_dataset("natural_questions", split="train")
            
            # 取子集
            if len(dataset) > subset_size:
                dataset = dataset.select(range(subset_size))
            
            # 转换为我们的格式
            queries = []
            documents = []
            
            for i, example in enumerate(dataset):
                # 查询
                query = {
                    "id": f"nq_{i}",
                    "query": example["question"]["text"],
                    "type": "factual"
                }
                queries.append(query)
                
                # 文档 (使用document_text)
                if example["document"]["tokens"]["token"]:
                    # 重构文档文本
                    tokens = example["document"]["tokens"]["token"]
                    content = " ".join(tokens[:500])  # 限制长度
                    
                    doc = {
                        "id": f"nq_doc_{i}",
                        "title": example["document"]["title"] or f"NQ Document {i}",
                        "content": content,
                        "source": "natural_questions"
                    }
                    documents.append(doc)
            
            # 保存数据
            with open(dataset_dir / "queries.json", 'w', encoding='utf-8') as f:
                json.dump(queries, f, indent=2, ensure_ascii=False)
            
            with open(dataset_dir / "documents.json", 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Natural Questions数据集下载完成: {len(queries)}个查询, {len(documents)}个文档")
            return True
            
        except Exception as e:
            self.logger.error(f"Natural Questions数据集下载失败: {str(e)}")
            return False
    
    def download_squad(self, subset_size: int = 1000) -> bool:
        """
        下载SQuAD数据集作为替代
        
        Args:
            subset_size: 子集大小
            
        Returns:
            bool: 是否下载成功
        """
        try:
            self.logger.info("开始下载SQuAD数据集...")
            
            # 创建数据目录
            dataset_dir = self.data_dir / "squad"
            dataset_dir.mkdir(exist_ok=True)
            
            # 下载SQuAD数据集
            dataset = self.datasets.load_dataset("squad", split="train")
            
            # 取子集
            if len(dataset) > subset_size:
                dataset = dataset.select(range(subset_size))
            
            # 转换为我们的格式
            queries = []
            documents = []
            
            for i, example in enumerate(dataset):
                # 查询
                query = {
                    "id": f"squad_{i}",
                    "query": example["question"],
                    "type": "factual",
                    "answer": example["answers"]["text"][0] if example["answers"]["text"] else ""
                }
                queries.append(query)
                
                # 文档
                doc = {
                    "id": f"squad_doc_{i}",
                    "title": example["title"],
                    "content": example["context"],
                    "source": "squad"
                }
                documents.append(doc)
            
            # 保存数据
            with open(dataset_dir / "queries.json", 'w', encoding='utf-8') as f:
                json.dump(queries, f, indent=2, ensure_ascii=False)
            
            with open(dataset_dir / "documents.json", 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"SQuAD数据集下载完成: {len(queries)}个查询, {len(documents)}个文档")
            return True
            
        except Exception as e:
            self.logger.error(f"SQuAD数据集下载失败: {str(e)}")
            return False
    
    def create_synthetic_dataset(self, size: int = 100) -> bool:
        """
        创建合成数据集用于快速测试
        
        Args:
            size: 数据集大小
            
        Returns:
            bool: 是否创建成功
        """
        try:
            self.logger.info("创建合成数据集...")
            
            # 创建数据目录
            dataset_dir = self.data_dir / "synthetic"
            dataset_dir.mkdir(exist_ok=True)
            
            # 合成查询模板
            query_templates = [
                "What is {topic}?",
                "How does {topic} work?",
                "What are the benefits of {topic}?",
                "Compare {topic} with alternatives",
                "Explain the concept of {topic}",
                "What are the applications of {topic}?",
                "How to implement {topic}?",
                "What are the challenges in {topic}?",
                "Future trends in {topic}",
                "Best practices for {topic}"
            ]
            
            # 主题列表
            topics = [
                "machine learning", "deep learning", "neural networks", "artificial intelligence",
                "natural language processing", "computer vision", "reinforcement learning",
                "transformer models", "attention mechanism", "BERT", "GPT", "retrieval systems",
                "information retrieval", "question answering", "text summarization",
                "sentiment analysis", "named entity recognition", "part-of-speech tagging",
                "semantic search", "vector databases", "embedding models", "fine-tuning",
                "transfer learning", "few-shot learning", "zero-shot learning", "meta-learning"
            ]
            
            # 生成查询和文档
            queries = []
            documents = []
            
            for i in range(size):
                topic = topics[i % len(topics)]
                template = query_templates[i % len(query_templates)]
                
                # 查询
                query = {
                    "id": f"synthetic_{i}",
                    "query": template.format(topic=topic),
                    "type": "synthetic"
                }
                queries.append(query)
                
                # 文档
                doc = {
                    "id": f"synthetic_doc_{i}",
                    "title": f"Introduction to {topic.title()}",
                    "content": f"{topic.title()} is an important concept in artificial intelligence and machine learning. It involves various techniques and methodologies that are widely used in modern applications. Understanding {topic} is crucial for developing effective AI systems.",
                    "source": "synthetic"
                }
                documents.append(doc)
            
            # 保存数据
            with open(dataset_dir / "queries.json", 'w', encoding='utf-8') as f:
                json.dump(queries, f, indent=2, ensure_ascii=False)
            
            with open(dataset_dir / "documents.json", 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"合成数据集创建完成: {len(queries)}个查询, {len(documents)}个文档")
            return True
            
        except Exception as e:
            self.logger.error(f"合成数据集创建失败: {str(e)}")
            return False
    
    def download_all(self, datasets: Optional[List[str]] = None, subset_size: int = 500):
        """
        下载所有数据集
        
        Args:
            datasets: 要下载的数据集列表
            subset_size: 每个数据集的子集大小
        """
        if datasets is None:
            datasets = ["squad", "synthetic"]  # 使用更可靠的数据集
        
        self.logger.info(f"开始下载数据集: {datasets}")
        
        download_methods = {
            "ms_marco": lambda: self.download_ms_marco(subset_size),
            "natural_questions": lambda: self.download_natural_questions(subset_size),
            "squad": lambda: self.download_squad(subset_size),
            "synthetic": lambda: self.create_synthetic_dataset(subset_size)
        }
        
        results = {}
        for dataset in datasets:
            if dataset in download_methods:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"处理数据集: {dataset}")
                self.logger.info(f"{'='*50}")
                
                results[dataset] = download_methods[dataset]()
            else:
                self.logger.warning(f"未知数据集: {dataset}")
                results[dataset] = False
        
        # 输出下载结果
        self.logger.info(f"\n{'='*50}")
        self.logger.info("下载结果总结:")
        for dataset, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            self.logger.info(f"  {dataset}: {status}")
        self.logger.info(f"{'='*50}")
        
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="下载HuggingFace RAG评估数据集")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["ms_marco", "natural_questions", "squad", "synthetic", "all"],
                       default=["squad", "synthetic"],
                       help="要下载的数据集")
    parser.add_argument("--data-dir", default="data", 
                       help="数据存储目录")
    parser.add_argument("--subset-size", type=int, default=500,
                       help="每个数据集的子集大小")
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = HFDatasetDownloader(args.data_dir)
    
    # 处理数据集列表
    if "all" in args.datasets:
        datasets = ["squad", "synthetic"]  # 使用可靠的数据集
    else:
        datasets = args.datasets
    
    # 下载数据集
    results = downloader.download_all(datasets, args.subset_size)
    
    # 输出结果
    success_count = sum(results.values())
    total_count = len(results)
    
    if success_count == total_count:
        print(f"\n🎉 所有数据集下载成功！({success_count}/{total_count})")
    else:
        print(f"\n⚠️  部分数据集下载失败 ({success_count}/{total_count})")


if __name__ == "__main__":
    main()
