#!/usr/bin/env python3
"""
数据集下载脚本

下载和准备标准RAG评估数据集，包括：
1. MS MARCO - 微软大规模问答数据集
2. Natural Questions - Google自然问题数据集  
3. BEIR - 信息检索基准数据集
4. HotpotQA - 多跳推理问答数据集
5. FiQA - 金融问答数据集
"""

import os
import sys
import json
import gzip
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger

class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据集下载器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger = setup_logger("DatasetDownloader")
        
        # 数据集配置
        self.datasets_config = {
            "ms_marco": {
                "name": "MS MARCO Passage Ranking",
                "description": "微软大规模段落排序数据集",
                "urls": {
                    "queries_train": "https://msmarco.blob.core.windows.net/msmarcoranking/queries.train.tsv",
                    "queries_dev": "https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.tsv",
                    "collection": "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tsv",
                    "qrels_train": "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv",
                    "qrels_dev": "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv"
                },
                "size": "~3.2GB"
            },
            "natural_questions": {
                "name": "Natural Questions",
                "description": "Google自然问题数据集",
                "urls": {
                    "train": "https://storage.googleapis.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz",
                    "dev": "https://storage.googleapis.com/natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz"
                },
                "size": "~1.1GB"
            },
            "hotpot_qa": {
                "name": "HotpotQA",
                "description": "多跳推理问答数据集",
                "urls": {
                    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
                    "dev": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
                    "test": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json"
                },
                "size": "~500MB"
            },
            "fiqa": {
                "name": "FiQA",
                "description": "金融问答数据集",
                "urls": {
                    "train": "https://sites.google.com/view/fiqa/",
                    "test": "https://sites.google.com/view/fiqa/"
                },
                "size": "~50MB",
                "note": "需要手动下载"
            }
        }
    
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """
        下载文件
        
        Args:
            url: 下载链接
            filepath: 保存路径
            chunk_size: 块大小
            
        Returns:
            bool: 是否下载成功
        """
        try:
            self.logger.info(f"开始下载: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            self.logger.info(f"下载完成: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"下载失败 {url}: {str(e)}")
            return False
    
    def extract_archive(self, filepath: Path) -> bool:
        """
        解压文件
        
        Args:
            filepath: 压缩文件路径
            
        Returns:
            bool: 是否解压成功
        """
        try:
            extract_dir = filepath.parent / filepath.stem
            
            if filepath.suffix == '.gz':
                if filepath.stem.endswith('.tar'):
                    # tar.gz文件
                    with tarfile.open(filepath, 'r:gz') as tar:
                        tar.extractall(extract_dir)
                else:
                    # .jsonl.gz等文件
                    with gzip.open(filepath, 'rb') as f_in:
                        with open(extract_dir.with_suffix(''), 'wb') as f_out:
                            f_out.write(f_in.read())
            elif filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            
            self.logger.info(f"解压完成: {extract_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"解压失败 {filepath}: {str(e)}")
            return False
    
    def download_ms_marco(self) -> bool:
        """下载MS MARCO数据集"""
        self.logger.info("开始下载MS MARCO数据集...")
        
        dataset_dir = self.data_dir / "ms_marco"
        dataset_dir.mkdir(exist_ok=True)
        
        config = self.datasets_config["ms_marco"]
        success = True
        
        for name, url in config["urls"].items():
            filepath = dataset_dir / f"{name}.tsv"
            
            if filepath.exists():
                self.logger.info(f"文件已存在，跳过: {filepath}")
                continue
            
            if not self.download_file(url, filepath):
                success = False
        
        return success
    
    def download_natural_questions(self) -> bool:
        """下载Natural Questions数据集"""
        self.logger.info("开始下载Natural Questions数据集...")
        
        dataset_dir = self.data_dir / "natural_questions"
        dataset_dir.mkdir(exist_ok=True)
        
        config = self.datasets_config["natural_questions"]
        success = True
        
        for name, url in config["urls"].items():
            filename = url.split('/')[-1]
            filepath = dataset_dir / filename
            
            if filepath.exists():
                self.logger.info(f"文件已存在，跳过: {filepath}")
                continue
            
            if self.download_file(url, filepath):
                # 解压.gz文件
                if filepath.suffix == '.gz':
                    self.extract_archive(filepath)
            else:
                success = False
        
        return success
    
    def download_hotpot_qa(self) -> bool:
        """下载HotpotQA数据集"""
        self.logger.info("开始下载HotpotQA数据集...")
        
        dataset_dir = self.data_dir / "hotpot_qa"
        dataset_dir.mkdir(exist_ok=True)
        
        config = self.datasets_config["hotpot_qa"]
        success = True
        
        for name, url in config["urls"].items():
            filename = url.split('/')[-1]
            filepath = dataset_dir / filename
            
            if filepath.exists():
                self.logger.info(f"文件已存在，跳过: {filepath}")
                continue
            
            if not self.download_file(url, filepath):
                success = False
        
        return success
    
    def create_sample_datasets(self):
        """创建小样本数据集用于快速测试"""
        self.logger.info("创建样本数据集...")
        
        sample_dir = self.data_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        # 创建样本查询数据
        sample_queries = [
            {"id": "1", "query": "What is machine learning?", "type": "factual"},
            {"id": "2", "query": "How do neural networks work?", "type": "technical"},
            {"id": "3", "query": "Compare supervised and unsupervised learning", "type": "comparison"},
            {"id": "4", "query": "What are the applications of deep learning?", "type": "application"},
            {"id": "5", "query": "Explain the transformer architecture", "type": "detailed"}
        ]
        
        # 创建样本文档数据
        sample_documents = [
            {"id": "doc_1", "title": "Machine Learning Basics", 
             "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."},
            {"id": "doc_2", "title": "Neural Networks", 
             "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information."},
            {"id": "doc_3", "title": "Supervised Learning", 
             "content": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs."},
            {"id": "doc_4", "title": "Deep Learning Applications", 
             "content": "Deep learning has applications in computer vision, natural language processing, speech recognition, and many other fields."},
            {"id": "doc_5", "title": "Transformer Architecture", 
             "content": "The transformer is a neural network architecture that relies entirely on attention mechanisms to draw global dependencies."}
        ]
        
        # 保存样本数据
        with open(sample_dir / "queries.json", 'w', encoding='utf-8') as f:
            json.dump(sample_queries, f, indent=2, ensure_ascii=False)
        
        with open(sample_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(sample_documents, f, indent=2, ensure_ascii=False)
        
        self.logger.info("样本数据集创建完成")
    
    def download_all(self, datasets: Optional[List[str]] = None):
        """
        下载所有数据集
        
        Args:
            datasets: 要下载的数据集列表，None表示下载所有
        """
        if datasets is None:
            datasets = ["ms_marco", "natural_questions", "hotpot_qa"]
        
        self.logger.info(f"开始下载数据集: {datasets}")
        
        # 创建样本数据集
        self.create_sample_datasets()
        
        # 下载指定数据集
        download_methods = {
            "ms_marco": self.download_ms_marco,
            "natural_questions": self.download_natural_questions,
            "hotpot_qa": self.download_hotpot_qa
        }
        
        results = {}
        for dataset in datasets:
            if dataset in download_methods:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"下载数据集: {self.datasets_config[dataset]['name']}")
                self.logger.info(f"描述: {self.datasets_config[dataset]['description']}")
                self.logger.info(f"大小: {self.datasets_config[dataset]['size']}")
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
    
    parser = argparse.ArgumentParser(description="下载RAG评估数据集")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["ms_marco", "natural_questions", "hotpot_qa", "all"],
                       default=["all"],
                       help="要下载的数据集")
    parser.add_argument("--data-dir", default="data", 
                       help="数据存储目录")
    parser.add_argument("--sample-only", action="store_true",
                       help="只创建样本数据集")
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = DatasetDownloader(args.data_dir)
    
    if args.sample_only:
        downloader.create_sample_datasets()
        print("✅ 样本数据集创建完成！")
        return
    
    # 处理数据集列表
    if "all" in args.datasets:
        datasets = ["ms_marco", "natural_questions", "hotpot_qa"]
    else:
        datasets = args.datasets
    
    # 下载数据集
    results = downloader.download_all(datasets)
    
    # 输出结果
    success_count = sum(results.values())
    total_count = len(results)
    
    if success_count == total_count:
        print(f"\n🎉 所有数据集下载成功！({success_count}/{total_count})")
    else:
        print(f"\n⚠️  部分数据集下载失败 ({success_count}/{total_count})")
        print("请检查网络连接或重试失败的数据集")


if __name__ == "__main__":
    main()
