"""
稀疏检索器 - Sparse Retriever

基于关键词匹配的稀疏检索实现。使用传统的信息检索算法如BM25
进行精确的关键词匹配检索。

主要特点：
1. 精确关键词匹配
2. 适合事实性和具体的查询
3. 计算效率高
4. 对术语和专有名词敏感
"""

import math
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any, Set
import logging

from .base_retriever import BaseRetriever
from ..models.retrieval_result import DocumentScore
from ..utils.logging import get_logger


class SparseRetriever(BaseRetriever):
    """
    稀疏检索器
    
    使用BM25算法进行关键词匹配检索。
    适合处理需要精确匹配的事实性查询。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化稀疏检索器
        
        Args:
            config: 配置参数字典
        """
        super().__init__(config)
        
        # BM25参数
        self.k1 = self.config.get("k1", 1.2)
        self.b = self.config.get("b", 0.75)
        self.algorithm = self.config.get("algorithm", "bm25")
        
        # 预处理配置
        self.preprocessing = self.config.get("preprocessing", {
            "stemming": True,
            "remove_stopwords": True,
            "min_term_freq": 1,
            "max_term_freq": 0.8
        })
        
        # 索引配置
        self.index_config = self.config.get("index", {
            "type": "inverted",
            "compression": True,
            "cache_size": 10000
        })
        
        # 初始化组件
        self.inverted_index = defaultdict(list)  # term -> [(doc_id, tf)]
        self.document_store = {}  # doc_id -> document_info
        self.document_lengths = {}  # doc_id -> document_length
        self.term_frequencies = defaultdict(int)  # term -> document_frequency
        self.average_doc_length = 0.0
        self.total_documents = 0
        
        # 停用词列表
        self.stop_words = self._get_stop_words()
        
        self.logger.info("稀疏检索器初始化完成")
    
    def _get_stop_words(self) -> Set[str]:
        """获取停用词列表"""
        # 简化的英文停用词列表
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        return stop_words
    
    def _initialize_index(self):
        """初始化倒排索引"""
        try:
            # 加载示例文档
            self._load_sample_documents()
            
            self.is_initialized = True
            self.logger.info("稀疏检索索引初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化稀疏检索索引失败: {str(e)}")
            raise
    
    def _load_sample_documents(self):
        """加载示例文档用于演示"""
        sample_documents = [
            {
                "id": "doc_1",
                "title": "Machine Learning Fundamentals",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."
            },
            {
                "id": "doc_2",
                "title": "Deep Learning Introduction", 
                "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has achieved remarkable success in image recognition, natural language processing, and speech recognition."
            },
            {
                "id": "doc_3",
                "title": "Natural Language Processing",
                "content": "Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on the interaction between computers and human language. NLP techniques include text analysis, language translation, and sentiment analysis."
            },
            {
                "id": "doc_4",
                "title": "Retrieval Augmented Generation",
                "content": "Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with text generation. RAG systems first retrieve relevant documents and then generate answers based on these documents, improving accuracy and reliability."
            },
            {
                "id": "doc_5",
                "title": "Vector Databases",
                "content": "Vector databases are specialized database systems designed to store and retrieve high-dimensional vectors efficiently. They calculate similarity between vectors to quickly find the most relevant data, widely used in recommendation systems and semantic search."
            }
        ]
        
        # 直接添加文档到索引，避免递归调用
        self._add_documents_internal(sample_documents)
        self.logger.info(f"已加载{len(sample_documents)}个示例文档")
    
    def _search_index(self, query: str, k: int) -> List[DocumentScore]:
        """
        在倒排索引中搜索
        
        Args:
            query: 查询字符串
            k: 返回结果数量
            
        Returns:
            List[DocumentScore]: 检索结果列表
        """
        try:
            # 1. 预处理查询
            query_terms = self._preprocess_text(query)
            
            if not query_terms:
                return []
            
            # 2. 计算文档分数
            doc_scores = self._calculate_bm25_scores(query_terms)
            
            # 3. 排序并获取top-k
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_docs = sorted_docs[:k]
            
            # 4. 构建结果
            results = []
            for doc_id, score in top_k_docs:
                if doc_id in self.document_store:
                    doc_info = self.document_store[doc_id]
                    
                    doc_score = DocumentScore(
                        document_id=doc_info["id"],
                        content=doc_info["content"],
                        title=doc_info.get("title", ""),
                        sparse_score=float(score),
                        final_score=float(score),
                        retrieval_method="sparse",
                        source=doc_info.get("source", "sample_corpus"),
                        metadata=doc_info.get("metadata", {})
                    )
                    
                    results.append(doc_score)
            
            return results
            
        except Exception as e:
            self.logger.error(f"稀疏检索搜索失败: {str(e)}")
            return []
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 预处理后的词项列表
        """
        # 转小写
        text = text.lower()
        
        # 移除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 分词
        terms = text.split()
        
        # 移除停用词
        if self.preprocessing.get("remove_stopwords", True):
            terms = [term for term in terms if term not in self.stop_words]
        
        # 过滤短词
        terms = [term for term in terms if len(term) > 2]
        
        # 简化的词干提取（移除常见后缀）
        if self.preprocessing.get("stemming", True):
            terms = [self._simple_stem(term) for term in terms]
        
        return terms
    
    def _simple_stem(self, word: str) -> str:
        """简化的词干提取"""
        # 移除常见英文后缀
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness', 'ment']
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        
        return word
    
    def _calculate_bm25_scores(self, query_terms: List[str]) -> Dict[str, float]:
        """
        计算BM25分数
        
        Args:
            query_terms: 查询词项列表
            
        Returns:
            Dict[str, float]: 文档ID到分数的映射
        """
        doc_scores = defaultdict(float)
        
        # 计算查询词频
        query_term_freq = Counter(query_terms)
        
        for term, qtf in query_term_freq.items():
            if term in self.inverted_index:
                # 文档频率
                df = self.term_frequencies[term]
                
                # IDF计算
                idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
                
                # 遍历包含该词的文档
                for doc_id, tf in self.inverted_index[term]:
                    # 文档长度
                    doc_length = self.document_lengths[doc_id]
                    
                    # BM25分数计算
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * (doc_length / self.average_doc_length)
                    )
                    
                    score = idf * (numerator / denominator)
                    doc_scores[doc_id] += score
        
        return doc_scores
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        添加文档到索引

        Args:
            documents: 文档列表
        """
        self._add_documents_internal(documents)

    def _add_documents_internal(self, documents: List[Dict[str, Any]]):
        """
        内部文档添加方法，避免递归调用

        Args:
            documents: 文档列表
        """
        try:
            total_length = 0

            for doc in documents:
                doc_id = doc["id"]

                # 合并标题和内容
                content = doc.get("content", "")
                if doc.get("title"):
                    content = doc["title"] + " " + content

                # 预处理文档
                terms = self._preprocess_text(content)

                # 计算词频
                term_freq = Counter(terms)
                doc_length = len(terms)

                # 存储文档信息
                self.document_store[doc_id] = doc
                self.document_lengths[doc_id] = doc_length
                total_length += doc_length

                # 构建倒排索引
                for term, tf in term_freq.items():
                    self.inverted_index[term].append((doc_id, tf))
                    self.term_frequencies[term] += 1

            # 更新统计信息
            self.total_documents += len(documents)
            self.average_doc_length = (
                (self.average_doc_length * (self.total_documents - len(documents)) + total_length)
                / self.total_documents
            )

            self.logger.info(f"已添加{len(documents)}个文档到稀疏检索索引")

        except Exception as e:
            self.logger.error(f"添加文档到稀疏索引失败: {str(e)}")
            raise
    
    def get_term_statistics(self) -> Dict[str, Any]:
        """
        获取词项统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "total_terms": len(self.inverted_index),
            "total_documents": self.total_documents,
            "average_doc_length": self.average_doc_length,
            "most_frequent_terms": dict(
                Counter(self.term_frequencies).most_common(10)
            ),
            "index_size": sum(len(postings) for postings in self.inverted_index.values())
        }
    
    def search_term(self, term: str) -> List[tuple]:
        """
        搜索特定词项
        
        Args:
            term: 搜索词项
            
        Returns:
            List[tuple]: (doc_id, tf) 列表
        """
        processed_term = self._preprocess_text(term)
        if processed_term:
            return self.inverted_index.get(processed_term[0], [])
        return []
