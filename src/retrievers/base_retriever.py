"""
基础检索器 - Base Retriever

定义了所有检索器的基础接口和通用功能。
这是一个抽象基类，所有具体的检索器都应该继承这个类。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging

from ..models.retrieval_result import DocumentScore
from ..utils.logging import get_logger


class BaseRetriever(ABC):
    """
    检索器基类
    
    定义了所有检索器必须实现的接口和通用功能。
    采用模板方法模式，提供可扩展的检索框架。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化基础检索器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # 通用配置
        self.max_docs = self.config.get("max_docs", 100)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.0)
        self.timeout = self.config.get("timeout", 30.0)
        
        # 初始化状态
        self.is_initialized = False
        self.index_loaded = False
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'total_documents_retrieved': 0,
            'average_retrieval_time': 0.0,
            'cache_hits': 0
        }
        
        self.logger.info(f"{self.__class__.__name__} 初始化完成")
    
    @abstractmethod
    def _initialize_index(self):
        """
        初始化检索索引
        
        子类必须实现这个方法来初始化具体的检索索引
        """
        pass
    
    @abstractmethod
    def _search_index(self, query: str, k: int) -> List[DocumentScore]:
        """
        在索引中搜索
        
        Args:
            query: 查询字符串
            k: 返回结果数量
            
        Returns:
            List[DocumentScore]: 检索结果列表
        """
        pass
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[DocumentScore]:
        """
        检索文档
        
        这是主要的检索接口，实现了通用的检索流程
        
        Args:
            query: 查询字符串
            k: 返回结果数量，默认使用配置中的max_docs
            
        Returns:
            List[DocumentScore]: 检索结果列表
        """
        if not query or not query.strip():
            self.logger.warning("空查询，返回空结果")
            return []
        
        # 确保索引已初始化
        if not self.index_loaded:
            self._initialize_index()
            self.index_loaded = True
        
        # 设置返回数量
        if k is None:
            k = self.max_docs
        
        k = min(k, self.max_docs)  # 不超过最大限制
        
        self.logger.debug(f"开始检索，查询: {query[:50]}..., k={k}")
        
        try:
            # 预处理查询
            processed_query = self._preprocess_query(query)
            
            # 执行检索
            results = self._search_index(processed_query, k)
            
            # 后处理结果
            processed_results = self._postprocess_results(results, query)
            
            # 过滤低分结果
            filtered_results = self._filter_results(processed_results)
            
            # 更新统计信息
            self._update_stats(len(filtered_results))
            
            self.logger.debug(f"检索完成，返回{len(filtered_results)}个结果")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"检索过程出错: {str(e)}")
            return []
    
    def _preprocess_query(self, query: str) -> str:
        """
        预处理查询
        
        Args:
            query: 原始查询
            
        Returns:
            str: 预处理后的查询
        """
        # 基础预处理：去除多余空格，转小写
        processed = query.strip().lower()
        
        # 移除特殊字符（保留基本标点）
        import re
        processed = re.sub(r'[^\w\s\?\!\.\,]', '', processed)
        
        return processed
    
    def _postprocess_results(self, results: List[DocumentScore], original_query: str) -> List[DocumentScore]:
        """
        后处理检索结果
        
        Args:
            results: 原始检索结果
            original_query: 原始查询
            
        Returns:
            List[DocumentScore]: 后处理后的结果
        """
        if not results:
            return []
        
        # 设置检索方法标识
        for doc in results:
            if not doc.retrieval_method:
                doc.retrieval_method = self._get_retriever_type()
        
        # 计算相关性分数（如果没有的话）
        for doc in results:
            if doc.relevance_score == 0.0:
                doc.relevance_score = self._calculate_relevance_score(doc, original_query)
        
        return results
    
    def _filter_results(self, results: List[DocumentScore]) -> List[DocumentScore]:
        """
        过滤检索结果
        
        Args:
            results: 检索结果
            
        Returns:
            List[DocumentScore]: 过滤后的结果
        """
        # 按相似度阈值过滤
        filtered = [
            doc for doc in results 
            if doc.final_score >= self.similarity_threshold
        ]
        
        # 按分数排序
        filtered.sort(key=lambda x: x.final_score, reverse=True)
        
        return filtered
    
    def _calculate_relevance_score(self, doc: DocumentScore, query: str) -> float:
        """
        计算相关性分数
        
        Args:
            doc: 文档
            query: 查询
            
        Returns:
            float: 相关性分数
        """
        # 简化的相关性计算：基于关键词重叠
        if not doc.content or not query:
            return 0.0
        
        query_words = set(query.lower().split())
        doc_words = set(doc.content.lower().split())
        
        if not query_words:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)
        
        jaccard_score = intersection / union if union > 0 else 0.0
        
        # 结合原始分数
        combined_score = (doc.final_score + jaccard_score) / 2
        
        return min(1.0, combined_score)
    
    def _get_retriever_type(self) -> str:
        """
        获取检索器类型标识
        
        Returns:
            str: 检索器类型
        """
        class_name = self.__class__.__name__.lower()
        if 'dense' in class_name:
            return 'dense'
        elif 'sparse' in class_name:
            return 'sparse'
        elif 'hybrid' in class_name:
            return 'hybrid'
        else:
            return 'unknown'
    
    def _update_stats(self, num_results: int):
        """
        更新统计信息
        
        Args:
            num_results: 返回的结果数量
        """
        self.stats['total_queries'] += 1
        self.stats['total_documents_retrieved'] += num_results
        
        # 更新平均检索时间（简化实现）
        # 在实际实现中应该测量真实的检索时间
        self.stats['average_retrieval_time'] = 0.1  # 占位符
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取检索器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return {
            'retriever_type': self._get_retriever_type(),
            'is_initialized': self.is_initialized,
            'index_loaded': self.index_loaded,
            'config': self.config,
            'stats': self.stats.copy()
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_queries': 0,
            'total_documents_retrieved': 0,
            'average_retrieval_time': 0.0,
            'cache_hits': 0
        }
        self.logger.info("统计信息已重置")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        添加文档到索引
        
        Args:
            documents: 文档列表，每个文档包含id, content, title等字段
        """
        # 这是一个通用接口，具体实现由子类提供
        raise NotImplementedError("子类必须实现add_documents方法")
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        更新配置
        
        Args:
            new_config: 新的配置参数
        """
        self.config.update(new_config)
        
        # 更新通用配置
        self.max_docs = self.config.get("max_docs", self.max_docs)
        self.similarity_threshold = self.config.get("similarity_threshold", self.similarity_threshold)
        self.timeout = self.config.get("timeout", self.timeout)
        
        self.logger.info("配置已更新")
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(type={self._get_retriever_type()}, initialized={self.is_initialized})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"{self.__class__.__name__}("
                f"type={self._get_retriever_type()}, "
                f"max_docs={self.max_docs}, "
                f"threshold={self.similarity_threshold}, "
                f"initialized={self.is_initialized})")
