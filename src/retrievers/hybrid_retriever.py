"""
混合检索器 - Hybrid Retriever

结合稠密检索和稀疏检索的混合检索实现。
通过预设的权重组合两种检索方法的结果。

主要特点：
1. 结合语义理解和精确匹配
2. 平衡召回率和精确率
3. 适合多样化的查询类型
4. 可配置的融合策略
"""

from typing import Dict, List, Optional, Any
import logging

from .base_retriever import BaseRetriever
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
from ..models.retrieval_result import DocumentScore
from ..utils.logging import get_logger


class HybridRetriever(BaseRetriever):
    """
    混合检索器
    
    组合稠密检索和稀疏检索的结果，提供平衡的检索性能。
    支持多种融合策略。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化混合检索器
        
        Args:
            config: 配置参数字典
        """
        super().__init__(config)
        
        # 混合检索配置
        self.dense_weight = self.config.get("dense_weight", 0.6)
        self.sparse_weight = self.config.get("sparse_weight", 0.4)
        self.fusion_method = self.config.get("fusion_method", "linear")
        
        # RRF参数（如果使用RRF融合）
        self.rrf_k = self.config.get("rrf_k", 60)
        
        # 初始化子检索器
        self.dense_retriever = None
        self.sparse_retriever = None
        
        # 融合策略映射
        self.fusion_strategies = {
            'linear': self._linear_fusion,
            'rrf': self._reciprocal_rank_fusion,
            'harmonic': self._harmonic_fusion,
            'max': self._max_fusion,
            'min': self._min_fusion
        }
        
        self.logger.info("混合检索器初始化完成")
    
    def _initialize_index(self):
        """初始化混合检索器的子组件"""
        try:
            # 初始化稠密检索器
            dense_config = self.config.get("dense_config", {})
            self.dense_retriever = DenseRetriever(dense_config)
            
            # 初始化稀疏检索器
            sparse_config = self.config.get("sparse_config", {})
            self.sparse_retriever = SparseRetriever(sparse_config)
            
            self.is_initialized = True
            self.logger.info("混合检索器子组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化混合检索器失败: {str(e)}")
            raise
    
    def _search_index(self, query: str, k: int) -> List[DocumentScore]:
        """
        执行混合检索
        
        Args:
            query: 查询字符串
            k: 返回结果数量
            
        Returns:
            List[DocumentScore]: 混合检索结果
        """
        try:
            # 1. 分别执行稠密和稀疏检索
            dense_results = self.dense_retriever.retrieve(query, k * 2)  # 获取更多结果用于融合
            sparse_results = self.sparse_retriever.retrieve(query, k * 2)
            
            # 2. 选择融合策略
            fusion_func = self.fusion_strategies.get(
                self.fusion_method, 
                self._linear_fusion
            )
            
            # 3. 执行融合
            fused_results = fusion_func(dense_results, sparse_results, k)
            
            # 4. 更新检索方法标识
            for doc in fused_results:
                doc.retrieval_method = "hybrid"
            
            return fused_results
            
        except Exception as e:
            self.logger.error(f"混合检索失败: {str(e)}")
            return []
    
    def _linear_fusion(self, dense_results: List[DocumentScore], 
                      sparse_results: List[DocumentScore], k: int) -> List[DocumentScore]:
        """
        线性融合策略
        
        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            k: 返回结果数量
            
        Returns:
            List[DocumentScore]: 融合后的结果
        """
        # 收集所有文档
        all_docs = {}
        
        # 处理稠密检索结果
        for doc in dense_results:
            doc_id = doc.document_id
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
                all_docs[doc_id].hybrid_score = 0.0
            
            # 线性组合分数
            all_docs[doc_id].hybrid_score += self.dense_weight * doc.dense_score
        
        # 处理稀疏检索结果
        for doc in sparse_results:
            doc_id = doc.document_id
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
                all_docs[doc_id].hybrid_score = 0.0
            
            # 线性组合分数
            all_docs[doc_id].hybrid_score += self.sparse_weight * doc.sparse_score
        
        # 更新最终分数并排序
        for doc in all_docs.values():
            doc.final_score = doc.hybrid_score
        
        # 排序并返回top-k
        sorted_docs = sorted(all_docs.values(), key=lambda x: x.final_score, reverse=True)
        return sorted_docs[:k]
    
    def _reciprocal_rank_fusion(self, dense_results: List[DocumentScore],
                               sparse_results: List[DocumentScore], k: int) -> List[DocumentScore]:
        """
        倒数排名融合 (RRF)
        
        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            k: 返回结果数量
            
        Returns:
            List[DocumentScore]: 融合后的结果
        """
        doc_scores = {}
        all_docs = {}
        
        # 处理稠密检索结果
        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc.document_id
            rrf_score = 1.0 / (self.rrf_k + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
                all_docs[doc_id] = doc
            
            doc_scores[doc_id] += rrf_score
        
        # 处理稀疏检索结果
        for rank, doc in enumerate(sparse_results, 1):
            doc_id = doc.document_id
            rrf_score = 1.0 / (self.rrf_k + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
                all_docs[doc_id] = doc
            
            doc_scores[doc_id] += rrf_score
        
        # 更新最终分数
        for doc_id, score in doc_scores.items():
            all_docs[doc_id].final_score = score
            all_docs[doc_id].hybrid_score = score
        
        # 排序并返回top-k
        sorted_docs = sorted(all_docs.values(), key=lambda x: x.final_score, reverse=True)
        return sorted_docs[:k]
    
    def _harmonic_fusion(self, dense_results: List[DocumentScore],
                        sparse_results: List[DocumentScore], k: int) -> List[DocumentScore]:
        """
        调和平均融合
        
        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            k: 返回结果数量
            
        Returns:
            List[DocumentScore]: 融合后的结果
        """
        all_docs = {}
        dense_scores = {}
        sparse_scores = {}
        
        # 收集稠密检索分数
        for doc in dense_results:
            doc_id = doc.document_id
            dense_scores[doc_id] = doc.dense_score
            all_docs[doc_id] = doc
        
        # 收集稀疏检索分数
        for doc in sparse_results:
            doc_id = doc.document_id
            sparse_scores[doc_id] = doc.sparse_score
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
        
        # 计算调和平均
        for doc_id, doc in all_docs.items():
            dense_score = dense_scores.get(doc_id, 0.0)
            sparse_score = sparse_scores.get(doc_id, 0.0)
            
            if dense_score > 0 and sparse_score > 0:
                # 调和平均
                harmonic_mean = 2 * dense_score * sparse_score / (dense_score + sparse_score)
            else:
                # 如果只有一个分数，使用该分数
                harmonic_mean = max(dense_score, sparse_score)
            
            doc.final_score = harmonic_mean
            doc.hybrid_score = harmonic_mean
        
        # 排序并返回top-k
        sorted_docs = sorted(all_docs.values(), key=lambda x: x.final_score, reverse=True)
        return sorted_docs[:k]
    
    def _max_fusion(self, dense_results: List[DocumentScore],
                   sparse_results: List[DocumentScore], k: int) -> List[DocumentScore]:
        """
        最大值融合
        
        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            k: 返回结果数量
            
        Returns:
            List[DocumentScore]: 融合后的结果
        """
        all_docs = {}
        
        # 收集所有文档，取最大分数
        for doc in dense_results + sparse_results:
            doc_id = doc.document_id
            current_score = max(doc.dense_score, doc.sparse_score)
            
            if doc_id not in all_docs or current_score > all_docs[doc_id].final_score:
                all_docs[doc_id] = doc
                all_docs[doc_id].final_score = current_score
                all_docs[doc_id].hybrid_score = current_score
        
        # 排序并返回top-k
        sorted_docs = sorted(all_docs.values(), key=lambda x: x.final_score, reverse=True)
        return sorted_docs[:k]
    
    def _min_fusion(self, dense_results: List[DocumentScore],
                   sparse_results: List[DocumentScore], k: int) -> List[DocumentScore]:
        """
        最小值融合（只返回两种方法都检索到的文档）
        
        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            k: 返回结果数量
            
        Returns:
            List[DocumentScore]: 融合后的结果
        """
        dense_docs = {doc.document_id: doc for doc in dense_results}
        sparse_docs = {doc.document_id: doc for doc in sparse_results}
        
        # 只保留两种方法都检索到的文档
        common_docs = []
        for doc_id in dense_docs:
            if doc_id in sparse_docs:
                doc = dense_docs[doc_id]
                # 取最小分数
                min_score = min(doc.dense_score, sparse_docs[doc_id].sparse_score)
                doc.final_score = min_score
                doc.hybrid_score = min_score
                common_docs.append(doc)
        
        # 排序并返回top-k
        sorted_docs = sorted(common_docs, key=lambda x: x.final_score, reverse=True)
        return sorted_docs[:k]
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        添加文档到混合索引
        
        Args:
            documents: 文档列表
        """
        if not self.is_initialized:
            self._initialize_index()
        
        try:
            # 添加到稠密检索器
            if self.dense_retriever:
                self.dense_retriever.add_documents(documents)
            
            # 添加到稀疏检索器
            if self.sparse_retriever:
                self.sparse_retriever.add_documents(documents)
            
            self.logger.info(f"已添加{len(documents)}个文档到混合检索索引")
            
        except Exception as e:
            self.logger.error(f"添加文档到混合索引失败: {str(e)}")
            raise
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """
        获取融合统计信息
        
        Returns:
            Dict[str, Any]: 融合统计信息
        """
        stats = {
            'fusion_method': self.fusion_method,
            'dense_weight': self.dense_weight,
            'sparse_weight': self.sparse_weight,
            'rrf_k': self.rrf_k if self.fusion_method == 'rrf' else None
        }
        
        # 添加子检索器统计
        if self.dense_retriever:
            stats['dense_retriever_stats'] = self.dense_retriever.get_stats()
        
        if self.sparse_retriever:
            stats['sparse_retriever_stats'] = self.sparse_retriever.get_stats()
        
        return stats
    
    def update_fusion_weights(self, dense_weight: float, sparse_weight: float):
        """
        更新融合权重
        
        Args:
            dense_weight: 稠密检索权重
            sparse_weight: 稀疏检索权重
        """
        # 归一化权重
        total_weight = dense_weight + sparse_weight
        if total_weight > 0:
            self.dense_weight = dense_weight / total_weight
            self.sparse_weight = sparse_weight / total_weight
        else:
            self.dense_weight = 0.5
            self.sparse_weight = 0.5
        
        self.logger.info(f"融合权重已更新: dense={self.dense_weight:.2f}, sparse={self.sparse_weight:.2f}")
