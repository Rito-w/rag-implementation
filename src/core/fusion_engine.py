"""
智能融合引擎 - Intelligent Fusion Engine

这个模块实现了多路径检索结果的智能融合算法，是智能自适应RAG系统的关键组件。

主要功能：
1. 多检索器结果的智能合并
2. 基于查询特征的动态融合策略
3. 结果质量评估和多样性优化
4. 去重和排序优化

融合算法：
score_final(d) = w_d·score_dense(d) + w_s·score_sparse(d) + w_h·score_hybrid(d) +
                λ·diversity_bonus(d) + μ·quality_bonus(d)

其中 λ 是多样性权重，μ 是质量权重
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
import logging

from ..models.query_analysis import QueryAnalysis, QueryType
from ..models.weight_allocation import WeightAllocation, RetrievalStrategy
from ..models.retrieval_result import DocumentScore
from ..utils.logging import get_logger


class IntelligentFusionEngine:
    """
    智能融合引擎
    
    负责将多个检索器的结果进行智能融合，生成最终的排序结果。
    融合过程考虑查询特征、权重分配和结果质量等多个因素。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化智能融合引擎
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.logger = get_logger("IntelligentFusionEngine")
        
        # 融合参数
        self.fusion_method = self.config.get("fusion_method", "intelligent_weighted")
        self.diversity_lambda = self.config.get("diversity_lambda", 0.1)
        self.quality_mu = self.config.get("quality_mu", 0.2)
        self.max_results = self.config.get("max_results", 20)
        self.deduplication_threshold = self.config.get("deduplication_threshold", 0.9)
        self.min_score = self.config.get("min_score", 0.1)
        
        # 质量评估权重
        self.quality_weights = self.config.get("quality_factors", {
            "relevance_weight": 0.4,
            "freshness_weight": 0.2,
            "authority_weight": 0.2,
            "completeness_weight": 0.2
        })
        
        # 初始化融合策略
        self._initialize_fusion_strategies()
        
        self.logger.info("智能融合引擎初始化完成")
    
    def _initialize_fusion_strategies(self):
        """初始化不同的融合策略"""
        self.fusion_strategies = {
            'simple_weighted': self._simple_weighted_fusion,
            'intelligent_weighted': self._intelligent_weighted_fusion,
            'learned_fusion': self._learned_fusion,
            'rrf': self._reciprocal_rank_fusion,
            'comb_sum': self._comb_sum_fusion,
            'comb_mnz': self._comb_mnz_fusion
        }
    
    def fuse_results(
        self, 
        retrieval_results: Dict[str, List[DocumentScore]], 
        query_analysis: QueryAnalysis,
        weight_allocation: WeightAllocation
    ) -> List[DocumentScore]:
        """
        融合多个检索器的结果
        
        Args:
            retrieval_results: 各检索器的结果字典
            query_analysis: 查询分析结果
            weight_allocation: 权重分配结果
            
        Returns:
            List[DocumentScore]: 融合后的文档列表
        """
        start_time = time.time()
        
        self.logger.debug(f"开始融合检索结果，方法: {self.fusion_method}")
        
        # 1. 预处理检索结果
        processed_results = self._preprocess_results(retrieval_results)
        
        # 2. 选择融合策略
        fusion_func = self.fusion_strategies.get(
            self.fusion_method, 
            self._intelligent_weighted_fusion
        )
        
        # 3. 执行融合
        fused_results = fusion_func(processed_results, query_analysis, weight_allocation)
        
        # 4. 后处理：去重、质量评估、多样性优化
        final_results = self._postprocess_results(fused_results, query_analysis)
        
        processing_time = time.time() - start_time
        self.logger.debug(f"融合完成，耗时: {processing_time:.3f}秒，结果数量: {len(final_results)}")
        
        return final_results
    
    def _preprocess_results(self, retrieval_results: Dict[str, List[DocumentScore]]) -> Dict[str, List[DocumentScore]]:
        """
        预处理检索结果
        
        Args:
            retrieval_results: 原始检索结果
            
        Returns:
            Dict[str, List[DocumentScore]]: 预处理后的结果
        """
        processed_results = {}
        
        for method, documents in retrieval_results.items():
            if not documents:
                processed_results[method] = []
                continue
            
            # 分数归一化
            scores = [doc.final_score for doc in documents]
            if scores:
                max_score = max(scores)
                min_score = min(scores)
                score_range = max_score - min_score
                
                normalized_docs = []
                for doc in documents:
                    # 创建文档副本并更新分数
                    normalized_doc = DocumentScore(
                        document_id=doc.document_id,
                        content=doc.content,
                        title=doc.title,
                        dense_score=doc.dense_score,
                        sparse_score=doc.sparse_score,
                        hybrid_score=doc.hybrid_score,
                        final_score=((doc.final_score - min_score) / max(score_range, 1e-8)),
                        retrieval_method=doc.retrieval_method,
                        relevance_score=doc.relevance_score,
                        quality_score=doc.quality_score,
                        diversity_bonus=doc.diversity_bonus,
                        source=doc.source,
                        metadata=doc.metadata
                    )
                    normalized_docs.append(normalized_doc)
                
                processed_results[method] = normalized_docs
            else:
                processed_results[method] = documents
        
        return processed_results
    
    def _intelligent_weighted_fusion(
        self, 
        results: Dict[str, List[DocumentScore]], 
        query_analysis: QueryAnalysis,
        weight_allocation: WeightAllocation
    ) -> List[DocumentScore]:
        """
        智能加权融合 - 核心融合算法
        
        实现论文中的融合公式：
        score_final(d) = w_d·score_dense(d) + w_s·score_sparse(d) + w_h·score_hybrid(d) +
                        λ·diversity_bonus(d) + μ·quality_bonus(d)
        
        Args:
            results: 预处理后的检索结果
            query_analysis: 查询分析结果
            weight_allocation: 权重分配结果
            
        Returns:
            List[DocumentScore]: 融合后的文档列表
        """
        # 收集所有文档
        all_documents = {}  # document_id -> DocumentScore
        document_scores = defaultdict(dict)  # document_id -> {method: score}
        
        # 权重映射
        weights = {
            'dense': weight_allocation.dense_weight,
            'sparse': weight_allocation.sparse_weight,
            'hybrid': weight_allocation.hybrid_weight
        }
        
        # 收集各方法的文档和分数
        for method, documents in results.items():
            for doc in documents:
                doc_id = doc.document_id
                all_documents[doc_id] = doc
                document_scores[doc_id][method] = doc.final_score
        
        # 计算融合分数
        fused_documents = []
        for doc_id, doc in all_documents.items():
            # 1. 基础加权分数
            weighted_score = 0.0
            total_weight = 0.0
            
            for method, weight in weights.items():
                if method in document_scores[doc_id]:
                    weighted_score += weight * document_scores[doc_id][method]
                    total_weight += weight
            
            # 归一化权重（处理某些方法没有返回该文档的情况）
            if total_weight > 0:
                base_score = weighted_score / total_weight
            else:
                base_score = 0.0
            
            # 2. 多样性奖励
            diversity_bonus = self._calculate_diversity_bonus(doc, all_documents, query_analysis)
            
            # 3. 质量奖励
            quality_bonus = self._calculate_quality_bonus(doc, query_analysis)
            
            # 4. 最终融合分数
            final_score = (
                base_score + 
                self.diversity_lambda * diversity_bonus + 
                self.quality_mu * quality_bonus
            )
            
            # 更新文档的最终分数
            fused_doc = DocumentScore(
                document_id=doc.document_id,
                content=doc.content,
                title=doc.title,
                dense_score=document_scores[doc_id].get('dense', 0.0),
                sparse_score=document_scores[doc_id].get('sparse', 0.0),
                hybrid_score=document_scores[doc_id].get('hybrid', 0.0),
                final_score=final_score,
                retrieval_method=self._determine_primary_method(document_scores[doc_id], weights),
                relevance_score=doc.relevance_score,
                quality_score=quality_bonus,
                diversity_bonus=diversity_bonus,
                source=doc.source,
                metadata=doc.metadata
            )
            
            fused_documents.append(fused_doc)
        
        return fused_documents
    
    def _calculate_diversity_bonus(
        self, 
        doc: DocumentScore, 
        all_documents: Dict[str, DocumentScore],
        query_analysis: QueryAnalysis
    ) -> float:
        """
        计算多样性奖励
        
        Args:
            doc: 当前文档
            all_documents: 所有文档
            query_analysis: 查询分析结果
            
        Returns:
            float: 多样性奖励分数
        """
        if not doc.content:
            return 0.0
        
        # 简化的多样性计算：基于内容相似度
        diversity_score = 1.0
        content_words = set(doc.content.lower().split())
        
        # 与其他高分文档的差异性
        high_score_docs = [d for d in all_documents.values() if d.final_score > 0.7]
        
        if len(high_score_docs) > 1:
            similarities = []
            for other_doc in high_score_docs:
                if other_doc.document_id != doc.document_id:
                    other_words = set(other_doc.content.lower().split())
                    if content_words and other_words:
                        intersection = len(content_words & other_words)
                        union = len(content_words | other_words)
                        similarity = intersection / union if union > 0 else 0
                        similarities.append(similarity)
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                diversity_score = 1.0 - avg_similarity
        
        # 基于查询类型调整多样性重要性
        if query_analysis.query_type == QueryType.GLOBAL_ANALYTICAL:
            diversity_score *= 1.2  # 分析类查询更需要多样性
        elif query_analysis.query_type == QueryType.LOCAL_FACTUAL:
            diversity_score *= 0.8  # 事实类查询多样性不那么重要
        
        return min(1.0, max(0.0, diversity_score))
    
    def _calculate_quality_bonus(self, doc: DocumentScore, query_analysis: QueryAnalysis) -> float:
        """
        计算质量奖励
        
        Args:
            doc: 文档
            query_analysis: 查询分析结果
            
        Returns:
            float: 质量奖励分数
        """
        quality_score = 0.0
        
        # 1. 相关性评分（如果已有）
        if hasattr(doc, 'relevance_score') and doc.relevance_score > 0:
            quality_score += self.quality_weights["relevance_weight"] * doc.relevance_score
        
        # 2. 内容完整性评分
        content_completeness = self._assess_content_completeness(doc, query_analysis)
        quality_score += self.quality_weights["completeness_weight"] * content_completeness
        
        # 3. 权威性评分（简化实现）
        authority_score = self._assess_authority(doc)
        quality_score += self.quality_weights["authority_weight"] * authority_score
        
        # 4. 新鲜度评分（简化实现）
        freshness_score = self._assess_freshness(doc)
        quality_score += self.quality_weights["freshness_weight"] * freshness_score
        
        return min(1.0, max(0.0, quality_score))
    
    def _assess_content_completeness(self, doc: DocumentScore, query_analysis: QueryAnalysis) -> float:
        """评估内容完整性"""
        if not doc.content:
            return 0.0
        
        # 基于内容长度和关键词覆盖的简化评估
        content_length_score = min(1.0, len(doc.content) / 500)  # 假设500字符为完整
        
        # 关键词覆盖度
        query_terms = set(query_analysis.key_terms)
        content_words = set(doc.content.lower().split())
        
        if query_terms:
            coverage = len(query_terms & content_words) / len(query_terms)
        else:
            coverage = 0.5
        
        return (content_length_score + coverage) / 2
    
    def _assess_authority(self, doc: DocumentScore) -> float:
        """评估文档权威性"""
        # 简化实现：基于来源和元数据
        authority_score = 0.5  # 默认中等权威性
        
        if doc.source:
            # 基于来源的权威性评估（简化）
            authoritative_sources = ['wikipedia', 'arxiv', 'pubmed', 'ieee', 'acm']
            if any(source in doc.source.lower() for source in authoritative_sources):
                authority_score = 0.9
        
        return authority_score
    
    def _assess_freshness(self, doc: DocumentScore) -> float:
        """评估文档新鲜度"""
        # 简化实现：假设所有文档都是相对新鲜的
        return 0.7
    
    def _determine_primary_method(self, method_scores: Dict[str, float], weights: Dict[str, float]) -> str:
        """确定文档的主要检索方法"""
        if not method_scores:
            return "unknown"
        
        # 计算加权分数
        weighted_scores = {}
        for method, score in method_scores.items():
            if method in weights:
                weighted_scores[method] = score * weights[method]
        
        if weighted_scores:
            return max(weighted_scores, key=weighted_scores.get)
        else:
            return max(method_scores, key=method_scores.get)

    def _postprocess_results(self, fused_results: List[DocumentScore], query_analysis: QueryAnalysis) -> List[DocumentScore]:
        """
        后处理融合结果

        Args:
            fused_results: 融合后的文档列表
            query_analysis: 查询分析结果

        Returns:
            List[DocumentScore]: 后处理后的最终结果
        """
        # 1. 过滤低分文档
        filtered_results = [doc for doc in fused_results if doc.final_score >= self.min_score]

        # 2. 去重
        deduplicated_results = self._deduplicate_documents(filtered_results)

        # 3. 排序
        sorted_results = sorted(deduplicated_results, key=lambda x: x.final_score, reverse=True)

        # 4. 限制结果数量
        final_results = sorted_results[:self.max_results]

        # 5. 重新计算相关性分数
        for i, doc in enumerate(final_results):
            # 基于排名的相关性调整
            rank_bonus = (len(final_results) - i) / len(final_results)
            doc.relevance_score = min(1.0, doc.final_score * rank_bonus)

        return final_results

    def _deduplicate_documents(self, documents: List[DocumentScore]) -> List[DocumentScore]:
        """
        文档去重

        Args:
            documents: 文档列表

        Returns:
            List[DocumentScore]: 去重后的文档列表
        """
        if not documents:
            return []

        unique_documents = []
        seen_contents = set()

        for doc in documents:
            # 简化的去重：基于内容相似度
            content_hash = self._get_content_hash(doc.content)

            is_duplicate = False
            for seen_hash in seen_contents:
                similarity = self._calculate_content_similarity(content_hash, seen_hash)
                if similarity > self.deduplication_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_documents.append(doc)
                seen_contents.add(content_hash)

        return unique_documents

    def _get_content_hash(self, content: str) -> str:
        """获取内容的简化哈希"""
        if not content:
            return ""

        # 简化实现：使用前100个字符的词集合
        words = set(content.lower()[:100].split())
        return " ".join(sorted(words))

    def _calculate_content_similarity(self, hash1: str, hash2: str) -> float:
        """计算内容相似度"""
        if not hash1 or not hash2:
            return 0.0

        words1 = set(hash1.split())
        words2 = set(hash2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    # 其他融合策略实现
    def _simple_weighted_fusion(
        self,
        results: Dict[str, List[DocumentScore]],
        query_analysis: QueryAnalysis,
        weight_allocation: WeightAllocation
    ) -> List[DocumentScore]:
        """简单加权融合"""
        # 简化实现：直接按权重合并分数
        all_documents = {}
        weights = {
            'dense': weight_allocation.dense_weight,
            'sparse': weight_allocation.sparse_weight,
            'hybrid': weight_allocation.hybrid_weight
        }

        for method, documents in results.items():
            weight = weights.get(method, 0.0)
            for doc in documents:
                if doc.document_id not in all_documents:
                    all_documents[doc.document_id] = doc
                    all_documents[doc.document_id].final_score = 0.0

                all_documents[doc.document_id].final_score += weight * doc.final_score

        return list(all_documents.values())

    def _reciprocal_rank_fusion(
        self,
        results: Dict[str, List[DocumentScore]],
        query_analysis: QueryAnalysis,
        weight_allocation: WeightAllocation
    ) -> List[DocumentScore]:
        """倒数排名融合 (RRF)"""
        k = 60  # RRF参数
        document_scores = defaultdict(float)
        all_documents = {}

        for method, documents in results.items():
            # 按分数排序
            sorted_docs = sorted(documents, key=lambda x: x.final_score, reverse=True)

            for rank, doc in enumerate(sorted_docs, 1):
                rrf_score = 1.0 / (k + rank)
                document_scores[doc.document_id] += rrf_score
                all_documents[doc.document_id] = doc

        # 更新最终分数
        for doc_id, score in document_scores.items():
            all_documents[doc_id].final_score = score

        return list(all_documents.values())

    def _comb_sum_fusion(
        self,
        results: Dict[str, List[DocumentScore]],
        query_analysis: QueryAnalysis,
        weight_allocation: WeightAllocation
    ) -> List[DocumentScore]:
        """CombSUM融合"""
        document_scores = defaultdict(float)
        all_documents = {}

        for method, documents in results.items():
            for doc in documents:
                document_scores[doc.document_id] += doc.final_score
                all_documents[doc.document_id] = doc

        # 更新最终分数
        for doc_id, score in document_scores.items():
            all_documents[doc_id].final_score = score

        return list(all_documents.values())

    def _comb_mnz_fusion(
        self,
        results: Dict[str, List[DocumentScore]],
        query_analysis: QueryAnalysis,
        weight_allocation: WeightAllocation
    ) -> List[DocumentScore]:
        """CombMNZ融合"""
        document_scores = defaultdict(float)
        document_counts = defaultdict(int)
        all_documents = {}

        for method, documents in results.items():
            for doc in documents:
                document_scores[doc.document_id] += doc.final_score
                document_counts[doc.document_id] += 1
                all_documents[doc.document_id] = doc

        # 更新最终分数（分数 × 出现次数）
        for doc_id, score in document_scores.items():
            count = document_counts[doc_id]
            all_documents[doc_id].final_score = score * count

        return list(all_documents.values())

    def _learned_fusion(
        self,
        results: Dict[str, List[DocumentScore]],
        query_analysis: QueryAnalysis,
        weight_allocation: WeightAllocation
    ) -> List[DocumentScore]:
        """学习式融合（简化实现）"""
        # 这里可以实现更复杂的机器学习融合方法
        # 目前使用智能加权融合作为替代
        return self._intelligent_weighted_fusion(results, query_analysis, weight_allocation)
