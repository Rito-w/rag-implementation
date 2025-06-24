"""
稠密检索器 - Dense Retriever

基于语义嵌入的稠密检索实现。使用预训练的语言模型将查询和文档
编码为稠密向量，通过向量相似度进行检索。

主要特点：
1. 语义理解能力强
2. 适合概念性和语义复杂的查询
3. 支持跨语言检索
4. 对同义词和释义敏感
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base_retriever import BaseRetriever
from ..models.retrieval_result import DocumentScore
from ..utils.logging import get_logger


class DenseRetriever(BaseRetriever):
    """
    稠密检索器
    
    使用预训练的语言模型进行语义检索。
    适合处理需要语义理解的复杂查询。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化稠密检索器
        
        Args:
            config: 配置参数字典
        """
        super().__init__(config)
        
        # 稠密检索特定配置
        self.model_name = self.config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.similarity_metric = self.config.get("similarity_metric", "cosine")
        self.embedding_dimension = self.config.get("dimension", 384)
        self.batch_size = self.config.get("batch_size", 32)
        self.normalize_embeddings = self.config.get("normalize", True)
        
        # 向量数据库配置
        self.vector_db_config = self.config.get("vector_db", {
            "type": "faiss",
            "index_type": "IVF",
            "nlist": 100,
            "nprobe": 10
        })
        
        # 初始化组件
        self.embedding_model = None
        self.vector_index = None
        self.document_store = {}  # document_id -> document_info
        
        self.logger.info("稠密检索器初始化完成")
    
    def _initialize_index(self):
        """初始化向量索引和嵌入模型"""
        try:
            # 初始化嵌入模型
            self._initialize_embedding_model()
            
            # 初始化向量索引
            self._initialize_vector_index()
            
            # 加载示例文档（用于演示）
            self._load_sample_documents()
            
            self.is_initialized = True
            self.logger.info("稠密检索索引初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化稠密检索索引失败: {str(e)}")
            raise
    
    def _initialize_embedding_model(self):
        """初始化嵌入模型"""
        try:
            # 尝试使用sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.model_name)
                self.logger.info(f"已加载sentence-transformers模型: {self.model_name}")
                return
            except ImportError:
                self.logger.warning("sentence-transformers未安装，使用简化的嵌入模型")
            
            # 简化的嵌入模型（用于演示）
            self.embedding_model = SimplifiedEmbeddingModel(self.embedding_dimension)
            self.logger.info("使用简化嵌入模型")
            
        except Exception as e:
            self.logger.error(f"初始化嵌入模型失败: {str(e)}")
            raise
    
    def _initialize_vector_index(self):
        """初始化向量索引"""
        try:
            # 尝试使用FAISS
            try:
                import faiss
                
                # 创建FAISS索引
                if self.vector_db_config["index_type"] == "IVF":
                    # IVF索引（适合大规模数据）
                    quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                    self.vector_index = faiss.IndexIVFFlat(
                        quantizer, 
                        self.embedding_dimension, 
                        self.vector_db_config["nlist"]
                    )
                else:
                    # 简单的平面索引
                    self.vector_index = faiss.IndexFlatIP(self.embedding_dimension)
                
                self.logger.info(f"已创建FAISS索引: {self.vector_db_config['index_type']}")
                return
                
            except ImportError:
                self.logger.warning("FAISS未安装，使用简化的向量索引")
            
            # 简化的向量索引（用于演示）
            self.vector_index = SimplifiedVectorIndex(self.embedding_dimension)
            self.logger.info("使用简化向量索引")
            
        except Exception as e:
            self.logger.error(f"初始化向量索引失败: {str(e)}")
            raise
    
    def _load_sample_documents(self):
        """加载示例文档用于演示"""
        sample_documents = [
            {
                "id": "doc_1",
                "title": "机器学习基础",
                "content": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。机器学习算法通过数据构建数学模型，以便对新数据进行预测或决策。"
            },
            {
                "id": "doc_2",
                "title": "深度学习介绍",
                "content": "深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的表示。深度学习在图像识别、自然语言处理和语音识别等领域取得了显著成功。"
            },
            {
                "id": "doc_3",
                "title": "自然语言处理",
                "content": "自然语言处理（NLP）是计算机科学和人工智能的一个分支，专注于计算机与人类语言之间的交互。NLP技术包括文本分析、语言翻译和情感分析等。"
            },
            {
                "id": "doc_4",
                "title": "检索增强生成",
                "content": "检索增强生成（RAG）是一种结合了信息检索和文本生成的技术。RAG系统首先检索相关文档，然后基于这些文档生成答案，提高了生成内容的准确性和可靠性。"
            },
            {
                "id": "doc_5",
                "title": "向量数据库",
                "content": "向量数据库是专门用于存储和检索高维向量的数据库系统。它们通过计算向量之间的相似度来快速找到最相关的数据，广泛应用于推荐系统和语义搜索。"
            }
        ]

        # 直接添加文档到索引，避免递归调用
        self._add_documents_internal(sample_documents)
        self.logger.info(f"已加载{len(sample_documents)}个示例文档")
    
    def _search_index(self, query: str, k: int) -> List[DocumentScore]:
        """
        在向量索引中搜索
        
        Args:
            query: 查询字符串
            k: 返回结果数量
            
        Returns:
            List[DocumentScore]: 检索结果列表
        """
        try:
            # 1. 将查询编码为向量
            query_embedding = self._encode_text(query)
            
            # 2. 在向量索引中搜索
            scores, doc_indices = self._search_vectors(query_embedding, k)
            
            # 3. 构建结果
            results = []
            for i, (score, doc_idx) in enumerate(zip(scores, doc_indices)):
                if doc_idx in self.document_store:
                    doc_info = self.document_store[doc_idx]
                    
                    doc_score = DocumentScore(
                        document_id=doc_info["id"],
                        content=doc_info["content"],
                        title=doc_info.get("title", ""),
                        dense_score=float(score),
                        final_score=float(score),
                        retrieval_method="dense",
                        source=doc_info.get("source", "sample_corpus"),
                        metadata=doc_info.get("metadata", {})
                    )
                    
                    results.append(doc_score)
            
            return results
            
        except Exception as e:
            self.logger.error(f"向量搜索失败: {str(e)}")
            return []
    
    def _encode_text(self, text: str) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 文本向量
        """
        if hasattr(self.embedding_model, 'encode'):
            # 使用sentence-transformers
            embedding = self.embedding_model.encode([text])[0]
        else:
            # 使用简化模型
            embedding = self.embedding_model.encode(text)
        
        # 归一化（如果需要）
        if self.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def _search_vectors(self, query_vector: np.ndarray, k: int) -> tuple:
        """
        在向量索引中搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            k: 返回数量
            
        Returns:
            tuple: (scores, indices)
        """
        if hasattr(self.vector_index, 'search'):
            # 使用FAISS
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
            scores, indices = self.vector_index.search(query_vector, k)
            return scores[0], indices[0]
        else:
            # 使用简化索引
            return self.vector_index.search(query_vector, k)
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        添加文档到索引

        Args:
            documents: 文档列表
        """
        if not self.is_initialized:
            self._initialize_index()

        self._add_documents_internal(documents)

    def _add_documents_internal(self, documents: List[Dict[str, Any]]):
        """
        内部文档添加方法，避免递归调用

        Args:
            documents: 文档列表
        """
        try:
            embeddings = []
            doc_indices = []

            for doc in documents:
                # 编码文档内容
                content = doc.get("content", "")
                if doc.get("title"):
                    content = doc["title"] + " " + content

                embedding = self._encode_text(content)
                embeddings.append(embedding)

                # 存储文档信息
                doc_idx = len(self.document_store)
                self.document_store[doc_idx] = doc
                doc_indices.append(doc_idx)

            # 添加到向量索引
            if embeddings:
                embeddings_array = np.array(embeddings).astype(np.float32)

                if hasattr(self.vector_index, 'is_trained'):
                    # 使用FAISS
                    if not self.vector_index.is_trained:
                        self.vector_index.train(embeddings_array)
                    self.vector_index.add(embeddings_array)
                else:
                    # 使用简化索引
                    self.vector_index.add(embeddings_array, doc_indices)

            self.logger.info(f"已添加{len(documents)}个文档到稠密检索索引")

        except Exception as e:
            self.logger.error(f"添加文档到索引失败: {str(e)}")
            raise


class SimplifiedEmbeddingModel:
    """简化的嵌入模型（用于演示）"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        np.random.seed(42)  # 确保可重现性
    
    def encode(self, text) -> np.ndarray:
        """简化的文本编码"""
        # 处理输入类型
        if isinstance(text, list):
            # 如果是列表，取第一个元素或连接所有元素
            if len(text) > 0:
                text = str(text[0]) if len(text) == 1 else " ".join(str(t) for t in text)
            else:
                text = ""
        elif not isinstance(text, str):
            text = str(text)

        # 基于文本哈希的简化嵌入
        text_hash = hash(text.lower()) % (2**31)
        np.random.seed(text_hash)

        # 生成随机向量
        embedding = np.random.normal(0, 1, self.dimension)

        # 添加一些基于文本特征的偏置
        word_count = len(text.split())
        char_count = len(text)

        embedding[0] += word_count * 0.1
        embedding[1] += char_count * 0.01

        return embedding.astype(np.float32)


class SimplifiedVectorIndex:
    """简化的向量索引（用于演示）"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.indices = []
    
    def add(self, vectors: np.ndarray, indices: List[int]):
        """添加向量到索引"""
        for i, vector in enumerate(vectors):
            self.vectors.append(vector)
            self.indices.append(indices[i])
    
    def search(self, query_vector: np.ndarray, k: int) -> tuple:
        """搜索最相似的向量"""
        if not self.vectors:
            return [], []
        
        # 计算余弦相似度
        similarities = []
        for vector in self.vectors:
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector) + 1e-8
            )
            similarities.append(similarity)
        
        # 获取top-k结果
        sorted_indices = np.argsort(similarities)[-k:][::-1]
        top_k_scores = [similarities[int(i)] for i in sorted_indices]
        top_k_doc_indices = [self.indices[int(i)] for i in sorted_indices]
        
        return top_k_scores, top_k_doc_indices
