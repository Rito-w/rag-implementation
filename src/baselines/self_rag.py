"""
Self-RAG 基线方法实现

论文: Self-RAG: Learning to Retrieve, Generate, and Critique
描述: 自适应检索增强生成，基于反思机制

核心组件:
1. 检索决策器 (Retrieval Decider)
2. 相关性评估器 (Relevance Evaluator)
3. 质量评估器 (Quality Evaluator)
4. 答案生成器 (Answer Generator)
"""

import re
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

from ..utils.logging import get_logger


@dataclass
class BaselineResult:
    """基线方法的简化结果类"""
    query: str
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    processing_time: float
    overall_confidence: float
    success: bool
    error_message: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalDecision:
    """检索决策结果"""
    should_retrieve: bool
    confidence: float
    reasoning: str
    token: str  # "[Retrieve]" or "[No Retrieve]"
    factual_score: float
    complexity_score: float
    knowledge_score: float


@dataclass
class RelevanceAssessment:
    """相关性评估结果"""
    is_relevant: bool
    relevance_score: float
    semantic_score: float
    keyword_score: float
    coverage_score: float
    token: str  # "[Relevant]" or "[Irrelevant]"


@dataclass
class SupportAssessment:
    """支持度评估结果"""
    support_level: str  # "[Fully Supported]", "[Partially Supported]", "[No Support]"
    support_score: float
    evidence_count: int
    fact_matches: List[str]


@dataclass
class UsefulnessAssessment:
    """有用性评估结果"""
    is_useful: bool
    usefulness_score: float
    completeness_score: float
    accuracy_score: float
    token: str  # "[Useful]" or "[Not Useful]"


@dataclass
class QualityAssessment:
    """质量评估结果"""
    support_assessment: SupportAssessment
    usefulness_assessment: UsefulnessAssessment
    overall_quality: float


@dataclass
class SelfRAGResult:
    """Self-RAG完整结果"""
    query: str
    retrieve_decision: RetrievalDecision
    documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    answer: str
    quality_assessment: QualityAssessment
    processing_time: float
    reflection_tokens: List[str]
    success: bool
    error_message: Optional[str] = None


class RetrievalDecider:
    """检索决策器 - 决定是否需要检索外部知识"""

    def __init__(self):
        self.factual_keywords = [
            'what is', 'who is', 'when did', 'where is', 'how many',
            'which', 'define', 'explain', 'describe', 'tell me about'
        ]

        self.complex_patterns = [
            r'compare.*with', r'difference between', r'similarities.*differences',
            r'advantages.*disadvantages', r'pros.*cons', r'better.*worse'
        ]

        self.knowledge_domains = [
            'history', 'science', 'technology', 'medicine', 'law',
            'economics', 'politics', 'geography', 'biology', 'physics'
        ]

    def decide(self, query: str) -> RetrievalDecision:
        """决定是否需要检索"""

        # 1. 事实性查询检测
        factual_score = self._detect_factual_query(query)

        # 2. 复杂度评估
        complexity_score = self._assess_complexity(query)

        # 3. 知识需求评估
        knowledge_score = self._assess_knowledge_need(query)

        # 4. 综合决策
        decision_score = (
            0.4 * factual_score +
            0.3 * complexity_score +
            0.3 * knowledge_score
        )

        should_retrieve = decision_score > 0.3  # 降低阈值，更容易触发检索

        # 5. 生成推理解释
        reasoning = self._generate_reasoning(factual_score, complexity_score, knowledge_score)

        return RetrievalDecision(
            should_retrieve=should_retrieve,
            confidence=decision_score,
            reasoning=reasoning,
            token="[Retrieve]" if should_retrieve else "[No Retrieve]",
            factual_score=factual_score,
            complexity_score=complexity_score,
            knowledge_score=knowledge_score
        )

    def _detect_factual_query(self, query: str) -> float:
        """检测事实性查询"""
        query_lower = query.lower()

        # 检查事实性关键词
        factual_count = sum(1 for keyword in self.factual_keywords
                           if keyword in query_lower)

        # 检查疑问词
        question_words = ['what', 'who', 'when', 'where', 'why', 'how']
        question_count = sum(1 for word in question_words
                           if word in query_lower.split())

        # 检查是否以问号结尾
        has_question_mark = query.strip().endswith('?')

        # 综合评分
        score = (
            min(factual_count / 2.0, 1.0) * 0.4 +
            min(question_count / 2.0, 1.0) * 0.4 +
            (1.0 if has_question_mark else 0.0) * 0.2
        )

        return min(score, 1.0)

    def _assess_complexity(self, query: str) -> float:
        """评估查询复杂度"""
        # 1. 长度复杂度
        word_count = len(query.split())
        length_score = min(word_count / 20.0, 1.0)

        # 2. 语法复杂度
        pattern_count = sum(1 for pattern in self.complex_patterns
                           if re.search(pattern, query.lower()))
        syntax_score = min(pattern_count / 2.0, 1.0)

        # 3. 实体复杂度
        entities = self._extract_entities(query)
        entity_score = min(len(entities) / 5.0, 1.0)

        return (length_score + syntax_score + entity_score) / 3.0

    def _assess_knowledge_need(self, query: str) -> float:
        """评估知识需求"""
        query_lower = query.lower()

        # 检查知识领域关键词
        domain_count = sum(1 for domain in self.knowledge_domains
                          if domain in query_lower)

        # 检查专业术语（大写字母开头的词）
        words = query.split()
        technical_terms = sum(1 for word in words
                             if word[0].isupper() and len(word) > 3)

        # 综合评分
        domain_score = min(domain_count / 3.0, 1.0)
        technical_score = min(technical_terms / 5.0, 1.0)

        return (domain_score + technical_score) / 2.0

    def _extract_entities(self, query: str) -> List[str]:
        """提取实体"""
        if nlp:
            doc = nlp(query)
            return [ent.text for ent in doc.ents]
        else:
            # 简单的实体提取：大写字母开头的词
            words = query.split()
            return [word for word in words if word[0].isupper()]

    def _generate_reasoning(self, factual_score: float, complexity_score: float,
                          knowledge_score: float) -> str:
        """生成推理解释"""
        reasons = []

        if factual_score > 0.7:
            reasons.append("查询包含明显的事实性问题")
        if complexity_score > 0.7:
            reasons.append("查询结构复杂，需要多方面信息")
        if knowledge_score > 0.7:
            reasons.append("查询涉及专业知识领域")

        if not reasons:
            reasons.append("查询相对简单，可能不需要外部检索")

        return "; ".join(reasons)


class RelevanceEvaluator:
    """相关性评估器 - 评估检索文档与查询的相关性"""

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.relevance_threshold = 0.3  # 降低阈值，更容易判定为相关

    def evaluate(self, query: str, document: str) -> RelevanceAssessment:
        """评估文档相关性"""

        # 1. 语义相似度
        semantic_score = self._compute_semantic_similarity(query, document)

        # 2. 关键词匹配
        keyword_score = self._compute_keyword_overlap(query, document)

        # 3. 信息覆盖度
        coverage_score = self._compute_information_coverage(query, document)

        # 4. 综合评分
        relevance_score = (
            0.5 * semantic_score +
            0.3 * keyword_score +
            0.2 * coverage_score
        )

        is_relevant = relevance_score > self.relevance_threshold

        return RelevanceAssessment(
            is_relevant=is_relevant,
            relevance_score=relevance_score,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            coverage_score=coverage_score,
            token="[Relevant]" if is_relevant else "[Irrelevant]"
        )

    def filter_relevant_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤相关文档"""
        relevant_docs = []
        for doc in documents:
            doc_text = doc.get('content', '') or doc.get('text', '')
            assessment = self.evaluate(query, doc_text)
            if assessment.is_relevant:
                doc['relevance_assessment'] = assessment
                relevant_docs.append(doc)
        return relevant_docs

    def _compute_semantic_similarity(self, query: str, document: str) -> float:
        """计算语义相似度"""
        if self.embedding_model:
            try:
                query_embedding = self.embedding_model.encode([query])
                doc_embedding = self.embedding_model.encode([document])

                # 计算余弦相似度
                similarity = np.dot(query_embedding[0], doc_embedding[0]) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(doc_embedding[0])
                )
                return max(0.0, float(similarity))
            except:
                pass

        # 简单的词汇重叠作为后备
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        if not query_words or not doc_words:
            return 0.0

        overlap = len(query_words.intersection(doc_words))
        return overlap / len(query_words.union(doc_words))

    def _compute_keyword_overlap(self, query: str, document: str) -> float:
        """计算关键词重叠度"""
        query_words = set(word.lower() for word in query.split() if len(word) > 3)
        doc_words = set(word.lower() for word in document.split() if len(word) > 3)

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(doc_words))
        return overlap / len(query_words)

    def _compute_information_coverage(self, query: str, document: str) -> float:
        """计算信息覆盖度"""
        # 提取查询中的关键概念
        query_concepts = self._extract_key_concepts(query)
        doc_concepts = self._extract_key_concepts(document)

        if not query_concepts:
            return 0.0

        covered_concepts = sum(1 for concept in query_concepts
                             if any(concept.lower() in doc_concept.lower()
                                   for doc_concept in doc_concepts))

        return covered_concepts / len(query_concepts)

    def _extract_key_concepts(self, text: str) -> List[str]:
        """提取关键概念"""
        if nlp:
            doc = nlp(text)
            # 提取名词短语和实体
            concepts = []
            concepts.extend([ent.text for ent in doc.ents])
            concepts.extend([chunk.text for chunk in doc.noun_chunks])
            return list(set(concepts))
        else:
            # 简单的关键词提取：长度大于3的词
            words = text.split()
            return [word for word in words if len(word) > 3]


class QualityEvaluator:
    """质量评估器 - 评估生成答案的支持度和有用性"""

    def __init__(self):
        self.support_threshold = 0.6
        self.usefulness_threshold = 0.7

    def evaluate(self, query: str, documents: List[Dict[str, Any]], answer: str) -> QualityAssessment:
        """评估答案质量"""

        # 1. 支持度评估
        support_assessment = self._evaluate_support(answer, documents)

        # 2. 有用性评估
        usefulness_assessment = self._evaluate_usefulness(query, answer)

        # 3. 计算整体质量
        overall_quality = (
            0.6 * support_assessment.support_score +
            0.4 * usefulness_assessment.usefulness_score
        )

        return QualityAssessment(
            support_assessment=support_assessment,
            usefulness_assessment=usefulness_assessment,
            overall_quality=overall_quality
        )

    def _evaluate_support(self, answer: str, documents: List[Dict[str, Any]]) -> SupportAssessment:
        """评估答案支持度"""
        if not documents:
            return SupportAssessment(
                support_level="[No Support]",
                support_score=0.0,
                evidence_count=0,
                fact_matches=[]
            )

        # 1. 提取答案中的事实
        answer_facts = self._extract_facts(answer)

        # 2. 在文档中查找支持证据
        fact_matches = []
        for fact in answer_facts:
            for doc in documents:
                doc_text = doc.get('content', '') or doc.get('text', '')
                if self._find_fact_support(fact, doc_text):
                    fact_matches.append(fact)
                    break

        # 3. 计算支持度
        if not answer_facts:
            support_score = 0.5  # 没有具体事实，给中等分
        else:
            support_score = len(fact_matches) / len(answer_facts)

        # 4. 语义对齐度
        semantic_alignment = self._compute_semantic_alignment(answer, documents)

        # 5. 综合评分
        final_support_score = 0.7 * support_score + 0.3 * semantic_alignment

        # 6. 支持级别判定
        if final_support_score > 0.8:
            support_level = "[Fully Supported]"
        elif final_support_score > 0.5:
            support_level = "[Partially Supported]"
        else:
            support_level = "[No Support]"

        return SupportAssessment(
            support_level=support_level,
            support_score=final_support_score,
            evidence_count=len(fact_matches),
            fact_matches=fact_matches
        )

    def _evaluate_usefulness(self, query: str, answer: str) -> UsefulnessAssessment:
        """评估答案有用性"""

        # 1. 完整性评估
        completeness_score = self._assess_completeness(query, answer)

        # 2. 准确性评估
        accuracy_score = self._assess_accuracy(query, answer)

        # 3. 综合有用性
        usefulness_score = 0.6 * completeness_score + 0.4 * accuracy_score

        is_useful = usefulness_score > self.usefulness_threshold

        return UsefulnessAssessment(
            is_useful=is_useful,
            usefulness_score=usefulness_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            token="[Useful]" if is_useful else "[Not Useful]"
        )

    def _extract_facts(self, text: str) -> List[str]:
        """提取文本中的事实"""
        # 简单的事实提取：句子分割
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        # 过滤掉太短的句子
        facts = [s for s in sentences if len(s.split()) > 3]

        return facts

    def _find_fact_support(self, fact: str, document: str) -> bool:
        """在文档中查找事实支持"""
        fact_words = set(word.lower() for word in fact.split() if len(word) > 3)
        doc_words = set(word.lower() for word in document.split())

        # 如果事实中的大部分关键词都在文档中出现，认为有支持
        if not fact_words:
            return False

        overlap = len(fact_words.intersection(doc_words))
        return overlap / len(fact_words) > 0.6

    def _compute_semantic_alignment(self, answer: str, documents: List[Dict[str, Any]]) -> float:
        """计算语义对齐度"""
        if not documents:
            return 0.0

        answer_words = set(word.lower() for word in answer.split())

        total_alignment = 0.0
        for doc in documents:
            doc_text = doc.get('content', '') or doc.get('text', '')
            doc_words = set(word.lower() for word in doc_text.split())

            if answer_words and doc_words:
                overlap = len(answer_words.intersection(doc_words))
                alignment = overlap / len(answer_words.union(doc_words))
                total_alignment += alignment

        return total_alignment / len(documents) if documents else 0.0

    def _assess_completeness(self, query: str, answer: str) -> float:
        """评估答案完整性"""
        # 1. 长度评估
        answer_length = len(answer.split())
        length_score = min(answer_length / 50.0, 1.0)  # 50词为满分

        # 2. 查询覆盖度
        query_words = set(word.lower() for word in query.split() if len(word) > 3)
        answer_words = set(word.lower() for word in answer.split())

        if query_words:
            coverage = len(query_words.intersection(answer_words)) / len(query_words)
        else:
            coverage = 0.0

        # 3. 结构完整性（是否有完整的句子）
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        structure_score = min(len(sentences) / 3.0, 1.0)  # 3句为满分

        return (length_score + coverage + structure_score) / 3.0

    def _assess_accuracy(self, query: str, answer: str) -> float:
        """评估答案准确性"""
        # 简单的准确性评估：基于答案的结构和内容质量

        # 1. 是否包含具体信息
        has_numbers = bool(re.search(r'\d+', answer))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', answer))

        # 2. 是否避免了模糊表达
        vague_phrases = ['maybe', 'perhaps', 'might be', 'could be', 'not sure']
        vague_count = sum(1 for phrase in vague_phrases if phrase in answer.lower())

        # 3. 综合评分
        specificity_score = (int(has_numbers) + int(has_proper_nouns)) / 2.0
        clarity_score = max(0.0, 1.0 - vague_count / 5.0)

        return (specificity_score + clarity_score) / 2.0


class SelfRag:
    """
    Self-RAG 基线方法

    自适应检索增强生成，基于反思机制

    关键特性:
    - 检索决策模块：智能决定何时检索
    - 相关性评估：评估检索文档质量
    - 生成质量评估：评估答案支持度和有用性
    - 多轮反思机制：端到端的质量控制
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化Self-RAG基线方法

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = get_logger("SelfRag")

        # 初始化组件
        self._initialize_components()

        self.logger.info("Self-RAG基线方法初始化完成")

    def _initialize_components(self):
        """初始化组件"""
        try:
            # 1. 检索决策器
            self.retrieval_decider = RetrievalDecider()

            # 2. 相关性评估器
            embedding_model = None
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                embedding_model = SentenceTransformer(model_name)
                self.logger.info(f"加载嵌入模型: {model_name}")
            except Exception as e:
                self.logger.warning(f"无法加载嵌入模型，将使用简单相似度计算: {e}")

            self.relevance_evaluator = RelevanceEvaluator(embedding_model)

            # 3. 质量评估器
            self.quality_evaluator = QualityEvaluator()

            # 4. 文档检索器（使用现有的检索器）
            self._initialize_retriever()

            self.logger.info("所有组件初始化完成")

        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise

    def _initialize_retriever(self):
        """初始化文档检索器"""
        try:
            # 尝试使用现有的检索器
            from ..retrievers.dense_retriever import DenseRetriever

            retriever_config = self.config.get('retriever', {})
            self.document_retriever = DenseRetriever(retriever_config)
            self.logger.info("使用DenseRetriever作为文档检索器")

        except Exception as e:
            self.logger.warning(f"无法初始化检索器: {e}")
            self.document_retriever = None

    def process_query(self, query: str) -> BaselineResult:
        """
        处理查询 - Self-RAG的主要处理流程

        Args:
            query: 输入查询

        Returns:
            RetrievalResult: 检索结果
        """
        start_time = time.time()

        try:
            # Self-RAG处理流程
            self_rag_result = self._self_rag_process(query)

            # 转换为标准的RetrievalResult格式
            processing_time = time.time() - start_time

            # 收集反思令牌
            reflection_tokens = [
                self_rag_result.retrieve_decision.token,
            ]

            # 添加相关性令牌
            for doc in self_rag_result.relevant_documents:
                if 'relevance_assessment' in doc:
                    reflection_tokens.append(doc['relevance_assessment'].token)

            # 添加质量评估令牌
            reflection_tokens.extend([
                self_rag_result.quality_assessment.support_assessment.support_level,
                self_rag_result.quality_assessment.usefulness_assessment.token
            ])

            return BaselineResult(
                query=query,
                answer=self_rag_result.answer,
                retrieved_documents=self_rag_result.documents,
                processing_time=processing_time,
                overall_confidence=self_rag_result.quality_assessment.overall_quality,
                success=self_rag_result.success,
                error_message=self_rag_result.error_message,
                explanation={
                    "method_name": "Self-RAG",
                    "retrieve_decision": self_rag_result.retrieve_decision.__dict__,
                    "quality_assessment": {
                        "support": self_rag_result.quality_assessment.support_assessment.__dict__,
                        "usefulness": self_rag_result.quality_assessment.usefulness_assessment.__dict__
                    },
                    "reflection_tokens": reflection_tokens,
                    "relevant_documents_count": len(self_rag_result.relevant_documents)
                }
            )

        except Exception as e:
            self.logger.error(f"处理查询失败: {str(e)}")
            processing_time = time.time() - start_time

            return BaselineResult(
                query=query,
                answer="",
                retrieved_documents=[],
                processing_time=processing_time,
                overall_confidence=0.0,
                success=False,
                error_message=str(e)
            )

    def _self_rag_process(self, query: str) -> SelfRAGResult:
        """
        Self-RAG核心处理流程

        1. 检索决策：决定是否需要检索
        2. 文档检索：如果需要，检索相关文档
        3. 相关性评估：评估检索文档的相关性
        4. 答案生成：基于相关文档生成答案
        5. 质量评估：评估答案的支持度和有用性
        """

        # 第1步：检索决策
        retrieve_decision = self.retrieval_decider.decide(query)
        self.logger.info(f"检索决策: {retrieve_decision.token} (置信度: {retrieve_decision.confidence:.3f})")

        documents = []
        relevant_documents = []

        # 第2步：条件检索
        if retrieve_decision.should_retrieve:
            if self.document_retriever:
                try:
                    # 使用现有检索器检索文档
                    retrieval_result = self.document_retriever.retrieve(query)
                    documents = retrieval_result.get('documents', [])
                    self.logger.info(f"检索到 {len(documents)} 个文档")

                    # 第3步：相关性评估
                    relevant_documents = self.relevance_evaluator.filter_relevant_documents(query, documents)
                    self.logger.info(f"过滤后得到 {len(relevant_documents)} 个相关文档")

                except Exception as e:
                    self.logger.error(f"文档检索失败: {e}")
                    # 继续处理，但没有文档支持
            else:
                self.logger.warning("文档检索器未初始化，跳过检索步骤")
        else:
            self.logger.info("决策不检索，直接生成答案")

        # 第4步：答案生成
        answer = self._generate_answer(query, relevant_documents)

        # 第5步：质量评估
        quality_assessment = self.quality_evaluator.evaluate(query, relevant_documents, answer)

        # 收集反思令牌
        reflection_tokens = [
            retrieve_decision.token,
            quality_assessment.support_assessment.support_level,
            quality_assessment.usefulness_assessment.token
        ]

        # 添加相关性令牌
        for doc in relevant_documents:
            if 'relevance_assessment' in doc:
                reflection_tokens.append(doc['relevance_assessment'].token)

        return SelfRAGResult(
            query=query,
            retrieve_decision=retrieve_decision,
            documents=documents,
            relevant_documents=relevant_documents,
            answer=answer,
            quality_assessment=quality_assessment,
            processing_time=0.0,  # 将在外层计算
            reflection_tokens=reflection_tokens,
            success=True
        )

    def _generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        生成答案

        Args:
            query: 查询
            documents: 相关文档列表

        Returns:
            str: 生成的答案
        """
        if not documents:
            # 没有文档支持，生成基础答案
            return self._generate_basic_answer(query)

        # 基于文档生成答案
        return self._generate_document_based_answer(query, documents)

    def _generate_basic_answer(self, query: str) -> str:
        """生成基础答案（无文档支持）"""
        # 简单的基于规则的答案生成
        query_lower = query.lower()

        if any(word in query_lower for word in ['what is', 'define']):
            return f"Based on general knowledge, {query.replace('What is', '').replace('what is', '').strip()} is a concept that requires specific domain knowledge for accurate definition."

        elif any(word in query_lower for word in ['how to', 'how do']):
            return f"To answer '{query}', specific procedural knowledge would be needed. This typically involves multiple steps that depend on the specific context."

        elif any(word in query_lower for word in ['why', 'because']):
            return f"The question '{query}' involves causal relationships that would require domain-specific knowledge to answer accurately."

        else:
            return f"To properly answer '{query}', additional context or domain-specific information would be helpful."

    def _generate_document_based_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """基于文档生成答案"""
        # 提取文档内容
        doc_contents = []
        for doc in documents:
            content = doc.get('content', '') or doc.get('text', '')
            if content:
                doc_contents.append(content)

        if not doc_contents:
            return self._generate_basic_answer(query)

        # 简单的抽取式答案生成
        # 1. 查找最相关的句子
        relevant_sentences = []
        query_words = set(word.lower() for word in query.split() if len(word) > 3)

        for content in doc_contents:
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            for sentence in sentences:
                sentence_words = set(word.lower() for word in sentence.split())
                if query_words and sentence_words:
                    overlap = len(query_words.intersection(sentence_words))
                    if overlap > 0:
                        relevant_sentences.append((sentence, overlap))

        # 2. 按相关性排序
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)

        # 3. 组合答案
        if relevant_sentences:
            # 取前3个最相关的句子
            top_sentences = [s[0] for s in relevant_sentences[:3]]
            answer = ". ".join(top_sentences)

            # 确保答案以句号结尾
            if not answer.endswith('.'):
                answer += '.'

            return answer
        else:
            return self._generate_basic_answer(query)
