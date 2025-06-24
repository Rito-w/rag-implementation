"""
评估指标工具模块

提供各种评估指标的计算功能，包括：
- 文本复杂度指标
- 检索质量指标
- 系统性能指标
"""

import re
import math
from typing import List, Dict, Any, Optional
import numpy as np


def calculate_text_complexity(text: str) -> Dict[str, float]:
    """
    计算文本复杂度指标
    
    Args:
        text: 输入文本
        
    Returns:
        Dict[str, float]: 复杂度指标字典
    """
    if not text or not text.strip():
        return {
            'lexical_complexity': 0.0,
            'syntactic_complexity': 0.0,
            'entity_complexity': 0.0,
            'domain_complexity': 0.0
        }
    
    # 词汇复杂度 (基于词汇多样性和平均词长)
    words = text.lower().split()
    unique_words = set(words)
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    lexical_diversity = len(unique_words) / len(words) if words else 0
    lexical_complexity = (avg_word_length / 10.0 + lexical_diversity) / 2.0
    
    # 语法复杂度 (基于句子长度和标点符号)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    punctuation_density = len(re.findall(r'[,;:()"\'-]', text)) / len(text) if text else 0
    syntactic_complexity = min((avg_sentence_length / 20.0 + punctuation_density * 10) / 2.0, 1.0)
    
    # 实体复杂度 (基于大写词和数字)
    capitalized_words = len(re.findall(r'\b[A-Z][a-z]+', text))
    numbers = len(re.findall(r'\b\d+', text))
    entity_density = (capitalized_words + numbers) / len(words) if words else 0
    entity_complexity = min(entity_density * 2.0, 1.0)
    
    # 领域复杂度 (基于技术词汇)
    technical_patterns = [
        r'\b\w+ing\b',  # 动名词
        r'\b\w+tion\b',  # 名词后缀
        r'\b\w+ment\b',  # 名词后缀
        r'\b\w+ness\b',  # 名词后缀
    ]
    technical_words = 0
    for pattern in technical_patterns:
        technical_words += len(re.findall(pattern, text, re.IGNORECASE))
    
    domain_complexity = min(technical_words / len(words) if words else 0, 1.0)
    
    return {
        'lexical_complexity': min(lexical_complexity, 1.0),
        'syntactic_complexity': syntactic_complexity,
        'entity_complexity': entity_complexity,
        'domain_complexity': domain_complexity
    }


def calculate_retrieval_metrics(retrieved_docs: List[Dict], 
                              relevant_docs: List[str] = None) -> Dict[str, float]:
    """
    计算检索质量指标
    
    Args:
        retrieved_docs: 检索到的文档列表
        relevant_docs: 相关文档ID列表 (可选)
        
    Returns:
        Dict[str, float]: 检索指标字典
    """
    if not retrieved_docs:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mrr': 0.0,
            'ndcg': 0.0
        }
    
    # 如果没有提供相关文档，使用模拟指标
    if not relevant_docs:
        # 基于分数的模拟指标
        scores = [doc.get('final_score', 0.0) for doc in retrieved_docs]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            'precision': min(avg_score, 1.0),
            'recall': min(avg_score * 0.8, 1.0),
            'f1_score': min(avg_score * 0.9, 1.0),
            'mrr': min(avg_score, 1.0),
            'ndcg': min(avg_score, 1.0)
        }
    
    # 计算实际指标
    retrieved_ids = [doc.get('document_id', '') for doc in retrieved_docs]
    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved_ids)
    
    # Precision
    true_positives = len(relevant_set & retrieved_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    
    # Recall
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    
    # F1 Score
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            mrr = 1.0 / i
            break
    
    # NDCG (简化版本)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 1)
    
    # 理想DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_set), len(retrieved_ids))))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mrr': mrr,
        'ndcg': ndcg
    }


def calculate_diversity_score(documents: List[Dict]) -> float:
    """
    计算文档多样性分数
    
    Args:
        documents: 文档列表
        
    Returns:
        float: 多样性分数 (0-1)
    """
    if len(documents) <= 1:
        return 0.0
    
    # 基于内容的简单多样性计算
    contents = [doc.get('content', '') for doc in documents]
    
    # 计算词汇重叠
    all_words = []
    doc_words = []
    
    for content in contents:
        words = set(content.lower().split())
        doc_words.append(words)
        all_words.extend(words)
    
    if not all_words:
        return 0.0
    
    # 计算平均Jaccard距离
    total_distance = 0.0
    comparisons = 0
    
    for i in range(len(doc_words)):
        for j in range(i + 1, len(doc_words)):
            intersection = len(doc_words[i] & doc_words[j])
            union = len(doc_words[i] | doc_words[j])
            
            if union > 0:
                jaccard_similarity = intersection / union
                jaccard_distance = 1.0 - jaccard_similarity
                total_distance += jaccard_distance
                comparisons += 1
    
    diversity_score = total_distance / comparisons if comparisons > 0 else 0.0
    return min(diversity_score, 1.0)


def calculate_coverage_score(query: str, documents: List[Dict]) -> float:
    """
    计算查询覆盖度分数
    
    Args:
        query: 查询字符串
        documents: 文档列表
        
    Returns:
        float: 覆盖度分数 (0-1)
    """
    if not query or not documents:
        return 0.0
    
    query_words = set(query.lower().split())
    if not query_words:
        return 0.0
    
    # 计算文档中覆盖的查询词
    covered_words = set()
    
    for doc in documents:
        content = doc.get('content', '').lower()
        doc_words = set(content.split())
        covered_words.update(query_words & doc_words)
    
    coverage_score = len(covered_words) / len(query_words)
    return min(coverage_score, 1.0)


def calculate_system_performance(processing_times: List[float],
                               memory_usage: List[float] = None) -> Dict[str, float]:
    """
    计算系统性能指标
    
    Args:
        processing_times: 处理时间列表 (秒)
        memory_usage: 内存使用列表 (MB, 可选)
        
    Returns:
        Dict[str, float]: 性能指标字典
    """
    if not processing_times:
        return {
            'avg_processing_time': 0.0,
            'max_processing_time': 0.0,
            'min_processing_time': 0.0,
            'throughput': 0.0,
            'avg_memory_usage': 0.0
        }
    
    avg_time = sum(processing_times) / len(processing_times)
    max_time = max(processing_times)
    min_time = min(processing_times)
    throughput = 1.0 / avg_time if avg_time > 0 else 0.0
    
    avg_memory = 0.0
    if memory_usage:
        avg_memory = sum(memory_usage) / len(memory_usage)
    
    return {
        'avg_processing_time': avg_time,
        'max_processing_time': max_time,
        'min_processing_time': min_time,
        'throughput': throughput,
        'avg_memory_usage': avg_memory
    }


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    归一化分数到指定范围
    
    Args:
        score: 原始分数
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        float: 归一化后的分数
    """
    if max_val <= min_val:
        return min_val
    
    normalized = (score - min_val) / (max_val - min_val)
    return max(min_val, min(max_val, normalized))


def weighted_average(scores: List[float], weights: List[float]) -> float:
    """
    计算加权平均

    Args:
        scores: 分数列表
        weights: 权重列表

    Returns:
        float: 加权平均分数
    """
    if not scores or not weights or len(scores) != len(weights):
        return 0.0

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    return weighted_sum / total_weight


def calculate_metrics(text: str, complexity_weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    计算文本的各种指标

    Args:
        text: 输入文本
        complexity_weights: 复杂度权重配置

    Returns:
        Dict[str, float]: 指标字典
    """
    if complexity_weights is None:
        complexity_weights = {
            'alpha': 0.3,
            'beta': 0.3,
            'gamma': 0.2,
            'delta': 0.2
        }

    # 计算文本复杂度
    complexity_metrics = calculate_text_complexity(text)

    # 计算总体复杂度分数
    overall_complexity = (
        complexity_weights['alpha'] * complexity_metrics['lexical_complexity'] +
        complexity_weights['beta'] * complexity_metrics['syntactic_complexity'] +
        complexity_weights['gamma'] * complexity_metrics['entity_complexity'] +
        complexity_weights['delta'] * complexity_metrics['domain_complexity']
    )

    # 添加总体复杂度到结果中
    result = complexity_metrics.copy()
    result['overall_complexity'] = overall_complexity

    return result
