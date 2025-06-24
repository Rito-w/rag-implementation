# 🚀 快速开始指南 - Intelligent Adaptive RAG System

## 📋 系统概述

智能自适应RAG系统是一个基于查询感知的自适应检索增强生成系统，具有以下核心特性：

- **🧠 查询智能分析**: 深度理解查询复杂度和类型
- **⚖️ 动态权重分配**: 基于查询特征自适应分配检索器权重
- **🔀 智能融合引擎**: 多路径检索结果的智能合并
- **💡 端到端可解释性**: 完整的决策过程透明化

## 🔧 环境准备

### 1. 系统要求
- Python 3.8+
- 内存: 4GB+
- 磁盘空间: 2GB+

### 2. 安装依赖

```bash
# 克隆或进入项目目录
cd rag_code_learning/rag-implementation

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目 (开发模式)
pip install -e .
```

### 3. 验证安装

```bash
# 运行核心组件测试
python test_core_components.py

# 如果看到 "🎉 所有核心组件测试通过！" 说明安装成功
```

## 🎯 快速体验

### 1. 运行演示脚本

```bash
# 运行完整演示
python demo.py

# 或使用Makefile
make demo
```

### 2. 基础使用示例

```python
from src.core.intelligent_adapter import IntelligentAdaptiveRAG

# 初始化系统
rag_system = IntelligentAdaptiveRAG()

# 处理查询
query = "What are the advantages of hybrid retrieval in RAG systems?"
result = rag_system.process_query(query)

# 查看结果
print(f"答案: {result.answer}")
print(f"置信度: {result.overall_confidence:.1%}")
print(f"查询理解: {result.query_analysis_explanation}")
print(f"权重分配: {result.weight_allocation_explanation}")
```

## 📊 核心功能展示

### 1. 查询智能分析

```python
from src.core.query_analyzer import QueryIntelligenceAnalyzer

analyzer = QueryIntelligenceAnalyzer()

# 分析不同复杂度的查询
queries = [
    "What is machine learning?",  # 简单事实查询
    "How do transformer models work and what are their advantages?",  # 复杂技术查询
    "Compare different RAG approaches and analyze their trade-offs"  # 分析比较查询
]

for query in queries:
    analysis = analyzer.analyze(query)
    print(f"查询: {query}")
    print(f"复杂度: {analysis.complexity_score:.1f}/5.0")
    print(f"类型: {analysis.query_type.value}")
    print(f"置信度: {analysis.confidence:.1%}")
    print("---")
```

### 2. 动态权重分配

```python
from src.core.weight_controller import DynamicWeightController

controller = DynamicWeightController()

# 基于查询分析计算权重
weights = controller.compute_weights(analysis)

print(f"稠密检索权重: {weights.dense_weight:.1%}")
print(f"稀疏检索权重: {weights.sparse_weight:.1%}")
print(f"混合检索权重: {weights.hybrid_weight:.1%}")
print(f"选择策略: {weights.strategy.value}")
```

### 3. 检索器使用

```python
from src.retrievers import DenseRetriever, SparseRetriever, HybridRetriever

# 稠密检索 (语义搜索)
dense_retriever = DenseRetriever()
dense_results = dense_retriever.retrieve("machine learning algorithms", k=5)

# 稀疏检索 (关键词匹配)
sparse_retriever = SparseRetriever()
sparse_results = sparse_retriever.retrieve("machine learning algorithms", k=5)

# 混合检索 (组合方法)
hybrid_retriever = HybridRetriever()
hybrid_results = hybrid_retriever.retrieve("machine learning algorithms", k=5)

print(f"稠密检索结果: {len(dense_results)}个")
print(f"稀疏检索结果: {len(sparse_results)}个")
print(f"混合检索结果: {len(hybrid_results)}个")
```

## 🔬 高级功能

### 1. 自定义配置

```python
# 使用自定义配置
custom_config = {
    "query_analyzer": {
        "complexity_weights": {
            "alpha": 0.4,  # 增加词汇复杂度权重
            "beta": 0.3,   # 增加语法复杂度权重
            "gamma": 0.2,  # 实体复杂度权重
            "delta": 0.1   # 领域复杂度权重
        }
    },
    "weight_controller": {
        "min_weight": 0.05,
        "max_weight": 0.95
    }
}

rag_system = IntelligentAdaptiveRAG(config=custom_config)
```

### 2. 添加自定义文档

```python
# 添加文档到检索器
documents = [
    {
        "id": "custom_doc_1",
        "title": "自定义文档标题",
        "content": "这是自定义文档的内容...",
        "metadata": {"source": "custom", "category": "technical"}
    }
]

# 添加到各个检索器
dense_retriever.add_documents(documents)
sparse_retriever.add_documents(documents)
hybrid_retriever.add_documents(documents)
```

### 3. 获取详细解释

```python
# 获取完整的决策解释
result = rag_system.process_query(query, return_detailed=True)

# 查看详细解释
explanations = result.get_comprehensive_explanation()
print("查询理解:", explanations['query_understanding'])
print("检索策略:", explanations['retrieval_strategy'])
print("答案生成:", explanations['answer_generation'])
```

## 📈 性能监控

### 1. 系统统计

```python
# 获取系统性能统计
stats = rag_system.get_system_stats()
print(f"处理查询数: {stats['queries_processed']}")
print(f"平均处理时间: {stats['average_processing_time']:.3f}秒")
```

### 2. 检索器统计

```python
# 获取检索器统计
dense_stats = dense_retriever.get_stats()
sparse_stats = sparse_retriever.get_stats()

print("稠密检索器统计:", dense_stats)
print("稀疏检索器统计:", sparse_stats)
```

## 🧪 测试和验证

### 1. 运行所有测试

```bash
# 运行单元测试
make test

# 运行核心组件测试
python test_core_components.py

# 运行覆盖率测试
make test-cov
```

### 2. 性能基准测试

```python
import time

# 性能测试
queries = ["What is AI?"] * 100
start_time = time.time()

for query in queries:
    result = rag_system.process_query(query)

end_time = time.time()
avg_time = (end_time - start_time) / len(queries)
print(f"平均查询处理时间: {avg_time:.3f}秒")
```

## 🔧 开发工具

### 1. 代码格式化

```bash
# 格式化代码
make format

# 检查代码质量
make lint
```

### 2. 开发循环

```bash
# 完整开发循环 (格式化 + 检查 + 测试)
make dev-cycle
```

## 🚨 常见问题

### 1. 导入错误
```bash
# 如果遇到导入错误，确保已安装依赖
pip install -r requirements.txt
pip install -e .
```

### 2. 内存不足
```bash
# 如果内存不足，可以减少批处理大小
# 在配置中设置较小的batch_size
```

### 3. 性能优化
```python
# 启用缓存以提高性能
config = {
    "query_analyzer": {"use_cache": True},
    "retrieval": {"cache_enabled": True}
}
```

## 📚 下一步

1. **阅读详细文档**: 查看 `docs/` 目录下的详细文档
2. **查看示例**: 运行 `docs/examples/` 中的示例代码
3. **自定义开发**: 基于需求修改和扩展系统
4. **性能调优**: 根据具体场景调整配置参数
5. **实验验证**: 在真实数据上验证系统性能

## 🤝 贡献和反馈

- **问题报告**: 在GitHub Issues中报告问题
- **功能建议**: 提交功能请求和改进建议
- **代码贡献**: 欢迎提交Pull Request

---

> 🎉 **恭喜！** 您已经成功启动了智能自适应RAG系统。现在可以开始探索这个强大的查询感知检索系统了！
