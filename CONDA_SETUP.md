# 🐍 Conda环境设置指南 - Intelligent Adaptive RAG System

## 📋 前置要求

- **Anaconda** 或 **Miniconda** 已安装
- **Python 3.8+** 支持
- **8GB+ RAM** 推荐
- **网络连接** (用于下载依赖和模型)

## 🚀 快速安装步骤

### 1. 进入项目目录

```bash
cd rag_code_learning/rag-implementation
```

### 2. 创建conda环境

```bash
# 使用environment.yml创建环境
conda env create -f environment.yml

# 或者手动创建环境
conda create -n intelligent-rag python=3.10 -y
```

### 3. 激活环境

```bash
conda activate intelligent-rag
```

### 4. 验证环境

```bash
# 检查Python版本
python --version

# 检查conda环境
conda info --envs
```

## 📦 依赖安装

### 方法一：使用environment.yml (推荐)

```bash
# 如果已经创建了环境
conda activate intelligent-rag

# 如果还没创建环境
conda env create -f environment.yml
conda activate intelligent-rag
```

### 方法二：手动安装

```bash
# 激活环境
conda activate intelligent-rag

# 安装核心依赖
conda install pytorch numpy scipy scikit-learn pandas -c pytorch -c conda-forge

# 安装其他依赖
pip install -r requirements.txt
```

### 5. 安装项目

```bash
# 以开发模式安装项目
pip install -e .
```

## 🔧 额外配置

### 1. 下载spaCy语言模型

```bash
# 下载英文模型
python -m spacy download en_core_web_sm

# 如果需要中文模型
python -m spacy download zh_core_web_sm
```

### 2. 下载NLTK数据

```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"
```

### 3. 验证安装

```bash
# 运行核心组件测试
python test_core_components.py
```

## 🎯 运行项目

### 1. 快速测试

```bash
# 运行核心组件测试
python test_core_components.py

# 预期输出：🎉 所有核心组件测试通过！
```

### 2. 运行演示

```bash
# 运行完整演示
python demo.py

# 使用Makefile (如果支持)
make demo
```

### 3. 交互式使用

```python
# 启动Python
python

# 导入和使用系统
from src.core.intelligent_adapter import IntelligentAdaptiveRAG

# 初始化系统
rag_system = IntelligentAdaptiveRAG()

# 处理查询
result = rag_system.process_query("What is machine learning?")
print(f"答案: {result.answer}")
print(f"置信度: {result.overall_confidence:.1%}")
```

## 🔍 故障排除

### 常见问题1: 依赖冲突

```bash
# 清理环境重新安装
conda deactivate
conda env remove -n intelligent-rag
conda env create -f environment.yml
conda activate intelligent-rag
```

### 常见问题2: PyTorch安装问题

```bash
# 手动安装PyTorch (CPU版本)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 或GPU版本 (如果有CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 常见问题3: FAISS安装问题

```bash
# 如果faiss-cpu安装失败，尝试conda安装
conda install faiss-cpu -c conda-forge

# 或者使用CPU版本
pip install faiss-cpu --no-cache-dir
```

### 常见问题4: spaCy模型下载失败

```bash
# 手动下载模型
python -m spacy download en_core_web_sm --user

# 或者使用直接链接
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl
```

### 常见问题5: 内存不足

```bash
# 如果内存不足，可以设置较小的批处理大小
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 📊 验证安装成功

### 1. 运行完整测试套件

```bash
# 基础功能测试
python -m pytest tests/test_basic_functionality.py -v

# 核心组件测试
python test_core_components.py

# 如果所有测试通过，说明安装成功
```

### 2. 检查系统状态

```python
python -c "
from src.core.intelligent_adapter import IntelligentAdaptiveRAG
rag = IntelligentAdaptiveRAG()
stats = rag.get_system_stats()
print('系统状态:', stats)
print('✅ 系统初始化成功!')
"
```

## 🔧 开发环境设置

### 1. 安装开发工具

```bash
# 代码格式化和检查
pip install black isort flake8 mypy

# 或者使用conda
conda install black isort flake8 -c conda-forge
```

### 2. 设置pre-commit hooks (可选)

```bash
pip install pre-commit
pre-commit install
```

### 3. 配置IDE

```bash
# 如果使用VSCode，安装Python扩展
# 设置Python解释器路径为conda环境路径
which python  # 获取环境路径
```

## 🚀 性能优化

### 1. GPU支持 (可选)

```bash
# 如果有GPU，安装GPU版本的依赖
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install faiss-gpu
```

### 2. 内存优化

```bash
# 设置环境变量优化内存使用
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
```

## 📚 下一步

1. **阅读快速开始指南**: `QUICK_START.md`
2. **查看项目文档**: `README.md`
3. **运行示例**: `python demo.py`
4. **开始开发**: 修改和扩展系统

---

> 🎉 **安装完成！** 现在您可以开始使用智能自适应RAG系统了！
