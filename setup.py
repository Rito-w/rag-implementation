"""
Setup script for Intelligent Adaptive RAG System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="intelligent-adaptive-rag",
    version="0.1.0",
    author="Intelligent RAG Research Team",
    author_email="research@example.com",
    description="An intelligent adaptive RAG system with query-aware retrieval and comprehensive explainability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/intelligent-adaptive-rag",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    python_requires=">=3.8",
    install_requires=requirements,
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
            "torch>=2.0.0+cu118",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "faiss-gpu>=1.7.4",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "intelligent-rag=src.cli:main",
        ],
    },
    
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    
    keywords=[
        "rag", "retrieval", "augmented", "generation", 
        "nlp", "ai", "machine learning", "information retrieval",
        "question answering", "explainable ai"
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/example/intelligent-adaptive-rag/issues",
        "Source": "https://github.com/example/intelligent-adaptive-rag",
        "Documentation": "https://intelligent-adaptive-rag.readthedocs.io/",
    },
)
