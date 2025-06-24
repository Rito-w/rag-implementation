# Makefile for Intelligent Adaptive RAG System

.PHONY: help install install-dev test test-coverage lint format clean demo docs

# Default target
help:
	@echo "🎯 Intelligent Adaptive RAG System - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      Install package and dependencies"
	@echo "  install-dev  Install package with development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  demo         Run demo script"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean        Clean build artifacts and cache"

# Installation
install:
	@echo "📦 Installing Intelligent Adaptive RAG System..."
	pip install -e .

install-dev:
	@echo "🔧 Installing development dependencies..."
	pip install -e ".[dev]"
	pip install -r requirements.txt

# Testing
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

test-cov:
	@echo "📊 Running tests with coverage..."
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	@echo "🔍 Running linting checks..."
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	@echo "✨ Formatting code..."
	black src/ tests/ demo.py --line-length=100
	isort src/ tests/ demo.py --profile black

# Demo
demo:
	@echo "🎬 Running demo..."
	python demo.py

# Documentation
docs:
	@echo "📚 Building documentation..."
	cd docs && make html

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Development workflow
dev-setup: install-dev
	@echo "🚀 Development environment setup complete!"
	@echo "Run 'make demo' to test the system"

# Quick development cycle
dev-cycle: format lint test
	@echo "✅ Development cycle complete!"

# Release preparation
release-check: clean format lint test-cov
	@echo "🎉 Release checks passed!"
