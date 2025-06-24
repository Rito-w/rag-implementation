"""
Logging utilities for the Intelligent Adaptive RAG system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logger(
    name: str,
    level: Union[str, int] = logging.INFO,
    format_string: Optional[str] = None,
    file_path: Optional[Union[str, Path]] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup a logger with specified configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
        file_path: Optional file path for file logging
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger by name."""
    return logging.getLogger(name)


class QueryLogger:
    """
    Specialized logger for query processing with structured logging.
    """
    
    def __init__(self, base_logger: logging.Logger):
        self.logger = base_logger
    
    def log_query_start(self, query: str, query_id: str = None):
        """Log query processing start."""
        self.logger.info(
            f"Query processing started",
            extra={
                'query_id': query_id,
                'query': query[:100] + "..." if len(query) > 100 else query,
                'timestamp': datetime.now().isoformat(),
                'stage': 'start'
            }
        )
    
    def log_analysis_result(self, query_id: str, analysis_result: dict):
        """Log query analysis result."""
        self.logger.debug(
            f"Query analysis completed",
            extra={
                'query_id': query_id,
                'complexity_score': analysis_result.get('complexity_score'),
                'query_type': analysis_result.get('query_type'),
                'confidence': analysis_result.get('confidence'),
                'stage': 'analysis'
            }
        )
    
    def log_weight_allocation(self, query_id: str, weights: dict):
        """Log weight allocation result."""
        self.logger.debug(
            f"Weight allocation completed",
            extra={
                'query_id': query_id,
                'dense_weight': weights.get('dense_weight'),
                'sparse_weight': weights.get('sparse_weight'),
                'hybrid_weight': weights.get('hybrid_weight'),
                'strategy': weights.get('strategy'),
                'stage': 'weight_allocation'
            }
        )
    
    def log_retrieval_result(self, query_id: str, retrieval_stats: dict):
        """Log retrieval result."""
        self.logger.debug(
            f"Retrieval completed",
            extra={
                'query_id': query_id,
                'documents_found': retrieval_stats.get('total_documents'),
                'avg_score': retrieval_stats.get('avg_score'),
                'processing_time': retrieval_stats.get('processing_time'),
                'stage': 'retrieval'
            }
        )
    
    def log_query_complete(self, query_id: str, total_time: float, success: bool = True):
        """Log query processing completion."""
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"Query processing {'completed' if success else 'failed'}",
            extra={
                'query_id': query_id,
                'total_processing_time': total_time,
                'success': success,
                'stage': 'complete'
            }
        )


class PerformanceLogger:
    """
    Logger for performance metrics and system monitoring.
    """
    
    def __init__(self, base_logger: logging.Logger):
        self.logger = base_logger
    
    def log_component_performance(self, component: str, operation: str, duration: float):
        """Log component performance metrics."""
        self.logger.debug(
            f"Component performance: {component}.{operation}",
            extra={
                'component': component,
                'operation': operation,
                'duration': duration,
                'metric_type': 'performance'
            }
        )
    
    def log_memory_usage(self, component: str, memory_mb: float):
        """Log memory usage."""
        self.logger.debug(
            f"Memory usage: {component}",
            extra={
                'component': component,
                'memory_mb': memory_mb,
                'metric_type': 'memory'
            }
        )
    
    def log_system_stats(self, stats: dict):
        """Log system statistics."""
        self.logger.info(
            "System statistics",
            extra={
                'queries_processed': stats.get('queries_processed'),
                'avg_processing_time': stats.get('average_processing_time'),
                'total_processing_time': stats.get('total_processing_time'),
                'metric_type': 'system_stats'
            }
        )


def create_experiment_logger(experiment_name: str, log_dir: Path) -> logging.Logger:
    """
    Create a logger specifically for experiments.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        
    Returns:
        Configured experiment logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    logger = setup_logger(
        name=f"experiment_{experiment_name}",
        level=logging.DEBUG,
        file_path=log_file,
        console_output=True
    )
    
    logger.info(f"Experiment logger created: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger
