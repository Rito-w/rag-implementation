"""
Configuration management for the Intelligent Adaptive RAG system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging


class Config:
    """
    Configuration manager for the Intelligent Adaptive RAG system.
    
    Supports YAML and JSON configuration files with environment variable
    substitution and nested key access.
    """
    
    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            **kwargs: Additional configuration parameters to override
        """
        self.config_data = {}
        
        # Load default configuration
        self._load_defaults()
        
        # Load from file if provided
        if config_path:
            self._load_from_file(config_path)
        
        # Override with kwargs
        if kwargs:
            self._update_config(kwargs)
    
    def _load_defaults(self):
        """Load default configuration values."""
        self.config_data = {
            # System settings
            "system": {
                "version": "0.1.0",
                "debug": False,
                "max_workers": 4
            },
            
            # Logging configuration
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None
            },
            
            # Query analyzer settings
            "query_analyzer": {
                "complexity_weights": {
                    "alpha": 0.3,  # lexical complexity weight
                    "beta": 0.25,  # syntactic complexity weight
                    "gamma": 0.25, # entity complexity weight
                    "delta": 0.2   # domain complexity weight
                },
                "classification_threshold": 0.7,
                "feature_dim": 768,
                "use_cache": True
            },
            
            # Weight controller settings
            "weight_controller": {
                "learning_rate": 0.001,
                "regularization": 0.01,
                "min_weight": 0.01,
                "max_weight": 0.98,
                "confidence_threshold": 0.8
            },
            
            # Fusion engine settings
            "fusion_engine": {
                "fusion_method": "intelligent_weighted",
                "diversity_lambda": 0.1,
                "quality_mu": 0.2,
                "max_results": 20,
                "deduplication_threshold": 0.9
            },
            
            # Explainer settings
            "explainer": {
                "explanation_level": "detailed",
                "include_confidence": True,
                "max_explanation_length": 500,
                "language": "zh"
            },
            
            # Retriever settings
            "retrievers": {
                "dense": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "max_docs": 100,
                    "similarity_threshold": 0.7
                },
                "sparse": {
                    "algorithm": "bm25",
                    "k1": 1.2,
                    "b": 0.75,
                    "max_docs": 100
                },
                "hybrid": {
                    "dense_weight": 0.6,
                    "sparse_weight": 0.4,
                    "max_docs": 100
                }
            },
            
            # Retrieval settings
            "retrieval": {
                "max_docs": 20,
                "min_score": 0.1,
                "timeout": 30.0
            },
            
            # Generation settings
            "generation": {
                "model_name": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.7,
                "timeout": 30.0
            },
            
            # Evaluation settings
            "evaluation": {
                "metrics": ["mrr", "ndcg", "recall", "precision"],
                "k_values": [1, 5, 10, 20],
                "save_results": True
            }
        }
    
    def _load_from_file(self, config_path: Union[str, Path]):
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Merge with existing configuration
            self._deep_update(self.config_data, file_config)
            
        except Exception as e:
            raise ValueError(f"Error loading configuration file {config_path}: {str(e)}")
    
    def _update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self._deep_update(self.config_data, updates)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'logging.level')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config_data
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def has(self, key: str) -> bool:
        """
        Check if configuration key exists.
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            True if key exists, False otherwise
        """
        return self.get(key) is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return self.config_data.copy()
    
    def save(self, file_path: Union[str, Path], format: str = 'yaml'):
        """
        Save configuration to file.
        
        Args:
            file_path: Output file path
            format: File format ('yaml' or 'json')
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
                elif format.lower() == 'json':
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            raise ValueError(f"Error saving configuration to {file_path}: {str(e)}")
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate required keys
            required_keys = [
                'query_analyzer.complexity_weights.alpha',
                'weight_controller.learning_rate',
                'retrievers.dense.model_name'
            ]
            
            for key in required_keys:
                if not self.has(key):
                    logging.warning(f"Missing required configuration key: {key}")
                    return False
            
            # Validate value ranges
            alpha = self.get('query_analyzer.complexity_weights.alpha')
            if not 0 <= alpha <= 1:
                logging.warning(f"Invalid alpha value: {alpha}")
                return False
            
            # Add more validation as needed
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation error: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config({len(self.config_data)} sections)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Config(sections={list(self.config_data.keys())})"
