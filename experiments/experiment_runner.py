#!/usr/bin/env python3
"""
实验运行器

自动化运行RAG系统实验，包括：
1. 基准系统对比实验
2. 消融实验
3. 性能分析实验
4. 可解释性评估实验
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.intelligent_adapter import IntelligentAdaptiveRAG
from src.utils.logging import setup_logger


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化实验运行器
        
        Args:
            config_path: 实验配置文件路径
        """
        self.logger = setup_logger("ExperimentRunner")
        
        # 创建实验目录
        self.experiment_dir = Path("experiments/results")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 实验ID
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # 结果存储
        self.results = {
            "experiment_id": self.experiment_id,
            "start_time": datetime.now().isoformat(),
            "config": self.config,
            "results": {}
        }
        
        self.logger.info(f"实验ID: {self.experiment_id}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载实验配置"""
        default_config = {
            "datasets": ["samples"],  # 默认使用样本数据集
            "systems": ["intelligent_rag"],  # 要测试的系统
            "metrics": ["retrieval_quality", "answer_quality", "efficiency"],
            "sample_size": 50,  # 每个数据集的样本数量
            "repetitions": 1,   # 重复次数
            "timeout": 30       # 单个查询超时时间(秒)
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"已加载配置文件: {config_path}")
            except Exception as e:
                self.logger.warning(f"配置文件加载失败，使用默认配置: {str(e)}")
        
        return default_config
    
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        加载数据集

        Args:
            dataset_name: 数据集名称

        Returns:
            List[Dict]: 查询列表
        """
        try:
            # 通用数据集加载逻辑
            data_path = Path(f"data/{dataset_name}/queries.json")

            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
                self.logger.info(f"已加载{dataset_name}数据集: {len(queries)}个查询")
                return queries

            # 如果是samples数据集，提供默认查询
            elif dataset_name == "samples":
                default_queries = [
                    {"id": "1", "query": "What is machine learning?", "type": "factual"},
                    {"id": "2", "query": "How do neural networks work?", "type": "technical"},
                    {"id": "3", "query": "Compare supervised and unsupervised learning", "type": "comparison"},
                    {"id": "4", "query": "What are the applications of deep learning?", "type": "application"},
                    {"id": "5", "query": "Explain the transformer architecture", "type": "detailed"}
                ]
                self.logger.info("使用默认查询数据")
                return default_queries
            else:
                self.logger.warning(f"数据集文件不存在: {data_path}")
                return []

        except Exception as e:
            self.logger.error(f"加载数据集失败 {dataset_name}: {str(e)}")
            return []
    
    def run_single_query(self, system: Any, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单个查询
        
        Args:
            system: RAG系统实例
            query: 查询信息
            
        Returns:
            Dict: 查询结果
        """
        try:
            start_time = time.time()
            
            # 执行查询
            result = system.process_query(query["query"])
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 构建结果
            query_result = {
                "query_id": query["id"],
                "query_text": query["query"],
                "query_type": query.get("type", "unknown"),
                "answer": result.answer,
                "confidence": result.overall_confidence,
                "processing_time": processing_time,
                "retrieval_results": len(result.retrieved_documents),
                "query_analysis": {
                    "complexity": result.query_analysis.complexity_score,
                    "type": result.query_analysis.query_type.value,
                    "confidence": result.query_analysis.confidence
                },
                "weight_allocation": {
                    "dense": result.weight_allocation.dense_weight,
                    "sparse": result.weight_allocation.sparse_weight,
                    "hybrid": result.weight_allocation.hybrid_weight,
                    "strategy": result.weight_allocation.strategy.value
                }
            }
            
            return query_result
            
        except Exception as e:
            self.logger.error(f"查询执行失败 {query['id']}: {str(e)}")
            return {
                "query_id": query["id"],
                "query_text": query["query"],
                "error": str(e),
                "processing_time": 0,
                "confidence": 0
            }
    
    def run_system_experiment(self, system_name: str, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        运行系统实验
        
        Args:
            system_name: 系统名称
            queries: 查询列表
            
        Returns:
            Dict: 实验结果
        """
        self.logger.info(f"开始运行系统实验: {system_name}")
        
        try:
            # 初始化系统
            if system_name == "intelligent_rag":
                system = IntelligentAdaptiveRAG()
            else:
                # TODO: 添加其他基准系统
                self.logger.warning(f"系统 {system_name} 暂未实现")
                return {"error": f"System {system_name} not implemented"}
            
            # 运行查询
            query_results = []
            total_queries = min(len(queries), self.config["sample_size"])
            
            self.logger.info(f"运行 {total_queries} 个查询...")
            
            for i, query in enumerate(queries[:total_queries]):
                self.logger.info(f"处理查询 {i+1}/{total_queries}: {query['id']}")
                
                # 运行多次重复实验
                repetition_results = []
                for rep in range(self.config["repetitions"]):
                    result = self.run_single_query(system, query)
                    repetition_results.append(result)
                
                # 计算平均结果
                if repetition_results:
                    avg_result = self._average_results(repetition_results)
                    query_results.append(avg_result)
            
            # 计算系统级统计
            system_stats = self._calculate_system_stats(query_results)
            
            return {
                "system_name": system_name,
                "total_queries": total_queries,
                "successful_queries": len([r for r in query_results if "error" not in r]),
                "query_results": query_results,
                "system_stats": system_stats
            }
            
        except Exception as e:
            self.logger.error(f"系统实验失败 {system_name}: {str(e)}")
            return {"error": str(e)}
    
    def _average_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算多次重复实验的平均结果"""
        if not results:
            return {}
        
        # 取第一个结果作为基础
        avg_result = results[0].copy()
        
        if len(results) > 1:
            # 计算数值字段的平均值
            numeric_fields = ["confidence", "processing_time"]
            for field in numeric_fields:
                if field in avg_result:
                    values = [r.get(field, 0) for r in results if field in r]
                    avg_result[field] = sum(values) / len(values) if values else 0
        
        return avg_result
    
    def _calculate_system_stats(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算系统级统计信息"""
        if not query_results:
            return {}
        
        # 过滤成功的查询
        successful_results = [r for r in query_results if "error" not in r]
        
        if not successful_results:
            return {"error": "No successful queries"}
        
        # 计算统计信息
        confidences = [r.get("confidence", 0) for r in successful_results]
        processing_times = [r.get("processing_time", 0) for r in successful_results]
        complexities = [r.get("query_analysis", {}).get("complexity", 0) for r in successful_results]
        
        stats = {
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "avg_complexity": sum(complexities) / len(complexities),
            "total_processing_time": sum(processing_times),
            "throughput": len(successful_results) / sum(processing_times) if sum(processing_times) > 0 else 0,
            "success_rate": len(successful_results) / len(query_results)
        }
        
        return stats
    
    def run_experiment(self) -> Dict[str, Any]:
        """运行完整实验"""
        self.logger.info("开始运行实验...")
        
        try:
            # 遍历数据集
            for dataset_name in self.config["datasets"]:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"数据集: {dataset_name}")
                self.logger.info(f"{'='*50}")
                
                # 加载数据集
                queries = self.load_dataset(dataset_name)
                if not queries:
                    self.logger.warning(f"数据集 {dataset_name} 为空，跳过")
                    continue
                
                dataset_results = {}
                
                # 遍历系统
                for system_name in self.config["systems"]:
                    self.logger.info(f"\n运行系统: {system_name}")
                    
                    system_result = self.run_system_experiment(system_name, queries)
                    dataset_results[system_name] = system_result
                
                self.results["results"][dataset_name] = dataset_results
            
            # 保存结果
            self.results["end_time"] = datetime.now().isoformat()
            self.results["duration"] = (
                datetime.fromisoformat(self.results["end_time"]) - 
                datetime.fromisoformat(self.results["start_time"])
            ).total_seconds()
            
            self._save_results()
            
            self.logger.info("实验完成！")
            return self.results
            
        except Exception as e:
            self.logger.error(f"实验运行失败: {str(e)}")
            self.results["error"] = str(e)
            return self.results
    
    def _save_results(self):
        """保存实验结果"""
        try:
            # 保存详细结果
            result_file = self.experiment_dir / f"{self.experiment_id}_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            # 保存简化报告
            report = self._generate_report()
            report_file = self.experiment_dir / f"{self.experiment_id}_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"结果已保存: {result_file}")
            self.logger.info(f"报告已保存: {report_file}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
    
    def _generate_report(self) -> str:
        """生成实验报告"""
        report = f"""# 实验报告

## 实验信息
- **实验ID**: {self.experiment_id}
- **开始时间**: {self.results['start_time']}
- **结束时间**: {self.results.get('end_time', 'N/A')}
- **持续时间**: {self.results.get('duration', 0):.2f}秒

## 实验配置
- **数据集**: {', '.join(self.config['datasets'])}
- **系统**: {', '.join(self.config['systems'])}
- **样本数量**: {self.config['sample_size']}
- **重复次数**: {self.config['repetitions']}

## 实验结果

"""
        
        # 添加结果详情
        for dataset_name, dataset_results in self.results["results"].items():
            report += f"### 数据集: {dataset_name}\n\n"
            
            for system_name, system_result in dataset_results.items():
                if "error" in system_result:
                    report += f"- **{system_name}**: ❌ 错误 - {system_result['error']}\n"
                    continue
                
                stats = system_result.get("system_stats", {})
                report += f"""- **{system_name}**:
  - 成功查询: {system_result.get('successful_queries', 0)}/{system_result.get('total_queries', 0)}
  - 平均置信度: {stats.get('avg_confidence', 0):.2%}
  - 平均处理时间: {stats.get('avg_processing_time', 0):.3f}秒
  - 吞吐量: {stats.get('throughput', 0):.2f} queries/sec
  - 成功率: {stats.get('success_rate', 0):.2%}

"""
        
        return report


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行RAG系统实验")
    parser.add_argument("--config", help="实验配置文件路径")
    parser.add_argument("--sample-only", action="store_true", help="只运行样本数据集")
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = ExperimentRunner(args.config)
    
    # 如果只运行样本，修改配置
    if args.sample_only:
        runner.config["datasets"] = ["samples"]
        runner.config["sample_size"] = 5
    
    # 运行实验
    results = runner.run_experiment()
    
    # 输出简要结果
    if "error" in results:
        print(f"❌ 实验失败: {results['error']}")
    else:
        print(f"✅ 实验完成: {results['experiment_id']}")
        print(f"📊 结果保存在: experiments/results/")


if __name__ == "__main__":
    main()
