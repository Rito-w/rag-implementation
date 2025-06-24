#!/usr/bin/env python3
"""
大规模实验脚本

运行智能自适应RAG系统的大规模实验，包括：
1. 更大样本量的实验 (100-500查询)
2. 性能基准测试
3. 详细的性能分析
4. 与理论预期的对比
"""

import os
import sys
import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """主函数"""
    print("🚀 智能自适应RAG系统 - 大规模实验")
    print("=" * 70)
    
    # 检查环境
    print("🔍 检查环境和依赖...")
    try:
        from src.core.intelligent_adapter import IntelligentAdaptiveRAG
        from experiments.experiment_runner import ExperimentRunner
        
        # 检查新安装的依赖
        try:
            import sentence_transformers
            import faiss
            print("✅ sentence-transformers 和 FAISS 已安装")
        except ImportError:
            print("⚠️  sentence-transformers 或 FAISS 未安装，将使用简化版本")
        
        # 检查数据集
        data_dir = Path("data")
        available_datasets = []
        total_queries = 0
        
        for dataset in ["samples", "squad", "synthetic"]:
            dataset_path = data_dir / dataset / "queries.json"
            if dataset_path.exists():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
                available_datasets.append(dataset)
                total_queries += len(queries)
                print(f"  ✅ {dataset}: {len(queries)}个查询")
            else:
                print(f"  ❌ {dataset}: 数据集不存在")
        
        print(f"✅ 环境检查通过，总计 {total_queries} 个查询可用")
        
    except ImportError as e:
        print(f"❌ 环境检查失败: {e}")
        return 1
    
    # 大规模实验配置
    experiment_configs = [
        {
            "name": "小规模基准测试",
            "description": "验证系统基础性能",
            "sample_size": 10,
            "datasets": available_datasets,
            "repetitions": 1
        },
        {
            "name": "中等规模性能测试", 
            "description": "测试系统在中等负载下的性能",
            "sample_size": 50,
            "datasets": available_datasets,
            "repetitions": 1
        },
        {
            "name": "大规模压力测试",
            "description": "测试系统在高负载下的性能和稳定性",
            "sample_size": 100,
            "datasets": available_datasets,
            "repetitions": 1
        }
    ]
    
    # 运行实验序列
    all_results = []
    
    for i, config in enumerate(experiment_configs, 1):
        print(f"\n{'='*70}")
        print(f"🧪 实验 {i}/{len(experiment_configs)}: {config['name']}")
        print(f"📝 描述: {config['description']}")
        print(f"📊 配置: {config['sample_size']} 样本/数据集, {len(config['datasets'])} 数据集")
        print(f"{'='*70}")
        
        try:
            # 创建实验配置文件
            config_path = f"experiments/large_scale_config_{i}.json"
            experiment_config = {
                "datasets": config["datasets"],
                "systems": ["intelligent_rag"],
                "metrics": ["retrieval_quality", "answer_quality", "efficiency"],
                "sample_size": config["sample_size"],
                "repetitions": config["repetitions"],
                "timeout": 60
            }
            
            os.makedirs("experiments", exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(experiment_config, f, indent=2)
            
            # 运行实验
            start_time = time.time()
            runner = ExperimentRunner(config_path)
            results = runner.run_experiment()
            end_time = time.time()
            
            if "error" in results:
                print(f"❌ 实验失败: {results['error']}")
                continue
            
            # 分析结果
            experiment_analysis = analyze_experiment_results(results, config, end_time - start_time)
            all_results.append(experiment_analysis)
            
            # 显示实验结果
            display_experiment_results(experiment_analysis)
            
        except Exception as e:
            print(f"❌ 实验 {i} 执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 综合分析
    if all_results:
        print(f"\n{'='*70}")
        print("📊 综合实验分析")
        print(f"{'='*70}")
        
        comprehensive_analysis = analyze_comprehensive_results(all_results)
        display_comprehensive_analysis(comprehensive_analysis)
        
        # 保存综合报告
        save_comprehensive_report(all_results, comprehensive_analysis)
    
    print(f"\n{'='*70}")
    print("🎉 大规模实验完成！")
    print("📁 详细结果保存在 experiments/results/ 目录")
    print("📄 综合报告保存在 experiments/comprehensive_report.md")
    print(f"{'='*70}")
    
    return 0


def analyze_experiment_results(results: Dict[str, Any], config: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """分析单个实验结果"""
    analysis = {
        "config": config,
        "duration": duration,
        "datasets": {},
        "overall": {}
    }
    
    total_queries = 0
    total_successful = 0
    all_confidences = []
    all_processing_times = []
    all_complexities = []
    
    # 分析各数据集结果
    for dataset_name, dataset_results in results["results"].items():
        for system_name, system_result in dataset_results.items():
            if "error" in system_result:
                continue
            
            stats = system_result.get("system_stats", {})
            successful = system_result.get("successful_queries", 0)
            total = system_result.get("total_queries", 0)
            
            total_queries += total
            total_successful += successful
            
            if stats.get("avg_confidence"):
                all_confidences.append(stats["avg_confidence"])
            if stats.get("avg_processing_time"):
                all_processing_times.append(stats["avg_processing_time"])
            if stats.get("avg_complexity"):
                all_complexities.append(stats["avg_complexity"])
            
            # 数据集级别分析
            analysis["datasets"][dataset_name] = {
                "total_queries": total,
                "successful_queries": successful,
                "success_rate": successful / total if total > 0 else 0,
                "avg_confidence": stats.get("avg_confidence", 0),
                "avg_processing_time": stats.get("avg_processing_time", 0),
                "throughput": stats.get("throughput", 0),
                "avg_complexity": stats.get("avg_complexity", 0)
            }
    
    # 整体分析
    analysis["overall"] = {
        "total_queries": total_queries,
        "successful_queries": total_successful,
        "success_rate": total_successful / total_queries if total_queries > 0 else 0,
        "avg_confidence": statistics.mean(all_confidences) if all_confidences else 0,
        "avg_processing_time": statistics.mean(all_processing_times) if all_processing_times else 0,
        "total_throughput": total_successful / duration if duration > 0 else 0,
        "avg_complexity": statistics.mean(all_complexities) if all_complexities else 0
    }
    
    # 性能等级评估
    avg_time = analysis["overall"]["avg_processing_time"]
    if avg_time < 0.1:
        performance_level = "🚀 优秀"
    elif avg_time < 0.5:
        performance_level = "✅ 良好"
    elif avg_time < 1.0:
        performance_level = "⚠️  一般"
    else:
        performance_level = "❌ 需要优化"
    
    analysis["performance_level"] = performance_level
    
    return analysis


def display_experiment_results(analysis: Dict[str, Any]):
    """显示实验结果"""
    config = analysis["config"]
    overall = analysis["overall"]
    
    print(f"\n📊 实验结果: {config['name']}")
    print("-" * 50)
    print(f"总查询数: {overall['total_queries']}")
    print(f"成功查询数: {overall['successful_queries']}")
    print(f"成功率: {overall['success_rate']:.1%}")
    print(f"平均置信度: {overall['avg_confidence']:.1%}")
    print(f"平均处理时间: {overall['avg_processing_time']:.3f}秒")
    print(f"总体吞吐量: {overall['total_throughput']:.1f} queries/sec")
    print(f"平均查询复杂度: {overall['avg_complexity']:.2f}/5.0")
    print(f"性能等级: {analysis['performance_level']}")
    
    # 各数据集详情
    print(f"\n📋 各数据集详情:")
    for dataset_name, dataset_stats in analysis["datasets"].items():
        print(f"  {dataset_name}:")
        print(f"    成功率: {dataset_stats['success_rate']:.1%}")
        print(f"    置信度: {dataset_stats['avg_confidence']:.1%}")
        print(f"    处理时间: {dataset_stats['avg_processing_time']:.3f}s")
        print(f"    吞吐量: {dataset_stats['throughput']:.1f} q/s")


def analyze_comprehensive_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """综合分析所有实验结果"""
    analysis = {
        "scalability": {},
        "performance_trends": {},
        "stability": {},
        "recommendations": []
    }
    
    # 可扩展性分析
    sample_sizes = [r["config"]["sample_size"] for r in all_results]
    success_rates = [r["overall"]["success_rate"] for r in all_results]
    processing_times = [r["overall"]["avg_processing_time"] for r in all_results]
    throughputs = [r["overall"]["total_throughput"] for r in all_results]
    
    analysis["scalability"] = {
        "sample_sizes": sample_sizes,
        "success_rates": success_rates,
        "processing_times": processing_times,
        "throughputs": throughputs,
        "stability_score": statistics.stdev(success_rates) if len(success_rates) > 1 else 0
    }
    
    # 性能趋势分析
    if len(processing_times) > 1:
        time_trend = "increasing" if processing_times[-1] > processing_times[0] else "stable"
        throughput_trend = "increasing" if throughputs[-1] > throughputs[0] else "decreasing"
    else:
        time_trend = "stable"
        throughput_trend = "stable"
    
    analysis["performance_trends"] = {
        "processing_time_trend": time_trend,
        "throughput_trend": throughput_trend
    }
    
    # 稳定性分析
    avg_success_rate = statistics.mean(success_rates)
    min_success_rate = min(success_rates)
    
    analysis["stability"] = {
        "avg_success_rate": avg_success_rate,
        "min_success_rate": min_success_rate,
        "stability_rating": "excellent" if min_success_rate > 0.95 else "good" if min_success_rate > 0.8 else "needs_improvement"
    }
    
    # 生成建议
    recommendations = []
    
    if avg_success_rate > 0.95:
        recommendations.append("✅ 系统稳定性优秀，可以进行生产部署")
    elif avg_success_rate > 0.8:
        recommendations.append("⚠️  系统基本稳定，建议进一步优化")
    else:
        recommendations.append("❌ 系统稳定性需要改进")
    
    if time_trend == "increasing":
        recommendations.append("⚠️  处理时间随负载增加，需要性能优化")
    else:
        recommendations.append("✅ 处理时间保持稳定")
    
    if throughput_trend == "decreasing":
        recommendations.append("⚠️  吞吐量随负载下降，需要扩展性优化")
    else:
        recommendations.append("✅ 吞吐量表现良好")
    
    analysis["recommendations"] = recommendations
    
    return analysis


def display_comprehensive_analysis(analysis: Dict[str, Any]):
    """显示综合分析结果"""
    scalability = analysis["scalability"]
    stability = analysis["stability"]
    
    print("🔍 可扩展性分析:")
    print(f"  样本规模: {scalability['sample_sizes']}")
    print(f"  成功率: {[f'{r:.1%}' for r in scalability['success_rates']]}")
    print(f"  处理时间: {[f'{t:.3f}s' for t in scalability['processing_times']]}")
    print(f"  吞吐量: {[f'{tp:.1f} q/s' for tp in scalability['throughputs']]}")
    print(f"  稳定性评分: {scalability['stability_score']:.3f}")
    
    print(f"\n📈 性能趋势:")
    print(f"  处理时间趋势: {analysis['performance_trends']['processing_time_trend']}")
    print(f"  吞吐量趋势: {analysis['performance_trends']['throughput_trend']}")
    
    print(f"\n🎯 稳定性评估:")
    print(f"  平均成功率: {stability['avg_success_rate']:.1%}")
    print(f"  最低成功率: {stability['min_success_rate']:.1%}")
    print(f"  稳定性等级: {stability['stability_rating']}")
    
    print(f"\n💡 建议:")
    for recommendation in analysis["recommendations"]:
        print(f"  {recommendation}")


def save_comprehensive_report(all_results: List[Dict[str, Any]], analysis: Dict[str, Any]):
    """保存综合报告"""
    report_path = "experiments/comprehensive_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 智能自适应RAG系统 - 大规模实验综合报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验概览\n\n")
        f.write(f"- **实验数量**: {len(all_results)}\n")
        f.write(f"- **总查询数**: {sum(r['overall']['total_queries'] for r in all_results)}\n")
        f.write(f"- **平均成功率**: {analysis['stability']['avg_success_rate']:.1%}\n\n")
        
        f.write("## 详细实验结果\n\n")
        for i, result in enumerate(all_results, 1):
            config = result["config"]
            overall = result["overall"]
            
            f.write(f"### 实验 {i}: {config['name']}\n\n")
            f.write(f"- **样本规模**: {config['sample_size']}\n")
            f.write(f"- **成功率**: {overall['success_rate']:.1%}\n")
            f.write(f"- **平均置信度**: {overall['avg_confidence']:.1%}\n")
            f.write(f"- **平均处理时间**: {overall['avg_processing_time']:.3f}秒\n")
            f.write(f"- **吞吐量**: {overall['total_throughput']:.1f} queries/sec\n")
            f.write(f"- **性能等级**: {result['performance_level']}\n\n")
        
        f.write("## 综合分析\n\n")
        f.write("### 可扩展性\n\n")
        scalability = analysis["scalability"]
        f.write(f"- **稳定性评分**: {scalability['stability_score']:.3f}\n")
        f.write(f"- **处理时间趋势**: {analysis['performance_trends']['processing_time_trend']}\n")
        f.write(f"- **吞吐量趋势**: {analysis['performance_trends']['throughput_trend']}\n\n")
        
        f.write("### 建议\n\n")
        for recommendation in analysis["recommendations"]:
            f.write(f"- {recommendation}\n")
        
        f.write("\n---\n\n")
        f.write("*报告由智能自适应RAG系统自动生成*\n")
    
    print(f"📄 综合报告已保存: {report_path}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
