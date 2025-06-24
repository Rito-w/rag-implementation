#!/usr/bin/env python3
"""
真实数据集实验脚本

在真实数据集上运行智能自适应RAG系统实验，
包括SQuAD和合成数据集的完整评估。
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """主函数"""
    print("🚀 智能自适应RAG系统 - 真实数据集实验")
    print("=" * 60)
    
    # 检查环境
    print("🔍 检查环境和数据...")
    try:
        from src.core.intelligent_adapter import IntelligentAdaptiveRAG
        from experiments.experiment_runner import ExperimentRunner
        
        # 检查数据集
        data_dir = Path("data")
        available_datasets = []
        
        for dataset in ["samples", "squad", "synthetic"]:
            dataset_path = data_dir / dataset / "queries.json"
            if dataset_path.exists():
                available_datasets.append(dataset)
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
                print(f"  ✅ {dataset}: {len(queries)}个查询")
            else:
                print(f"  ❌ {dataset}: 数据集不存在")
        
        if not available_datasets:
            print("❌ 没有可用的数据集")
            return 1
            
        print(f"✅ 环境检查通过，可用数据集: {available_datasets}")
        
    except ImportError as e:
        print(f"❌ 环境检查失败: {e}")
        return 1
    
    # 运行多数据集实验
    print(f"\n🧪 运行多数据集实验...")
    try:
        # 创建实验配置
        experiment_config = {
            "datasets": available_datasets,
            "systems": ["intelligent_rag"],
            "metrics": ["retrieval_quality", "answer_quality", "efficiency"],
            "sample_size": 20,  # 每个数据集测试20个查询
            "repetitions": 1,
            "timeout": 30
        }
        
        # 保存配置
        config_path = "experiments/real_experiment_config.json"
        os.makedirs("experiments", exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, indent=2)
        
        # 运行实验
        print(f"📊 实验配置:")
        print(f"  - 数据集: {', '.join(available_datasets)}")
        print(f"  - 每个数据集样本数: {experiment_config['sample_size']}")
        print(f"  - 总查询数: {len(available_datasets) * experiment_config['sample_size']}")
        
        runner = ExperimentRunner(config_path)
        start_time = time.time()
        results = runner.run_experiment()
        end_time = time.time()
        
        if "error" in results:
            print(f"❌ 实验失败: {results['error']}")
            return 1
        
        print("✅ 真实数据集实验完成")
        
        # 显示详细结果
        print(f"\n📈 实验结果详情:")
        print("-" * 60)
        print(f"总实验时间: {end_time - start_time:.2f}秒")
        
        total_queries = 0
        total_successful = 0
        all_confidences = []
        all_processing_times = []
        
        for dataset_name, dataset_results in results["results"].items():
            print(f"\n📊 数据集: {dataset_name}")
            print("-" * 30)
            
            for system_name, system_result in dataset_results.items():
                if "error" in system_result:
                    print(f"  {system_name}: ❌ 错误 - {system_result['error']}")
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
                
                print(f"  📋 {system_name}:")
                print(f"    - 成功查询: {successful}/{total} ({successful/total*100:.1f}%)")
                print(f"    - 平均置信度: {stats.get('avg_confidence', 0):.1%}")
                print(f"    - 平均处理时间: {stats.get('avg_processing_time', 0):.3f}秒")
                print(f"    - 吞吐量: {stats.get('throughput', 0):.1f} queries/sec")
                print(f"    - 平均查询复杂度: {stats.get('avg_complexity', 0):.2f}/5.0")
        
        # 总体统计
        print(f"\n🎯 总体性能统计:")
        print("-" * 30)
        print(f"总查询数: {total_queries}")
        print(f"成功查询数: {total_successful}")
        print(f"总体成功率: {total_successful/total_queries*100:.1f}%")
        
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            print(f"平均置信度: {avg_confidence:.1%}")
        
        if all_processing_times:
            avg_time = sum(all_processing_times) / len(all_processing_times)
            total_throughput = total_successful / sum(all_processing_times) if sum(all_processing_times) > 0 else 0
            print(f"平均处理时间: {avg_time:.3f}秒")
            print(f"总体吞吐量: {total_throughput:.1f} queries/sec")
        
        print(f"\n📁 详细结果保存在: experiments/results/{results['experiment_id']}_results.json")
        print(f"📄 实验报告保存在: experiments/results/{results['experiment_id']}_report.md")
        
    except Exception as e:
        print(f"❌ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 性能分析
    print(f"\n📊 性能分析:")
    print("-" * 30)
    
    if all_processing_times and len(all_processing_times) > 1:
        import statistics
        
        min_time = min(all_processing_times)
        max_time = max(all_processing_times)
        median_time = statistics.median(all_processing_times)
        std_time = statistics.stdev(all_processing_times)
        
        print(f"处理时间分布:")
        print(f"  - 最小值: {min_time:.3f}秒")
        print(f"  - 最大值: {max_time:.3f}秒")
        print(f"  - 中位数: {median_time:.3f}秒")
        print(f"  - 标准差: {std_time:.3f}秒")
        
        # 性能等级评估
        if avg_time < 0.1:
            performance_level = "🚀 优秀 (< 0.1s)"
        elif avg_time < 0.5:
            performance_level = "✅ 良好 (< 0.5s)"
        elif avg_time < 1.0:
            performance_level = "⚠️  一般 (< 1.0s)"
        else:
            performance_level = "❌ 需要优化 (> 1.0s)"
        
        print(f"性能等级: {performance_level}")
    
    # 下一步建议
    print(f"\n📋 下一步建议:")
    print("-" * 30)
    
    if total_successful / total_queries > 0.9:
        print("✅ 系统稳定性良好，可以进行更大规模实验")
        print("🔬 建议: 增加样本数量到100-500个查询")
        print("📊 建议: 实现更多基准方法进行对比")
    elif total_successful / total_queries > 0.7:
        print("⚠️  系统基本稳定，需要进一步优化")
        print("🔧 建议: 检查失败查询的原因")
        print("⚙️  建议: 调优系统参数")
    else:
        print("❌ 系统稳定性需要改进")
        print("🐛 建议: 调试系统错误")
        print("🔧 建议: 优化核心组件")
    
    print("📝 建议: 开始撰写技术论文")
    print("🚀 建议: 准备开源发布")
    
    print("\n" + "=" * 60)
    print("🎉 真实数据集实验完成！")
    print("🎯 智能自适应RAG系统已在多个数据集上验证性能！")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
