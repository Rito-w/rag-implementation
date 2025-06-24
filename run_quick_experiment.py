#!/usr/bin/env python3
"""
快速实验脚本

快速运行智能自适应RAG系统的基础实验，验证系统功能和性能。
适合初次使用和快速验证。
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
    print("🚀 智能自适应RAG系统 - 快速实验")
    print("=" * 60)
    
    # 检查环境
    print("🔍 检查环境...")
    try:
        from src.core.intelligent_adapter import IntelligentAdaptiveRAG
        from experiments.experiment_runner import ExperimentRunner
        from scripts.download_datasets import DatasetDownloader
        print("✅ 环境检查通过")
    except ImportError as e:
        print(f"❌ 环境检查失败: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        return 1
    
    # 步骤1: 准备样本数据
    print("\n📊 步骤1: 准备样本数据...")
    try:
        downloader = DatasetDownloader("data")
        downloader.create_sample_datasets()
        print("✅ 样本数据准备完成")
    except Exception as e:
        print(f"❌ 样本数据准备失败: {e}")
        return 1
    
    # 步骤2: 运行基础功能测试
    print("\n🧪 步骤2: 运行基础功能测试...")
    try:
        os.system("python test_core_components.py")
        print("✅ 基础功能测试完成")
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return 1
    
    # 步骤3: 运行快速实验
    print("\n⚡ 步骤3: 运行快速实验...")
    try:
        # 创建实验配置
        quick_config = {
            "datasets": ["samples"],
            "systems": ["intelligent_rag"],
            "metrics": ["retrieval_quality", "answer_quality", "efficiency"],
            "sample_size": 5,
            "repetitions": 1,
            "timeout": 30
        }
        
        # 保存配置
        config_path = "experiments/quick_config.json"
        os.makedirs("experiments", exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(quick_config, f, indent=2)
        
        # 运行实验
        runner = ExperimentRunner(config_path)
        results = runner.run_experiment()
        
        if "error" in results:
            print(f"❌ 实验失败: {results['error']}")
            return 1
        
        print("✅ 快速实验完成")
        
        # 显示结果摘要
        print("\n📈 实验结果摘要:")
        print("-" * 40)
        
        for dataset_name, dataset_results in results["results"].items():
            print(f"\n数据集: {dataset_name}")
            
            for system_name, system_result in dataset_results.items():
                if "error" in system_result:
                    print(f"  {system_name}: ❌ 错误")
                    continue
                
                stats = system_result.get("system_stats", {})
                print(f"  {system_name}:")
                print(f"    - 成功查询: {system_result.get('successful_queries', 0)}/{system_result.get('total_queries', 0)}")
                print(f"    - 平均置信度: {stats.get('avg_confidence', 0):.1%}")
                print(f"    - 平均处理时间: {stats.get('avg_processing_time', 0):.3f}秒")
                print(f"    - 吞吐量: {stats.get('throughput', 0):.1f} queries/sec")
        
        print(f"\n📁 详细结果保存在: experiments/results/{results['experiment_id']}_results.json")
        print(f"📄 实验报告保存在: experiments/results/{results['experiment_id']}_report.md")
        
    except Exception as e:
        print(f"❌ 快速实验失败: {e}")
        return 1
    
    # 步骤4: 运行演示
    print("\n🎬 步骤4: 运行系统演示...")
    try:
        print("运行演示脚本...")
        os.system("python demo.py")
        print("✅ 系统演示完成")
    except Exception as e:
        print(f"❌ 系统演示失败: {e}")
        return 1
    
    # 总结
    print("\n" + "=" * 60)
    print("🎉 快速实验完成！")
    print("\n📋 下一步建议:")
    print("1. 📊 下载完整数据集: python scripts/download_datasets.py")
    print("2. 🔬 运行完整实验: python experiments/experiment_runner.py")
    print("3. 📈 分析实验结果: 查看 experiments/results/ 目录")
    print("4. ⚙️  调优系统参数: 修改 configs/default.yaml")
    print("5. 📝 撰写论文: 基于实验结果撰写技术论文")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
