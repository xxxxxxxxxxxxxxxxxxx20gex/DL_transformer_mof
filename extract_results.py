"""
实验结果提取工具

自动从训练日志中提取实验结果，方便记录到实验日志中。

使用方法:
    python extract_results.py --log_dir training_results/finetuning/Transformer/Trans_Transformer_hMOF_CO2_0.5_1_2025-12-06_11-58-16
    python extract_results.py --log_dir training_results/finetuning/Transformer/latest
"""

import argparse
import csv
import yaml
from pathlib import Path
import re
from datetime import datetime


def extract_from_config(config_path):
    """从配置文件提取信息"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except Exception as e:
        print(f"⚠ 无法读取配置文件: {e}")
        return None


def extract_from_log(log_path):
    """从训练日志提取信息"""
    results = {
        'best_valid_mae': None,
        'best_valid_loss': None,
        'best_epoch': None,
        'test_mae': None,
        'test_loss': None,
        'total_epochs': 0
    }
    
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        # 提取最佳验证MAE
        valid_pattern = r'Validation Complete - Final MAE: ([\d.]+)'
        valid_matches = re.findall(valid_pattern, log_content)
        if valid_matches:
            valid_maes = [float(m) for m in valid_matches]
            results['best_valid_mae'] = min(valid_maes)
        
        # 提取测试结果
        test_pattern = r'Test Complete - Final MAE: ([\d.]+)'
        test_match = re.search(test_pattern, log_content)
        if test_match:
            results['test_mae'] = float(test_match.group(1))
        
        # 提取epoch数
        epoch_pattern = r'Epoch\s+(\d+)'
        epoch_matches = re.findall(epoch_pattern, log_content)
        if epoch_matches:
            results['total_epochs'] = max([int(e) for e in epoch_matches])
        
    except Exception as e:
        print(f"⚠ 无法读取日志文件: {e}")
    
    return results


def extract_from_test_results(test_results_path):
    """从测试结果CSV提取信息"""
    try:
        with open(test_results_path, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        
        if data:
            # 最后一行通常是汇总结果
            last_row = data[-1]
            if len(last_row) >= 2:
                return {
                    'test_loss': float(last_row[0]),
                    'test_mae': float(last_row[1])
                }
    except Exception as e:
        print(f"⚠ 无法读取测试结果: {e}")
    
    return None


def format_experiment_summary(log_dir):
    """格式化实验摘要"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"❌ 目录不存在: {log_dir}")
        return
    
    print("\n" + "=" * 100)
    print("  实验结果摘要")
    print("=" * 100)
    
    # 读取配置
    config_path = log_path / 'checkpoints' / 'config_ft_transformer.yaml'
    config = extract_from_config(config_path)
    
    # 读取训练日志
    log_file = log_path / 'training.log'
    log_results = extract_from_log(log_file)
    
    # 读取测试结果
    test_results_file = log_path / 'test_results.csv'
    test_results = extract_from_test_results(test_results_file)
    
    # 提取实验时间
    dir_name = log_path.name
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
    timestamp_match = re.search(timestamp_pattern, dir_name)
    exp_time = timestamp_match.group(1) if timestamp_match else "未知"
    
    print("\n📋 基本信息")
    print("-" * 100)
    print(f"实验目录: {log_path}")
    print(f"实验时间: {exp_time}")
    
    if config:
        print(f"\n📊 配置信息")
        print("-" * 100)
        print(f"数据集: {config['dataset']['data_name']}")
        print(f"数据路径: {config['dataset']['dataPath']}")
        print(f"随机种子: {config['dataloader']['randomSeed']}")
        print(f"批次大小: {config['batch_size']}")
        print(f"训练轮数: {config['epochs']}")
        print(f"学习率: {config['optim']['init_lr']}")
        print(f"优化器: {config['optim']['optimizer']}")
        print(f"预训练权重: {config['fine_tune_from']}")
        
        print(f"\n🏗️  模型配置")
        print("-" * 100)
        tf_config = config['Transformer']
        print(f"词汇表大小: {tf_config['ntoken']}")
        print(f"模型维度: {tf_config['d_model']}")
        print(f"注意力头数: {tf_config['nhead']}")
        print(f"前馈网络维度: {tf_config['d_hid']}")
        print(f"Transformer层数: {tf_config['nlayers']}")
        print(f"Dropout: {tf_config['dropout']}")
    
    print(f"\n📈 训练结果")
    print("-" * 100)
    if log_results['best_valid_mae']:
        print(f"最佳验证 MAE: {log_results['best_valid_mae']:.4f}")
    if log_results['total_epochs']:
        print(f"训练轮数: {log_results['total_epochs']}")
    
    if test_results:
        print(f"\n🎯 测试结果")
        print("-" * 100)
        print(f"测试 Loss: {test_results['test_loss']:.6f}")
        print(f"测试 MAE: {test_results['test_mae']:.4f}")
    
    # 生成用于记录的格式
    print("\n" + "=" * 100)
    print("  复制以下内容到 EXPERIMENTS_LOG.md")
    print("=" * 100)
    
    exp_id = f"EXP-{datetime.now().strftime('%m%d')}-{config['dataloader']['randomSeed'] if config else '1'}"
    
    print("\n### 实验汇总表格行：")
    print("```markdown")
    dataset_name = config['dataset']['data_name'] if config else "未知"
    pretrain_mark = "✓" if (config and config['fine_tune_from'] != 'scratch') else "✗"
    test_mae = f"{test_results['test_mae']:.4f}" if test_results else "-"
    print(f"| {exp_id} | {exp_time[:10]} | {dataset_name} | {pretrain_mark} | {test_mae} | [填写备注] |")
    print("```")
    
    print("\n### 详细记录：")
    print("```yaml")
    print(f"实验ID: {exp_id}")
    print(f"实验日期: {exp_time[:10]}")
    print(f"数据集: {dataset_name}")
    print(f"随机种子: {config['dataloader']['randomSeed'] if config else 1}")
    print(f"")
    print(f"模型配置:")
    if config:
        print(f"  d_model: {tf_config['d_model']}")
        print(f"  nhead: {tf_config['nhead']}")
        print(f"  nlayers: {tf_config['nlayers']}")
    print(f"")
    print(f"训练配置:")
    if config:
        print(f"  batch_size: {config['batch_size']}")
        print(f"  epochs: {config['epochs']}")
        print(f"  learning_rate: {config['optim']['init_lr']}")
        print(f"  optimizer: {config['optim']['optimizer']}")
        print(f"  pretrained: {pretrain_mark}")
    print(f"")
    print(f"结果:")
    if log_results['best_valid_mae']:
        print(f"  验证 MAE: {log_results['best_valid_mae']:.4f}")
    if test_results:
        print(f"  测试 Loss: {test_results['test_loss']:.6f}")
        print(f"  测试 MAE: {test_results['test_mae']:.4f}")
    print("```")
    
    print("\n" + "=" * 100)
    print("✓ 结果提取完成！")
    
    return {
        'exp_id': exp_id,
        'config': config,
        'log_results': log_results,
        'test_results': test_results
    }


def find_latest_experiment():
    """查找最新的实验目录"""
    results_dir = Path('training_results/finetuning/Transformer')
    
    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return None
    
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('Trans_')]
    
    if not exp_dirs:
        print(f"❌ 未找到实验目录")
        return None
    
    # 按修改时间排序
    latest_dir = max(exp_dirs, key=lambda x: x.stat().st_mtime)
    
    return latest_dir


def main():
    parser = argparse.ArgumentParser(description='提取实验结果')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='实验日志目录路径 (默认: 自动查找最新)')
    
    args = parser.parse_args()
    
    if args.log_dir:
        if args.log_dir == 'latest':
            log_dir = find_latest_experiment()
            if log_dir is None:
                return
        else:
            log_dir = Path(args.log_dir)
    else:
        print("正在查找最新实验...")
        log_dir = find_latest_experiment()
        if log_dir is None:
            return
        print(f"✓ 找到最新实验: {log_dir.name}")
    
    format_experiment_summary(log_dir)


if __name__ == "__main__":
    main()

