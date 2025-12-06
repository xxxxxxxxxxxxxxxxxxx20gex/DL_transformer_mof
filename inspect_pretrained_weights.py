"""
预训练权重查看工具

查看预训练权重文件的详细信息，包括参数名称、形状、统计信息等。

使用方法:
    python inspect_pretrained_weights.py
    python inspect_pretrained_weights.py --weight_path ./ckpt/pretraining/model_transformer_14.pth
    python inspect_pretrained_weights.py --show_values  # 显示部分参数值
"""

import argparse
import torch
import yaml
from pathlib import Path
import numpy as np


def format_number(num):
    """格式化数字，添加千位分隔符"""
    return f"{num:,}"


def print_weight_info(weight_dict, show_values=False, show_stats=True):
    """
    打印权重信息
    
    Args:
        weight_dict: 权重字典
        show_values: 是否显示部分参数值
        show_stats: 是否显示统计信息
    """
    print("\n" + "=" * 120)
    print("  预训练权重文件内容")
    print("=" * 120)
    print(f"{'序号':<6} {'参数名称':<60} {'形状':<25} {'参数量':<15}")
    print("-" * 120)
    
    total_params = 0
    param_list = []
    
    for idx, (name, param) in enumerate(weight_dict.items(), 1):
        if isinstance(param, torch.Tensor):
            shape = list(param.shape)
            num_params = param.numel()
            total_params += num_params
            
            shape_str = str(shape)
            print(f"{idx:<6} {name:<60} {shape_str:<25} {format_number(num_params):<15}")
            
            param_list.append({
                'name': name,
                'shape': shape,
                'num_params': num_params,
                'tensor': param
            })
    
    print("-" * 120)
    print(f"{'总计':<6} {len(weight_dict)} 个参数{'':<48} {'总参数量:':<25} {format_number(total_params):<15}")
    print(f"{'':<6} {'':<60} {'模型大小:':<25} {total_params * 4 / 1024 / 1024:.2f} MB")
    print("=" * 120)
    
    # # 显示统计信息
    # if show_stats and param_list:
    #     print("\n" + "=" * 120)
    #     print("  参数统计信息（前10个参数）")
    #     print("=" * 120)
    #     print(f"{'参数名称':<60} {'最小值':<12} {'最大值':<12} {'均值':<12} {'标准差':<12}")
    #     print("-" * 120)
        
    #     for item in param_list[:10]:
    #         name = item['name']
    #         tensor = item['tensor'].float()
            
    #         min_val = tensor.min().item()
    #         max_val = tensor.max().item()
    #         mean_val = tensor.mean().item()
    #         std_val = tensor.std().item()
            
    #         print(f"{name:<60} {min_val:<12.6f} {max_val:<12.6f} {mean_val:<12.6f} {std_val:<12.6f}")
        
    #     if len(param_list) > 10:
    #         print(f"... 还有 {len(param_list) - 10} 个参数")
    #     print("=" * 120)
    
    # # 显示部分参数值
    # if show_values and param_list:
    #     print("\n" + "=" * 120)
    #     print("  参数值示例（第一个参数的前20个值）")
    #     print("=" * 120)
    #     first_param = param_list[0]
    #     print(f"参数名称: {first_param['name']}")
    #     print(f"参数形状: {first_param['shape']}")
        
    #     tensor = first_param['tensor'].flatten()
    #     num_show = min(20, len(tensor))
    #     values = tensor[:num_show].tolist()
        
    #     print(f"\n前 {num_show} 个值:")
    #     for i, val in enumerate(values):
    #         print(f"  [{i}] = {val:.8f}")
        
    #     if len(tensor) > num_show:
    #         print(f"  ... (共 {format_number(len(tensor))} 个值)")
    #     print("=" * 120)
    
    return param_list


def compare_with_model(weight_dict, config):
    """
    对比预训练权重与当前模型配置
    
    Args:
        weight_dict: 预训练权重字典
        config: 配置字典
    """
    from model.transformer import Transformer
    
    print("\n" + "=" * 100)
    print("  预训练权重与当前模型对比")
    print("=" * 100)
    
    # 创建当前模型
    transformer = Transformer(**config['Transformer'])
    model_state = transformer.state_dict()
    
    print(f"\n{'项目':<40} {'预训练权重':<20} {'当前模型':<20}")
    print("-" * 100)
    print(f"{'参数数量':<40} {len(weight_dict):<20} {len(model_state):<20}")
    
    # 统计参数量
    pretrain_params = sum(p.numel() for p in weight_dict.values() if isinstance(p, torch.Tensor))
    model_params = sum(p.numel() for p in model_state.values())
    
    print(f"{'总参数量':<40} {format_number(pretrain_params):<20} {format_number(model_params):<20}")
    print(f"{'模型大小 (MB)':<40} {pretrain_params * 4 / 1024 / 1024:.2f}{'':>16} {model_params * 4 / 1024 / 1024:.2f}")
    
    print("\n" + "-" * 100)
    print("  参数匹配情况")
    print("-" * 100)
    
    # 检查匹配
    matched = []
    shape_mismatch = []
    only_in_pretrain = []
    only_in_model = []
    
    for name in weight_dict.keys():
        if name in model_state:
            if weight_dict[name].shape == model_state[name].shape:
                matched.append(name)
            else:
                shape_mismatch.append({
                    'name': name,
                    'pretrain_shape': list(weight_dict[name].shape),
                    'model_shape': list(model_state[name].shape)
                })
        else:
            only_in_pretrain.append(name)
    
    for name in model_state.keys():
        if name not in weight_dict:
            only_in_model.append(name)
    
    print(f"\n✓ 完全匹配:         {len(matched)} 个参数")
    print(f"✗ 形状不匹配:       {len(shape_mismatch)} 个参数")
    print(f"⚠ 仅在预训练中:     {len(only_in_pretrain)} 个参数")
    print(f"⚠ 仅在当前模型中:   {len(only_in_model)} 个参数 (将随机初始化)")
    
    if matched:
        match_percentage = len(matched) / len(model_state) * 100
        print(f"\n匹配率: {match_percentage:.2f}% ({len(matched)}/{len(model_state)})")
    
    # 详细信息
    if shape_mismatch:
        print("\n❌ 形状不匹配的参数:")
        for item in shape_mismatch:
            print(f"  {item['name']}")
            print(f"    预训练: {item['pretrain_shape']} vs 模型: {item['model_shape']}")
    
    if only_in_model:
        print("\n⚠ 需要随机初始化的参数 (不在预训练权重中):")
        for name in only_in_model[:10]:
            print(f"  - {name}")
        if len(only_in_model) > 10:
            print(f"  ... 还有 {len(only_in_model) - 10} 个参数")
    
    if only_in_pretrain:
        print("\n⚠ 预训练中多余的参数 (不在当前模型中):")
        for name in only_in_pretrain[:10]:
            print(f"  - {name}")
        if len(only_in_pretrain) > 10:
            print(f"  ... 还有 {len(only_in_pretrain) - 10} 个参数")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='查看预训练权重信息')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='预训练权重文件路径')
    parser.add_argument('--config', type=str, default='config_ft_transformer.yaml',
                        help='配置文件路径 (默认: config_ft_transformer.yaml)')
    parser.add_argument('--show_values', action='store_true',
                        help='显示部分参数值')
    parser.add_argument('--show_stats', action='store_true', default=True,
                        help='显示统计信息 (默认: True)')
    parser.add_argument('--compare', action='store_true', default=True,
                        help='与当前模型配置对比 (默认: True)')
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"✓ 成功加载配置文件: {args.config}")
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {args.config}")
        return
    except Exception as e:
        print(f"❌ 加载配置文件时出错: {e}")
        return
    
    # 确定权重文件路径
    if args.weight_path:
        weight_path = Path(args.weight_path)
    else:
        checkpoints_folder = config.get('fine_tune_from', './ckpt/pretraining')
        model_file = config.get('pretrained_model_file', 'model_transformer_14.pth')
        weight_path = Path(checkpoints_folder) / model_file
    
    print(f"预训练权重文件: {weight_path}")
    
    # 检查文件是否存在
    if not weight_path.exists():
        print(f"❌ 预训练权重文件不存在: {weight_path}")
        print("\n可用的预训练权重文件:")
        
        parent_dir = weight_path.parent
        if parent_dir.exists():
            weight_files = list(parent_dir.glob('*.pth'))
            if weight_files:
                for f in weight_files:
                    print(f"  - {f}")
            else:
                print("  (未找到任何 .pth 文件)")
        else:
            print(f"  (目录不存在: {parent_dir})")
        return
    
    # 加载权重
    print(f"\n正在加载预训练权重...")
    try:
        weight_dict = torch.load(str(weight_path), map_location='cpu')
        print(f"✓ 成功加载预训练权重")
    except Exception as e:
        print(f"❌ 加载预训练权重时出错: {e}")
        return
    
    # 1. 打印权重信息
    param_list = print_weight_info(weight_dict, 
                                   show_values=args.show_values, 
                                   show_stats=args.show_stats)

    # 3. 与当前模型对比
    if args.compare:
        try:
            compare_with_model(weight_dict, config)
        except Exception as e:
            print(f"\n⚠ 无法对比模型: {e}")
    
    print("\n✓ 预训练权重检查完成！")
    print("\n提示:")
    print("  - 使用 --show_values 查看部分参数值")
    print("  - 使用 --weight_path 指定其他权重文件")
    print("\n示例:")
    print("  python inspect_pretrained_weights.py --show_values")
    print("  python inspect_pretrained_weights.py --weight_path ./ckpt/pretraining/model_transformer_14.pth")


if __name__ == "__main__":
    main()

