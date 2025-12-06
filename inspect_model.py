"""
模型结构查看工具

用于查看Transformer微调模型的参数结构、参数量统计等信息。

使用方法:
    python inspect_model.py
    python inspect_model.py --config config_ft_transformer.yaml
    python inspect_model.py --load_pretrained  # 加载预训练权重并查看
"""

import argparse
import yaml
import torch
from pathlib import Path
from model.transformer import Transformer, TransformerRegressor


def count_parameters(model, trainable_only=False):
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
        trainable_only: 是否只统计可训练参数
    
    Returns:
        参数总数
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_number(num):
    """格式化数字，添加千位分隔符"""
    return f"{num:,}"


def print_model_structure(model, model_name="Model"):
    """
    打印模型结构和参数统计
    
    Args:
        model: PyTorch模型
        model_name: 模型名称
    """
    print("\n" + "=" * 80)
    print(f"  {model_name} 结构")
    print("=" * 80)
    print(model)
    print("\n" + "=" * 80)
    print(f"  {model_name} 参数统计")
    print("=" * 80)
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params
    
    print(f"总参数量:        {format_number(total_params)}")
    print(f"可训练参数:      {format_number(trainable_params)}")
    print(f"冻结参数:        {format_number(frozen_params)}")
    print(f"参数大小 (MB):   {total_params * 4 / 1024 / 1024:.2f}")
    print("=" * 80)


def print_layer_parameters(model, show_shapes=True):
    """
    打印每一层的参数信息
    
    Args:
        model: PyTorch模型
        show_shapes: 是否显示参数形状
    """
    print("\n" + "=" * 100)
    print("  各层参数详情")
    print("=" * 100)
    print(f"{'层名称':<50} {'参数量':<15} {'可训练':<10} {'形状':<20}")
    print("-" * 100)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        is_trainable = param.requires_grad
        shape_str = str(list(param.shape)) if show_shapes else ""
        
        total_params += num_params
        if is_trainable:
            trainable_params += num_params
        
        trainable_mark = "✓" if is_trainable else "✗"
        print(f"{name:<50} {format_number(num_params):<15} {trainable_mark:<10} {shape_str:<20}")
    
    print("-" * 100)
    print(f"{'总计':<50} {format_number(total_params):<15} {format_number(trainable_params):<15}")
    print("=" * 100)


def print_module_statistics(model):
    """
    打印模块级别的参数统计
    
    Args:
        model: PyTorch模型
    """
    print("\n" + "=" * 80)
    print("  模块级别参数统计")
    print("=" * 80)
    print(f"{'模块名称':<40} {'参数量':<20} {'占比':<15}")
    print("-" * 80)
    
    total_params = count_parameters(model)
    module_params = {}
    
    # 统计每个主要模块的参数
    for name, module in model.named_children():
        params = count_parameters(module)
        module_params[name] = params
    
    # 按参数量排序
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    
    for name, params in sorted_modules:
        percentage = (params / total_params * 100) if total_params > 0 else 0
        print(f"{name:<40} {format_number(params):<20} {percentage:>6.2f}%")
    
    print("-" * 80)
    print(f"{'总计':<40} {format_number(total_params):<20} {'100.00%':>15}")
    print("=" * 80)


def inspect_pretrained_loading(config):
    """
    检查预训练权重加载情况
    
    Args:
        config: 配置字典
    """
    print("\n" + "=" * 80)
    print("  预训练权重加载检查")
    print("=" * 80)
    
    # 创建模型
    transformer = Transformer(**config['Transformer'])
    
    try:
        # 尝试加载预训练权重
        checkpoints_folder = config['fine_tune_from']
        model_file = config.get('pretrained_model_file', 'model_transformer_14.pth')
        weight_path = Path(checkpoints_folder) / model_file
        
        if not weight_path.exists():
            print(f"❌ 预训练权重文件不存在: {weight_path}")
            return
        
        print(f"✓ 找到预训练权重文件: {weight_path}")
        
        # 加载权重
        load_state = torch.load(str(weight_path), map_location='cpu')
        model_state = transformer.state_dict()
        
        print(f"\n预训练文件中的参数数量: {len(load_state)}")
        print(f"当前模型的参数数量:     {len(model_state)}")
        
        # 检查匹配情况
        matched = []
        shape_mismatch = []
        not_in_model = []
        not_in_checkpoint = []
        
        for name in load_state.keys():
            if name in model_state:
                if load_state[name].shape == model_state[name].shape:
                    matched.append(name)
                else:
                    shape_mismatch.append((name, load_state[name].shape, model_state[name].shape))
            else:
                not_in_model.append(name)
        
        for name in model_state.keys():
            if name not in load_state:
                not_in_checkpoint.append(name)
        
        # 打印结果
        print("\n" + "-" * 80)
        print(f"✓ 匹配的参数:       {len(matched)} 个")
        print(f"✗ 形状不匹配:       {len(shape_mismatch)} 个")
        print(f"✗ 不在模型中:       {len(not_in_model)} 个")
        print(f"⚠ 不在预训练文件中:  {len(not_in_checkpoint)} 个 (这些将随机初始化)")
        
        if matched:
            print("\n匹配的参数（前10个）:")
            for name in matched[:10]:
                print(f"  ✓ {name}")
            if len(matched) > 10:
                print(f"  ... 还有 {len(matched) - 10} 个参数")
        
        if shape_mismatch:
            print("\n❌ 形状不匹配的参数:")
            for name, ckpt_shape, model_shape in shape_mismatch:
                print(f"  ✗ {name}")
                print(f"    预训练: {ckpt_shape} → 模型: {model_shape}")
        
        if not_in_checkpoint:
            print("\n⚠ 需要随机初始化的参数（不在预训练文件中）:")
            for name in not_in_checkpoint:
                print(f"  ⚠ {name} - 将使用随机初始化")
        
        if not_in_model:
            print("\n⚠ 预训练文件中多余的参数（不在当前模型中）:")
            for name in not_in_model[:10]:
                print(f"  ⚠ {name}")
            if len(not_in_model) > 10:
                print(f"  ... 还有 {len(not_in_model) - 10} 个参数")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 加载预训练权重时出错: {e}")
        print("=" * 80)


def compare_models(config):
    """
    比较Transformer主干和完整微调模型的参数
    
    Args:
        config: 配置字典
    """
    print("\n" + "=" * 80)
    print("  模型对比")
    print("=" * 80)
    
    # 创建Transformer主干
    transformer = Transformer(**config['Transformer'])
    
    # 创建完整的微调模型
    model_ft = TransformerRegressor(
        transformer=transformer,
        d_model=config['Transformer']['d_model']
    )
    
    transformer_params = count_parameters(transformer)
    total_params = count_parameters(model_ft)
    regression_head_params = total_params - transformer_params
    
    print(f"\n{'模型组件':<30} {'参数量':<20} {'占比':<15}")
    print("-" * 80)
    print(f"{'Transformer主干':<30} {format_number(transformer_params):<20} {transformer_params/total_params*100:>6.2f}%")
    print(f"{'回归头(RegressionHead)':<30} {format_number(regression_head_params):<20} {regression_head_params/total_params*100:>6.2f}%")
    print("-" * 80)
    print(f"{'微调模型总计':<30} {format_number(total_params):<20} {'100.00%':>15}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='查看模型结构和参数信息')
    parser.add_argument('--config', type=str, default='config_ft_transformer.yaml',
                        help='配置文件路径 (默认: config_ft_transformer.yaml)')
    parser.add_argument('--load_pretrained', action='store_true',
                        help='检查预训练权重加载情况')
    parser.add_argument('--show_shapes', action='store_true', default=True,
                        help='显示参数形状 (默认: True)')
    parser.add_argument('--detailed', action='store_true',
                        help='显示详细的层级参数信息')
    
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
    
    # 创建模型
    print("\n正在创建模型...")
    transformer = Transformer(**config['Transformer'])
    model = TransformerRegressor(
        transformer=transformer,
        d_model=config['Transformer']['d_model']
    )
    
    # 1. 打印模型结构和基本统计
    print_model_structure(model, "TransformerRegressor (完整微调模型)")
    
    # 2. 打印模块统计
    print_module_statistics(model)
    
    # 3. 比较模型组件
    compare_models(config)
    
    # 4. 详细的层级参数信息
    if args.detailed:
        print_layer_parameters(model, show_shapes=args.show_shapes)
    
    # 5. 检查预训练权重加载（如果指定）
    if args.load_pretrained:
        inspect_pretrained_loading(config)
    
    # 打印配置信息摘要
    print("\n" + "=" * 80)
    print("  配置信息摘要")
    print("=" * 80)
    print(f"词汇表大小 (ntoken):    {config['Transformer']['ntoken']}")
    print(f"模型维度 (d_model):     {config['Transformer']['d_model']}")
    print(f"注意力头数 (nhead):     {config['Transformer']['nhead']}")
    print(f"隐藏层维度 (d_hid):     {config['Transformer']['d_hid']}")
    print(f"Transformer层数:        {config['Transformer']['nlayers']}")
    print(f"Dropout:                {config['Transformer']['dropout']}")
    print(f"批次大小 (batch_size):  {config['batch_size']}")
    print(f"训练轮数 (epochs):      {config['epochs']}")
    print(f"GPU设备:                {config['gpu']}")
    print("=" * 80)
    
    print("\n✓ 模型检查完成！")
    print("\n提示:")
    print("  - 使用 --detailed 查看详细的层级参数信息")
    print("  - 使用 --load_pretrained 检查预训练权重加载情况")
    print("\n示例:")
    print("  python inspect_model.py --detailed")
    print("  python inspect_model.py --load_pretrained")
    print("  python inspect_model.py --detailed --load_pretrained")


if __name__ == "__main__":
    main()

