"""
简单的MOFormer推理测试脚本

用途：加载训练好的模型，对单个或多个MOFid进行CO2吸附量预测
"""

import torch
import yaml
import numpy as np
import csv
from pathlib import Path
from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import Transformer, TransformerRegressor
from model.utils import Normalizer, split_data


class MOFPredictor:
    """MOF性质预测器"""
    
    def __init__(self, model_dir: str, config_path: str = None, device: str = 'cuda:0', 
                 enable_denormalize: bool = True):
        """
        初始化预测器
        
        Args:
            model_dir: 模型权重所在目录（包含checkpoints文件夹）
            config_path: 配置文件路径（默认从model_dir/checkpoints读取）
            device: 计算设备
            enable_denormalize: 是否启用反归一化（输出真实值）
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_dir = Path(model_dir)
        self.enable_denormalize = enable_denormalize
        
        # 加载配置
        if config_path is None:
            config_path = self.model_dir / 'checkpoints' / 'config_ft_transformer.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 初始化tokenizer
        vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(vocab_path, model_max_length=512, padding_side='right')
        
        # 初始化Normalizer（用于反归一化）
        if self.enable_denormalize:
            self._init_normalizer()
        else:
            self.normalizer = None
        
        # 构建模型
        self._build_model()
        
        # 加载权重
        self._load_weights()
        
        print(f"✓ 模型加载成功！设备: {self.device}")
        print(f"✓ 配置: {self.config['Transformer']}")
        if self.normalizer:
            print(f"✓ 反归一化已启用 (均值={self.normalizer.mean:.4f}, 标准差={self.normalizer.std:.4f})")
    
    def _init_normalizer(self):
        """初始化Normalizer（从训练数据集计算均值和标准差）"""
        print("正在初始化Normalizer...")
        
        # 加载数据集
        data_path = self.config['dataset']['dataPath']
        with open(data_path) as f:
            reader = csv.reader(f)
            data = np.array([row for row in reader])
        
        # 使用与训练相同的划分方式获取训练集
        train_data, _, _ = split_data(
            data,
            test_ratio=self.config['dataloader']['test_ratio'],
            valid_ratio=self.config['dataloader']['valid_ratio'],
            use_ratio=self.config['dataloader']['use_ratio'],
            randomSeed=self.config['dataloader']['randomSeed']
        )
        
        # 提取训练集标签（第二列）
        train_labels = train_data[:, 1].astype(float)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
        
        # 创建Normalizer
        self.normalizer = Normalizer(train_labels_tensor)
        print(f"✓ Normalizer初始化完成")
    
    def _build_model(self):
        """构建模型结构"""
        # 创建Transformer主干
        transformer = Transformer(**self.config['Transformer'])
        
        # 创建完整的回归模型
        self.model = TransformerRegressor(
            transformer=transformer,
            d_model=self.config['Transformer']['d_model']
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_weights(self):
        """加载训练好的权重"""
        model_path = self.model_dir / 'checkpoints' / 'model.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"✓ 权重加载自: {model_path}")
    
    def predict(self, mofid: str, return_normalized: bool = False) -> float:
        """
        预测单个MOFid的性质
        
        Args:
            mofid: MOFid字符串（例如："SMILES&&topology"）
            return_normalized: 是否返回归一化值（默认False，返回真实值）
            
        Returns:
            预测值（CO2吸附量，单位：mol/kg）
        """
        # Tokenize
        tokens = self.tokenizer(mofid, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokens['input_ids'].to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_ids)
        
        # 反归一化
        if self.normalizer and not return_normalized:
            output = self.normalizer.denorm(output.cpu())
            return output.item()
        else:
            return output.item()
    
    def predict_batch(self, mofids: list, return_normalized: bool = False) -> np.ndarray:
        """
        批量预测多个MOFid的性质
        
        Args:
            mofids: MOFid字符串列表
            return_normalized: 是否返回归一化值（默认False，返回真实值）
            
        Returns:
            预测值数组（CO2吸附量，单位：mol/kg）
        """
        # Tokenize批量数据
        tokens = self.tokenizer(mofids, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokens['input_ids'].to(self.device)
        
        # 批量推理
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        # 反归一化
        if self.normalizer and not return_normalized:
            outputs = self.normalizer.denorm(outputs.cpu())
            return outputs.numpy().flatten()
        else:
            return outputs.cpu().numpy().flatten()


def demo_inference():
    """演示推理流程"""
    
    # 1. 初始化预测器
    model_dir = './training_results/finetuning/Transformer/017-20260120_165056-hMOF_CO2_0.5_seed1'
    predictor = MOFPredictor(model_dir)
    
    print("\n" + "="*60)
    print("开始推理测试")
    print("="*60)
    
    # 2. 测试单个样本
    print("\n【测试1】单个MOFid预测：")
    test_mofid = "CCCOC1C(O[CH]CC)C2(C(=O)[O-])C(C(C1(C(=O)[O-])C(C2(OCCC)OCCC)OCCC)(OCCC)OCCC)O.[O-]C(=O)C#CC(=O)[O-].[Zn][O]([Zn])([Zn])[Zn]&&pcu.cat0"
    prediction = predictor.predict(test_mofid)
    prediction_norm = predictor.predict(test_mofid, return_normalized=True)
    print(f"输入MOFid: {test_mofid[:80]}...")
    print(f"预测CO2吸附量（真实值）: {prediction:.4f} mol/kg")
    print(f"预测CO2吸附量（归一化）: {prediction_norm:.4f}")
    print(f"数据集中真实值（用于对比）: 2.86059 mol/kg")
    
    # 3. 测试批量样本
    print("\n【测试2】批量MOFid预测：")
    test_mofids = [
        "N=NC12C3(C#CC(=O)[O-])C4C2(C2(C1(C3C42N=N)N=N)C#CC(=O)[O-])N=N.N=NC12C3C4C2(C2C1C3(C42)C#CC(=O)[O-])C#CC(=O)[O-].N=Nc1cc2[CH]N=C(C3=NNN(c(n1)c23)[NH])N=N.[Cu][Cu]&&pcu.cat0",
        "[O-]C(=O)C#CC#CC#CC(=O)[O-].[Zn][O]([Zn])([Zn])[Zn]&&pcu.cat1",
        "[O-]C(=O)C#CC(=O)[O-].[O-]C(=O)C#Cc1ccc(c(c1)O)C#CC(=O)[O-].[O].[V]&&rna.cat0"
    ]
    
    # 真实标签（来自数据集）
    true_labels = [0.805715, 0.532732, 0.89358]
    
    predictions = predictor.predict_batch(test_mofids)
    predictions_norm = predictor.predict_batch(test_mofids, return_normalized=True)
    
    for i, (mofid, pred, pred_norm, true_val) in enumerate(zip(test_mofids, predictions, predictions_norm, true_labels)):
        print(f"\n样本 {i+1}:")
        print(f"  MOFid: {mofid[:60]}...")
        print(f"  真实值: {true_val:.4f} mol/kg")
        print(f"  预测值: {pred:.4f} mol/kg")
        print(f"  预测误差: {abs(pred - true_val):.4f} mol/kg")
        print(f"  归一化预测: {pred_norm:.4f}")
    
    print("\n" + "="*60)
    print("推理测试完成！")
    print("="*60)


if __name__ == "__main__":
    demo_inference()
