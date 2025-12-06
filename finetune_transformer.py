
from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerRegressor, Transformer
from model.utils import *
from torch.utils.data import DataLoader
import time
import csv
import yaml
import argparse
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_finetune_transformer import MOF_ID_Dataset

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _parse_task_name(data_name: str) -> str:
    """解析数据集名称，提取任务名称"""
    if 'hMOF' in data_name:
        return data_name  # 保留完整的 hMOF 名称包括压力信息
    elif 'QMOF' in data_name:
        return 'QMOF'
    else:
        return data_name


def _parse_finetuning_info(config) -> Tuple[str, str]:
    """解析微调源和预训练方法信息"""
    if config['fine_tune_from'] == 'scratch':
        return 'scratch', 'scratch'
    else:
        ftf = config['fine_tune_from'].split('/')[-1]
        ptw = config['trained_with']
        return ftf, ptw

def _create_log_directory(ptw: str, task_name: str, seed: int) -> str:
    """创建训练日志目录"""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path('training_results/finetuning/Transformer') / f'Trans_{ptw}_{task_name}_{seed}_{timestamp}'
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


class FineTune(object):
    """
    Transformer 微调训练器

    主要职责：
    - 解析配置、选择设备、创建日志记录器
    - 读取 CSV 数据，切分 train/valid/test，并构建 DataLoader
    - 构建 Transformer 主干与回归头，按需加载预训练权重
    - 训练循环：前向 -> 计算损失 -> 反向传播 -> 参数更新
    - 验证与测试：记录损失与 MAE，并保存最佳模型
    """
    def __init__(self, config, log_dir: str, logger):
        self.logger = logger
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=log_dir)

        self.random_seed = self.config['dataloader']['randomSeed']

        # 数据来源：CSV。默认第一列为文本(MOFid: "SMILES&&拓扑")，第二列为数值标签。
        self.mofdata = self._load_csv_data(self.config['dataset']['dataPath'])
        self.vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(self.vocab_path, model_max_length = 512, padding_side='right')

        self.train_data, self.valid_data, self.test_data = split_data(
            self.mofdata, 
            test_ratio = self.config['dataloader']['test_ratio'], 
            valid_ratio = self.config['dataloader']['valid_ratio'], 
            use_ratio = self.config['dataloader']['use_ratio'],
            randomSeed = self.config['dataloader']['randomSeed']
        )
        
        # 将文本 MOFid 编码为 token id，并与数值标签打包为样本
        self.train_dataset = MOF_ID_Dataset(data = self.train_data, tokenizer = self.tokenizer)
        self.valid_dataset = MOF_ID_Dataset(data = self.valid_data, tokenizer = self.tokenizer)
        self.test_dataset = MOF_ID_Dataset(data = self.test_data, tokenizer = self.tokenizer)
        
        # 使用工厂函数创建DataLoader，减少重复代码
        self.train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        self.valid_loader = self._create_dataloader(self.valid_dataset, shuffle=False)
        self.test_loader = self._create_dataloader(self.test_dataset, shuffle=False)

        # 回归损失：MSE(训练优化的目标)
        self.criterion = nn.MSELoss()

        # 用训练集标签统计均值/方差，训练时对目标做标准化，评估时再反标准化计算 MAE
        self.normalizer = Normalizer(torch.tensor(self.train_dataset.label, dtype=torch.float32))

    def _load_csv_data(self, data_path: str) -> np.ndarray:
        """加载CSV数据文件"""
        try:
            with open(data_path) as f:
                reader = csv.reader(f)
                data = [row for row in reader]
            return np.array(data)
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {data_path}")
            raise

    def _create_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        """创建DataLoader的工厂函数，减少重复代码"""
        return DataLoader(
            dataset, 
            batch_size=self.config['batch_size'], 
            num_workers=self.config['num_workers'], 
            drop_last=False, 
            shuffle=shuffle, 
            pin_memory=False
        )

    def _move_to_device(self, tensor) -> torch.Tensor:
        """根据配置将张量移动到目标设备(CPU/GPU)"""
        return tensor.to(self.device, non_blocking=self.config['cuda'])

    def _prepare_batch(self, inputs, target) -> Tuple[torch.Tensor, torch.Tensor]:
        """对一个 batch 做设备迁移与标签标准化，返回可用于前向计算的张量对"""
        input_var = self._move_to_device(inputs)
        target_normed = self.normalizer.norm(target)
        target_var = self._move_to_device(target_normed)
        return input_var, target_var

    def _separate_model_parameters(self, model, new_layer_identifier):
        """分离模型参数为新层参数和基础参数，用于差异化学习率设置"""
        new_layer_params = []
        base_params = []
        
        for name, param in model.named_parameters():
            # 识别回归头和其他新添加的层
            if new_layer_identifier in name or 'regressionhead' in name.lower():
                self.logger.info(f"New layer: {name}")
                new_layer_params.append(param)
            else:
                base_params.append(param)
        
        return new_layer_params, base_params

    def _create_optimizer(self, new_layer_params, base_params):
        """创建优化器，为新层和基础层设置不同的学习率"""
        base_lr = self.config['optim']['init_lr']
        base_multiplier = self.config['optim'].get('base_layer_lr_multiplier', 1)
        new_multiplier = self.config['optim'].get('new_layer_lr_multiplier', 200)
        weight_decay = eval(self.config['optim']['weight_decay'])
        
        param_groups = [
            {'params': base_params, 'lr': base_lr * base_multiplier}, 
            {'params': new_layer_params, 'lr': base_lr * new_multiplier}
        ]
        
        optimizer_name = self.config['optim']['optimizer']
        if optimizer_name == 'SGD':
            return optim.SGD(param_groups, momentum=self.config['optim'].get('momentum', 0.9), weight_decay=weight_decay)
        elif optimizer_name == 'Adam':
            return optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            raise ValueError(f'Only SGD or Adam is allowed as optimizer, got: {optimizer_name}')

    def _get_device(self):
        """选择计算设备：有 CUDA 且配置不是 CPU 时使用 GPU，否则使用 CPU"""
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        self.logger.info(f"Running on device: {device}")
        return device

    def _train_epoch(self, model, optimizer, epoch: int) -> int:
        """训练一个epoch，返回迭代次数"""
        model.train()
        n_iter = 0
        
        for bn, (inputs, target) in enumerate(self.train_loader):
            input_var, target_var = self._prepare_batch(inputs, target)

            # compute output
            output = model(input_var)
            loss = self.criterion(output, target_var)

            if bn % self.config['log_every_n_steps'] == 0:
                self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                self.logger.info(f'Epoch {epoch+1:3d} - Training Progress: [{bn+1:3d}/{len(self.train_loader):3d}] | Loss: {loss.item():.4f}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter += 1
        
        return n_iter

    def train(self):
        """主训练入口：组网、优化器设置、迭代训练并定期验证与保存最优权重"""
        # 构建模型
        self.transformer = Transformer(**self.config['Transformer'])
        if self.config['cuda']:
            self.transformer = self.transformer.to(self.device)

        model_transformer = self._load_pre_trained_weights(self.transformer)
        
        # 构建微调模型（Transformer + 回归头）
        model = TransformerRegressor(
            transformer=model_transformer, 
            d_model=self.config['Transformer']['d_model']
        ).to(self.device)
        
        # 分离新层参数和基础参数，用于差异化学习率
        new_layer_params, base_params = self._separate_model_parameters(model, 'fc_out')

        # 创建优化器，为不同层设置不同的学习率
        optimizer = self._create_optimizer(new_layer_params, base_params)
        
        model_checkpoints_folder = str(Path(self.writer.log_dir) / 'checkpoints')
        save_config_file(model_checkpoints_folder, './config_ft_transformer.yaml')

        n_iter = 0
        valid_n_iter = 0
        best_valid_mae = np.inf

        # 训练循环
        for epoch_counter in range(self.config['epochs']):
            n_iter = self._train_epoch(model, optimizer, epoch_counter)

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss, valid_mae = self._validate(model, self.valid_loader, epoch_counter)
                if valid_mae < best_valid_mae:
                    # save the model weights
                    best_valid_mae = valid_mae
                    torch.save(model.state_dict(), str(Path(model_checkpoints_folder) / 'model.pth'))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
        self.model = model
           
    def _load_pre_trained_weights(self, model):
        """按名称与形状匹配地加载预训练权重；若找不到则从头训练"""
        try:
            checkpoints_folder = self.config['fine_tune_from']
            model_file = self.config.get('pretrained_model_file', 'model_transformer_14.pth')
            load_state = torch.load(str(Path(checkpoints_folder) / model_file), map_location=self.device) 
            model_state = model.state_dict()

            # Load only matching parameters
            loaded_count = 0
            for name, param in load_state.items():
                if name in model_state:
                    if isinstance(param, nn.parameter.Parameter):
                        param = param.data
                    model_state[name].copy_(param)
                    loaded_count += 1
                else:
                    self.logger.debug(f'Parameter not loaded: {name}')
            
            self.logger.info(f"Loaded {loaded_count} pre-trained parameters successfully.")
        except FileNotFoundError:
            self.logger.info("Pre-trained weights not found. Training from scratch.")
        except Exception as e:
            self.logger.warning(f"Error loading pre-trained weights: {e}. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_epoch: int) -> Tuple[float, float]:
        """验证阶段：不回传梯度，记录当前损失与 MAE，并打印平均值"""
        losses = AverageMeter()
        mae_errors = AverageMeter()

        with torch.no_grad():
            model.eval()
            for bn, (inputs, target) in enumerate(valid_loader):
                input_var, target_var = self._prepare_batch(inputs, target)

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

            self.logger.info(f'Epoch {n_epoch+1:3d} - Validation Progress: [{bn+1:3d}/{len(valid_loader):3d}] | '
                  f'Loss: {losses.val:.4f} (avg: {losses.avg:.4f}) | '
                  f'MAE: {mae_errors.val:.3f} (avg: {mae_errors.avg:.3f})')
        
        model.train()
        self.logger.info(f'Validation Complete - Final MAE: {mae_errors.avg:.3f}')
        return losses.avg, mae_errors.avg

    def test(self) -> Tuple[float, float]:
        """测试阶段：加载保存的最佳权重，对测试集评估并保存预测与真值"""
        self.logger.info('Test Phase - Starting test on test set')
        model_path = str(Path(self.writer.log_dir) / 'checkpoints' / 'model.pth')
        self.logger.info(f'Model Path: {model_path}')
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.logger.info("Loaded trained model successfully")
        except FileNotFoundError:
            self.logger.error(f"Model file not found: {model_path}")
            raise

        losses = AverageMeter()
        mae_errors = AverageMeter()
        
        test_targets = []
        test_preds = []

        with torch.no_grad():
            self.model.eval()
            for bn, (inputs, target) in enumerate(self.test_loader):
                input_var, target_var = self._prepare_batch(inputs, target)

                # compute output
                output = self.model(input_var)
                loss = self.criterion(output, target_var)

                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                
                test_pred = self.normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()

            self.logger.info(f'Test Progress: [{bn+1:3d}/{len(self.test_loader):3d}] | '
                  f'Loss: {losses.val:.4f} (avg: {losses.avg:.4f}) | '
                  f'MAE: {mae_errors.val:.3f} (avg: {mae_errors.avg:.3f})')

        # Save test results
        with open(str(Path(self.writer.log_dir) / 'test_results.csv'), 'w') as f:
            writer = csv.writer(f)
            for target, pred in zip(test_targets, test_preds):
                writer.writerow((target, pred))
        
        self.model.train()
        self.logger.info(f'Test Complete - Final MAE: {mae_errors.avg:.3f}')
        return losses.avg, mae_errors.avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer finetuning')
    parser.add_argument('--seed', default=1, type=int, metavar='Seed', help='random seed for splitting data (default: 1)')
    args = parser.parse_args(sys.argv[1:])
    
    config = yaml.load(open("config_ft_transformer.yaml", "r"), Loader=yaml.FullLoader)
    config['dataloader']['randomSeed'] = args.seed

    # 解析任务名称
    task_name = _parse_task_name(config['dataset']['data_name'])

    # 解析微调源和预训练方法
    ftf, ptw = _parse_finetuning_info(config)

    # 创建日志目录
    log_dir = _create_log_directory(ptw, task_name, config['dataloader']['randomSeed'])
    log_file = str(Path(log_dir) / 'training.log')
    
    # 配置logger：同时输出到控制台和文件
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),        # 控制台输出
            logging.FileHandler(log_file)   # 文件输出
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded configuration: {config}")

    fine_tune = FineTune(config, log_dir, logger)
    fine_tune.train()
    loss, metric = fine_tune.test()

    # Save results
    seed = config['dataloader']['randomSeed']
    fn = f'Trans_{ptw}_{task_name}_{seed}.csv'
    logger.info(f"Saving results to: {fn}")
    df = pd.DataFrame([[loss, metric.item()]])
    df.to_csv(
        str(Path(log_dir) / fn),
        mode='a', index=False, header=False
    )