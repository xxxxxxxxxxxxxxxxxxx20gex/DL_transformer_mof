

from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerRegressor, Transformer, regressoionHead
from model.utils import *
from torch.utils.data import DataLoader
import time
import csv
import yaml
import shutil
import argparse
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_finetune_transformer import MOF_ID_Dataset

import warnings

warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    """将当前使用的配置文件备份到 checkpoint 目录，便于复现实验。"""
    checkpoints_path = Path(model_checkpoints_folder)
    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True, exist_ok=True)
        shutil.copy('./config_ft_transformer.yaml', str(checkpoints_path / 'config_ft_transformer.yaml'))


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

    def __init__(self, config, log_dir):
        self.logger = self._setup_logger(log_dir)
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=log_dir)

        self.random_seed = self.config['dataloader']['randomSeed']

        # self.mofdata = np.load(self.config['dataset']['dataPath'], allow_pickle=True)
        # 数据来源：CSV。默认第一列为文本（MOFid: "SMILES&&拓扑"），第二列为数值标签。
        with open(self.config['dataset']['dataPath']) as f:
            reader = csv.reader(f)
            self.mofdata = [row for row in reader]
        self.mofdata = np.array(self.mofdata)
        self.vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(self.vocab_path, model_max_length=512, padding_side='right')

        self.train_data, self.valid_data, self.test_data = split_data(
            self.mofdata, valid_ratio=self.config['dataloader']['valid_ratio'],
            test_ratio=self.config['dataloader']['test_ratio'],
            randomSeed=self.config['dataloader']['randomSeed']
        )

        # 将文本 MOFid 编码为 token id，并与数值标签打包为样本
        self.train_dataset = MOF_ID_Dataset(data=self.train_data, tokenizer=self.tokenizer)
        self.valid_dataset = MOF_ID_Dataset(data=self.valid_data, tokenizer=self.tokenizer)
        self.test_dataset = MOF_ID_Dataset(data=self.test_data, tokenizer=self.tokenizer)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'],
            drop_last=False,
            shuffle=True, pin_memory=False
        )

        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'],
            drop_last=False,
            shuffle=False, pin_memory=False
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'],
            drop_last=False,
            shuffle=False, pin_memory=False
        )

        # 回归损失：MSE（训练优化的目标）
        self.criterion = nn.MSELoss()

        # 用训练集标签统计均值/方差，训练时对目标做标准化，评估时再反标准化计算 MAE
        self.normalizer = Normalizer(torch.from_numpy(self.train_dataset.label))

    def _move_to_device(self, tensor):
        """根据配置将张量移动到目标设备（CPU/GPU）。"""
        return tensor.to(self.device, non_blocking=self.config['cuda'])

    def _prepare_batch(self, inputs, target):
        """对一个 batch 做设备迁移与标签标准化，返回可用于前向计算的张量对。"""
        input_var = self._move_to_device(inputs)
        target_normed = self.normalizer.norm(target)
        target_var = self._move_to_device(target_normed)
        return input_var, target_var

    def _setup_logger(self, log_dir):
        """配置logger：同时输出到控制台和文件"""
        logger = logging.getLogger('FineTune')
        logger.setLevel(logging.INFO)

        # 防止重复添加handler
        if logger.handlers:
            logger.handlers.clear()

        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        log_file = str(Path(log_dir) / 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _get_device(self):
        """选择计算设备：有 CUDA 且配置不是 CPU 时使用 GPU，否则使用 CPU。"""
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        self.logger.info(f"Running on device: {device}")
        return device

    def train(self):
        """主训练入口：组网、优化器设置、迭代训练并定期验证与保存最优权重。"""
        self.transformer = Transformer(**self.config['Transformer'])
        if self.config['cuda']:
            self.transformer = self.transformer.to(self.device)

        model_transformer = self._load_pre_trained_weights(self.transformer)
        model = TransformerRegressor(transformer=model_transformer, d_model=512).to(self.device)

        # Identify new layers for different learning rates
        layer_list = []
        for name, param in model.named_parameters():
            if 'fc_out' in name:
                self.logger.info(f"New layer: {name}")
                layer_list.append(name)

        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        if self.config['optim']['optimizer'] == 'SGD':
            # 注意：当前配置中未提供 momentum/lr 字段，若要使用 SGD 需在 YAML 中补齐对应键值
            optimizer = optim.SGD(
                [{'params': base_params, 'lr': self.config['optim']['lr'] * 0.2}, {'params': params}],
                self.config['optim']['init_lr'], momentum=self.config['optim']['momentum'],
                weight_decay=eval(self.config['optim']['weight_decay'])
            )
        elif self.config['optim']['optimizer'] == 'Adam':
            # 常用做法：微调时给新加入的回归头更大学习率，骨干较小学习率
            optimizer = optim.Adam(
                [{'params': base_params, 'lr': self.config['optim']['init_lr'] * 1}, {'params': params}],
                self.config['optim']['init_lr'] * 200, weight_decay=eval(self.config['optim']['weight_decay'])
            )
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')

        model_checkpoints_folder = str(Path(self.writer.log_dir) / 'checkpoints')
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_mae = np.inf
        best_valid_loss = np.inf
        best_valid_roc_auc = 0

        model.train()

        for epoch_counter in range(self.config['epochs']):
            for bn, (inputs, target) in enumerate(self.train_loader):
                input_var, target_var = self._prepare_batch(inputs, target)

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                if bn % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    self.logger.info(
                        f'Epoch {epoch_counter + 1:3d} - Training Progress: [{bn + 1:3d}/{len(self.train_loader):3d}] | Loss: {loss.item():.4f}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

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
        """按名称与形状匹配地加载预训练权重；若找不到则从头训练。"""
        try:
            checkpoints_folder = self.config['fine_tune_from']
            load_state = torch.load(str(Path(checkpoints_folder) / 'model_transformer_14.pth'),
                                    map_location=self.device)
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
                    self.logger.info(f'Parameter not loaded: {name}')

            self.logger.info(f"Loaded {loaded_count} pre-trained parameters successfully.")
        except FileNotFoundError:
            self.logger.info("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_epoch):
        """验证阶段：不回传梯度，记录当前损失与 MAE，并打印平均值。"""
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

            self.logger.info(f'Epoch {n_epoch + 1:3d} - Validation Progress: [{bn + 1:3d}/{len(valid_loader):3d}] | '
                             f'Loss: {losses.val:.4f} (avg: {losses.avg:.4f}) | '
                             f'MAE: {mae_errors.val:.3f} (avg: {mae_errors.avg:.3f})')

        model.train()
        self.logger.info(f'Validation Complete - Final MAE: {mae_errors.avg:.3f}')
        return losses.avg, mae_errors.avg

    def test(self):
        """测试阶段：加载保存的最佳权重，对测试集评估并保存预测与真值。"""
        self.logger.info('Test Phase - Starting test on test set')
        model_path = str(Path(self.writer.log_dir) / 'checkpoints' / 'model.pth')
        self.logger.info(f'Model Path: {model_path}')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.logger.info("Loaded trained model successfully")

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

            self.logger.info(f'Test Progress: [{bn + 1:3d}/{len(self.test_loader):3d}] | '
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
    parser.add_argument('--seed', default=1, type=int, metavar='Seed',
                        help='random seed for splitting data (default: 1)')
    args = parser.parse_args(sys.argv[1:])

    # 配置基础logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger('Main')

    config = yaml.load(open("config_ft_transformer.yaml", "r"), Loader=yaml.FullLoader)
    logger.info(f"Loaded configuration: {config}")
    config['dataloader']['randomSeed'] = args.seed

    # Determine task name and pressure
    if 'hMOF' in config['dataset']['data_name']:
        task_name = config['dataset']['data_name']
        pressure = config['dataset']['data_name'].split('_')[-1]
    elif 'QMOF' in config['dataset']['data_name']:
        task_name = 'QMOF'
    else:
        task_name = config['dataset']['data_name']

    # Determine fine-tuning source and pre-training method
    if config['fine_tune_from'] == 'scratch':
        ftf = 'scratch'
        ptw = 'scratch'
    else:
        ftf = config['fine_tune_from'].split('/')[-1]
        ptw = config['trained_with']

    seed = config['dataloader']['randomSeed']
    log_dir = Path(
        'training_results/finetuning/Transformer') / f'Trans_{ptw}_{task_name}_{seed}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = str(log_dir)  # 转换为字符串以保持与SummaryWriter的兼容性

    fine_tune = FineTune(config, log_dir)
    fine_tune.train()
    loss, metric = fine_tune.test()

    # Save results
    fn = f'Trans_{ptw}_{task_name}_{seed}.csv'
    logger.info(f"Saving results to: {fn}")
    df = pd.DataFrame([[loss, metric.item()]])
    df.to_csv(
        str(Path(log_dir) / fn),
        mode='a', index=False, header=False
    )