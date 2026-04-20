
import os
import time
import logging
import shutil
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerPretrain
from model.utils import *
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_multiview import build_multiview_dataset, collate_pool, get_train_val_test_loader
from datetime import datetime
from loss.clip_loss import ClipContrastiveLoss
import yaml
from model.cgcnn_pretrain import CrystalGraphConvNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
import warnings
warnings.simplefilter("ignore")

_REPO_ROOT = Path(__file__).resolve().parent


def _resolve_ssl_exp_root(config: Optional[Dict[str, Any]]) -> str:
    """统一实验根目录，默认 <repo>/exp/SSL；可在 config_multiview.yaml 中覆盖 ssl_exp_root。"""
    raw = (config or {}).get("ssl_exp_root")
    if not raw:
        return str(_REPO_ROOT / "exp" / "SSL")
    p = Path(raw)
    return str(p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve())


def setup_logger(ssl_exp_root: str) -> logging.Logger:
    """日志写入 <ssl_exp_root>/logs/，并同时输出到控制台。"""
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join(ssl_exp_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join(log_dir, f"{module_name}_{today}.log")

    lg = logging.getLogger(module_name)
    lg.handlers.clear()
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(sh)
    lg.propagate = False
    return lg


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_multiview.yaml', os.path.join(model_checkpoints_folder, 'config_multiview.yaml'))


class Multiview(object):
    """
    多视图自监督学习训练器
    
    架构说明：
    - 使用Transformer处理序列数据，CGCNN处理图数据
    - 通过 CLIP 式对称 InfoNCE 对齐两路 embedding（同一样本为正，batch 内其它为负）
    - 支持预训练权重加载和模型检查点保存
    """
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 包含所有训练配置的字典
        """
        self.config = config
        self.ssl_exp_root = _resolve_ssl_exp_root(config)
        self.device = self._get_device()
        
        # 启用cuDNN benchmark优化（适用于固定输入尺寸的网络）
        if self.config.get('cuda', False):
            cudnn.benchmark = True
            logger.info("已启用cuDNN benchmark优化")
        
        # TensorBoard 与 checkpoints 根目录：<ssl_exp_root>/runs_multiview/<时间戳>/
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time
        log_dir = os.path.join(self.ssl_exp_root, 'runs_multiview', dir_name)
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"SSL 实验输出目录: {self.ssl_exp_root} (TensorBoard/checkpoints: {log_dir})")
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.dual_criterion = ClipContrastiveLoss(**config['clip_loss'])
        
        # 初始化分词器和数据集
        self.vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(self.vocab_path, model_max_length = 512, padding_side='right')
        self.dataset = build_multiview_dataset(**self.config['graph_dataset'], tokenizer=self.tokenizer)
        logger.info(f"训练数据集后端: {self.dataset.__class__.__name__}")
        if getattr(self.dataset, 'uses_graph_cache', False):
            logger.info(f"图缓存目录: {self.dataset.cache_dir}")
 
        # 设置数据加载器
        logger.info("开始创建数据加载器...")
        collate_fn = collate_pool
        self.train_loader, self.valid_loader = get_train_val_test_loader(
            dataset=self.dataset,
            collate_fn=collate_fn,
            pin_memory=self.config.get('pin_memory', self.config['gpu']),
            batch_size=self.config['batch_size'], 
            **self.config['dataloader']
        )
        logger.info(f"数据加载器创建完成 - 训练集: {len(self.train_loader)} batches, 验证集: {len(self.valid_loader)} batches")
        
        # 打印设备信息
        logger.info(f"CUDA available: {torch.cuda.is_available()}, device setting: {self.device}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            logger.info(f"Current GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def _get_device(self):
        """设置训练设备（GPU/CPU）"""
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            self.config['cuda'] = True
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
            self.config['cuda'] = False
        logger.info(f"Running on: {device}")
        return device

    def _move_data_to_device(self, graph_data, transformer_data):
        """
        将数据移动到指定设备
        
        Args:
            graph_data: 图数据 (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            transformer_data: Transformer输入数据
            
        Returns:
            tuple: 移动到设备后的图数据和Transformer数据
        """
        if self.config['cuda']:
            input_graph = (Variable(graph_data[0].to(self.device, non_blocking=True)),
                        Variable(graph_data[1].to(self.device, non_blocking=True)),
                        graph_data[2].to(self.device, non_blocking=True),
                        [crys_idx.to(self.device, non_blocking=True) for crys_idx in graph_data[3]])
            input_transformer = transformer_data.to(self.device, non_blocking = True)
        else:
            input_graph = (Variable(graph_data[0]),
                        Variable(graph_data[1]),
                        graph_data[2],
                        graph_data[3])
            input_transformer = transformer_data
        return input_graph, input_transformer

    def _step(self, transformer_model, graph_model, transformer_data, graph_data, epsilon = 0):
        """
        单步训练：CLIP 式对称对比损失（Transformer 与 CGCNN 各为一路）。
        """
        z_graph = graph_model(*graph_data)
        z_text = transformer_model(transformer_data)
        return self.dual_criterion(z_text, z_graph)

    def train(self):
        """
        主训练循环
        
        训练流程：
        1. 初始化Transformer和CGCNN模型
        2. 加载预训练权重（如果存在）
        3. 设置优化器和学习率调度器
        4. 执行训练循环，包括验证和模型保存
        """
        # 获取数据特征维度
        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]

        # 初始化模型
        transformer_model = TransformerPretrain(**self.config["Transformer"]).to(self.device)
        graph_model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, **self.config['model_cgcnn']).to(self.device)

        # 打印模型设备信息
        logger.info(f"Transformer model device: {next(transformer_model.parameters()).device}")
        logger.info(f"Graph model device: {next(graph_model.parameters()).device}")

        # 加载预训练权重
        transformer_model, graph_model = self._load_pre_trained_weights(transformer_model, graph_model)

        # 设置优化器和调度器
        optimizer = torch.optim.Adam(list(transformer_model.parameters()) + list(graph_model.parameters()), 
                                   lr = self.config['optim']['init_lr'], 
                                   weight_decay=eval(self.config['optim']['weight_decay']))
        scheduler = CosineAnnealingLR(optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1)

        # 初始化AMP scaler
        use_amp = self.config.get('use_amp', True)
        scaler = GradScaler(enabled=use_amp and self.config.get('cuda', False))
        if use_amp and self.config.get('cuda', False):
            logger.info("已启用AMP混合精度训练")

        # 设置检查点目录
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)

        # 训练循环变量
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            logger.info(f"开始第 {epoch_counter} 个epoch的训练...")
            epoch_start_time = time.time()

            train_iterator = iter(self.train_loader)
            for bn in range(len(self.train_loader)):
                fetch_start_time = time.time()
                graph_data, transformer_data, _ = next(train_iterator)
                fetch_time = time.time() - fetch_start_time
                batch_start_time = time.time()
                
                # 移动数据到设备
                move_start_time = time.time()
                input_graph, input_transformer = self._move_data_to_device(graph_data, transformer_data)
                move_time = time.time() - move_start_time
                
                # 前向传播计算损失（使用AMP）
                optimizer.zero_grad()
                with autocast(enabled=use_amp and self.config.get('cuda', False)):
                    loss = self._step(transformer_model, graph_model, input_transformer, input_graph)

                # 反向传播（使用AMP scaler）
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                step_time = time.time() - batch_start_time

                # 记录训练日志
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    # 计算GPU利用率相关指标
                    gpu_util = 'n/a'
                    gpu_memory = 0
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        try:
                            gpu_util = f"{torch.cuda.utilization()}%"
                        except (ModuleNotFoundError, RuntimeError, AttributeError):
                            gpu_util = 'n/a'
                    logger.info(f"Epoch {epoch_counter}, Batch {bn}, Loss: {loss.item():.6f}, "
                              f"fetch_time={fetch_time:.3f}s, move_time={move_time:.3f}s, "
                              f"step_time={step_time:.3f}s, "
                              f"gpu_util={gpu_util}, gpu_mem={gpu_memory:.1f}GB")
                
                n_iter += 1

            torch.cuda.empty_cache()
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch_counter} 完成，耗时: {epoch_time:.2f}s")

            # 验证模型
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(transformer_model, graph_model, self.valid_loader)
                logger.info(f"Validation Loss: {valid_loss:.6f}")
                if valid_loss < best_valid_loss:
                    # 保存最佳模型（验证损失最低的模型）
                    best_valid_loss = valid_loss
                    torch.save(transformer_model.state_dict(), os.path.join(model_checkpoints_folder, 'best_transformer_model.pth'))
                    torch.save(graph_model.state_dict(), os.path.join(model_checkpoints_folder, 'best_graph_model.pth'))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # 定期保存模型检查点（用于恢复训练或分析）
            # 默认每5个epoch保存一次，可通过配置中的 save_every_n_epochs 参数调整
            if epoch_counter > 0 and epoch_counter % self.config.get('save_every_n_epochs', 5) == 0:
                torch.save(transformer_model.state_dict(), os.path.join(model_checkpoints_folder, f'model_transformer_epoch_{epoch_counter}.pth'))
                torch.save(graph_model.state_dict(), os.path.join(model_checkpoints_folder, f'model_graph_epoch_{epoch_counter}.pth'))
            
            # 学习率调度（前5个epoch为warmup）
            if epoch_counter >= 5:
                scheduler.step()
    
    def _load_pre_trained_weights(self, transformer_model, graph_model):
        """
        加载预训练权重
        
        Args:
            transformer_model: Transformer模型
            graph_model: CGCNN模型
            
        Returns:
            tuple: 加载权重后的模型
        """
        try:
            ftf = self.config.get('fine_tune_from')
            if ftf in (None, '', 'None'):
                logger.info("fine_tune_from 未设置，从头训练。")
                return transformer_model, graph_model
            checkpoints_folder = os.path.join(
                self.ssl_exp_root, 'runs_multiview', str(ftf), 'checkpoints'
            )
            state_dict_t = torch.load(os.path.join(checkpoints_folder, 'model_transformer_11.pth'), map_location=self.config['gpu'])
            transformer_model.load_state_dict(state_dict_t)

            state_dict_g = torch.load(os.path.join(checkpoints_folder, 'model_graph_11.pth'), map_location = self.config['gpu'])
            graph_model.load_state_dict(state_dict_g)

            logger.info("Loaded pre-trained model with success.")
            
        except FileNotFoundError:
            logger.info("Pre-trained weights not found. Training from scratch.")

        return transformer_model, graph_model

    def _validate(self, transformer_model, graph_model, valid_loader):
        """
        验证模型性能
        
        Args:
            transformer_model: Transformer模型
            graph_model: CGCNN模型
            valid_loader: 验证数据加载器
            
        Returns:
            float: 平均验证损失
        """
        use_amp = self.config.get('use_amp', True)
        
        with torch.no_grad():
            transformer_model.eval()
            graph_model.eval()

            loss_total = 0.0
            total_num = 0
            for graph_data, transformer_data, batch_cif_ids in valid_loader:
                # 移动数据到设备
                input_graph, input_transformer = self._move_data_to_device(graph_data, transformer_data)
                
                # 计算验证损失（使用AMP）
                with autocast(enabled=use_amp and self.config.get('cuda', False)):
                    loss = self._step(transformer_model, graph_model, input_transformer, input_graph)
                loss_total += loss.item() * len(batch_cif_ids)
                total_num += len(batch_cif_ids)
                
            loss_total /= total_num
        torch.cuda.empty_cache()
        transformer_model.train()
        graph_model.train()
        return loss_total


if __name__ == "__main__":
    config = yaml.load(open("config_multiview.yaml", "r"), Loader=yaml.FullLoader)
    ssl_root = _resolve_ssl_exp_root(config)
    setup_logger(ssl_root)
    logger.info(f"Configuration loaded: {config}")

    mof_multiview = Multiview(config)
    mof_multiview.train()
