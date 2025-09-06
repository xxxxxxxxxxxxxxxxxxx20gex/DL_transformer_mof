
import os
from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerPretrain
from model.utils import *
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_multiview import CIFData,collate_pool,get_train_val_test_loader
from datetime import datetime
from loss.barlow_twins import BarlowTwinsLoss
import yaml
from model.cgcnn_pretrain import CrystalGraphConvNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
import warnings
warnings.simplefilter("ignore")


def setup_logger():
    """
    使用basicConfig设置日志配置，支持文件保存和控制台输出
    """
    # 创建logs目录
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 获取模块名
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # 日志文件名格式：模块名_日期.log
    today = datetime.now().strftime('%Y-%m-%d')
    log_filename = os.path.join(log_dir, f'{module_name}_{today}.log')
    
    # 使用basicConfig设置日志配置（只配置一次）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 控制台输出
        ],
        force=True  # 强制重新配置，避免重复配置问题
    )
    
    # 获取logger实例
    logger = logging.getLogger(module_name)
    return logger


# 模块级别的logger实例，避免重复创建
logger = setup_logger()


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_multiview.yaml', os.path.join(model_checkpoints_folder, 'config_multiview.yaml'))


class Multiview(object):
    """
    多视图自监督学习训练器
    
    架构说明：
    - 使用Transformer处理序列数据，CGCNN处理图数据
    - 通过Barlow Twins损失函数实现多视图对比学习
    - 支持预训练权重加载和模型检查点保存
    """
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 包含所有训练配置的字典
        """
        self.config = config
        self.device = self._get_device()
        
        # 设置TensorBoard日志
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time
        log_dir = os.path.join('runs_multiview', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 初始化Barlow Twins损失函数
        self.dual_criterion = BarlowTwinsLoss(self.device, config['batch_size'], **config['barlow_loss'])
        
        # 初始化分词器和数据集
        self.vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(self.vocab_path, model_max_length = 512, padding_side='right')
        self.dataset = CIFData(**self.config['graph_dataset'], tokenizer = self.tokenizer)
 
        # 设置数据加载器
        logger.info("开始创建数据加载器...")
        collate_fn = collate_pool
        self.train_loader, self.valid_loader = get_train_val_test_loader(
            dataset=self.dataset,
            collate_fn=collate_fn,
            pin_memory=self.config['gpu'],
            batch_size=self.config['batch_size'], 
            **self.config['dataloader']
        )
        logger.info(f"数据加载器创建完成 - 训练集: {len(self.train_loader)} batches, 验证集: {len(self.valid_loader)} batches")

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
        单步训练：计算Barlow Twins损失
        """
        # 获取图数据的表示
        zjs = graph_model(*graph_data)  # [N,C]
        
        # 获取Transformer数据的表示
        zis = transformer_model(transformer_data)

        # 计算Barlow Twins损失
        loss_barlow = self.dual_criterion(zis, zjs)
        return loss_barlow

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

        # 加载预训练权重
        transformer_model, graph_model = self._load_pre_trained_weights(transformer_model, graph_model)

        # 设置优化器和调度器
        optimizer = torch.optim.Adam(list(transformer_model.parameters()) + list(graph_model.parameters()), 
                                   lr = self.config['optim']['init_lr'], 
                                   weight_decay=eval(self.config['optim']['weight_decay']))
        scheduler = CosineAnnealingLR(optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1)

        # 设置检查点目录
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)

        # 训练循环变量
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            logger.info(f"开始第 {epoch_counter} 个epoch的训练...")
            for bn, (graph_data, transformer_data, _) in enumerate(self.train_loader):
                # 移动数据到设备
                input_graph, input_transformer = self._move_data_to_device(graph_data, transformer_data)
                
                # 前向传播计算损失
                loss = self._step(transformer_model, graph_model, input_transformer, input_graph)

                # 记录训练日志
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    logger.info(f"Epoch {epoch_counter}, Batch {bn}, Loss: {loss.item():.6f}")
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

            torch.cuda.empty_cache()

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
            checkpoints_folder = os.path.join('./runs_multiview', self.config['fine_tune_from'], 'checkpoints')
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
        with torch.no_grad():
            transformer_model.eval()
            graph_model.eval()

            loss_total = 0.0
            total_num = 0
            for graph_data, transformer_data, batch_cif_ids in valid_loader:
                # 移动数据到设备
                input_graph, input_transformer = self._move_data_to_device(graph_data, transformer_data)
                
                # 计算验证损失
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
    logger.info(f"Configuration loaded: {config}")

    mof_multiview = Multiview(config)
    mof_multiview.train()
