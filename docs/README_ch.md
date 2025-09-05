# MOFormer

<strong>MOFormer：用于金属有机框架性质预测的自监督Transformer模型</strong> </br>
<em>美国化学会志(JACS)</em> https://pubs.acs.org/doi/10.1021/jacs.2c11420 https://arxiv.org/abs/2210.14188 https://arxiv.org/pdf/2210.14188.pdf </br>
https://www.linkedin.com/in/zhonglincao/?trk=public_profile_browsemap, https://www.linkedin.com/in/rishikesh-magar, https://yuyangw.github.io/, https://www.meche.engineering.cmu.edu/directory/bios/barati-farimani-amir.html (*同等贡献) </br>
卡内基梅隆大学 </br>

<img src="figs/pipeline.png" width="600">

这是论文https://pubs.acs.org/doi/10.1021/jacs.2c11420的官方实现。在这项工作中，我们提出了一种基于Transformer模型的与结构无关的深度学习方法，命名为<strong><em>MOFormer</em></strong>，用于MOF的性质预测。<strong><em>MOFormer</em></strong>以MOF的文本字符串表示(MOFid)作为输入，从而避免了获取假设MOF的3D结构的需求，并加速了筛选过程。此外，我们引入了一个自监督学习框架，通过在超过40万个公开可用的MOF数据上最大化其与结构无关的表示和基于结构的晶体图卷积神经网络(CGCNN)表示之间的互相关性来预训练<strong><em>MOFormer</em></strong>。基准测试表明，预训练提高了两种模型在各种下游预测任务上的预测准确性。如果您在我们的研究中发现我们的工作有用，请引用：

```
@article{doi:10.1021/jacs.2c11420,
    author = {Cao, Zhonglin and Magar, Rishikesh and Wang, Yuyang and Barati Farimani, Amir},
    title = {MOFormer: Self-Supervised Transformer Model for Metal–Organic Framework Property Prediction},
    journal = {Journal of the American Chemical Society},
    volume = {145},
    number = {5},
    pages = {2958-2967},
    year = {2023},
    doi = {10.1021/jacs.2c11420},
    URL = {https://doi.org/10.1021/jacs.2c11420}
}
```


## 开始使用

### 安装

设置conda环境并克隆github仓库

```
# 创建新环境
$ conda create -n myenv python=3.9
$ conda activate moformer
$ conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
$ conda install --channel conda-forge pymatgen
$ pip install transformers
$ conda install -c conda-forge tensorboard

# 克隆MOFormer源代码
$ git clone https://github.com/zcao0420/MOFormer
$ cd MOFormer
```

### 数据集

本工作中使用的所有数据都可以在`benchmark_datasets`文件夹中找到。如果您使用本工作中的任何数据，请引用致谢部分中包含的相应参考文献。

### 检查点

预训练模型可以在`ckpt`文件夹中找到。

## 运行模型

### 预训练

要从头开始使用SSL预训练模型，可以运行`python pretrain_SSL.py`。预训练的配置文件以cif文件的目录和一个名为`id_prop.npy`的文件作为输入。`id_prop.npy`包含`cif id`及其对应的`mof id`字符串表示。我们已经添加了一个名为`cif_toy`的文件夹，其中包含100个MOF的cif文件和`cif_toy`文件夹中数据对应的`id_prop.npy`。如果您打算为`cif_toy`文件夹运行预训练，请确保更新`config_multiview.yaml`，指示根目录的正确位置。预训练数据集可在https://figshare.com/articles/journal_contribution/cif_tar_xz/23532918上找到。
```
python pretrain_SSL.py
```

### 微调

要微调预训练的Transformer，可以运行`finetune_transformer.py`，其中配置在`config_ft_transformer.yaml`中定义。
```
python finetune_transformer.py
```
类似地，要微调预训练的CGCNN，可以运行`finetune_cgcnn.py`，其中配置在`config_ft_cgcnn.yaml`中定义。
```
python finetune_cgcnn.py
```

我们还提供了一个jupyter笔记本`demo.ipynb`用于微调/监督训练。

## 致谢
- CGCNN: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301 和 https://github.com/txie-93/cgcnn
- Barlow Twins: https://arxiv.org/abs/2103.03230 和 https://github.com/facebookresearch/barlowtwins
- Crystal Twins: https://www.nature.com/articles/s41524-022-00921-5 和 https://github.com/RishikeshMagar/Crystal-Twins
- MOFid: https://pubs.acs.org/doi/full/10.1021/acs.cgd.9b01050 和 https://github.com/snurr-group/mofid/tree/master
- Boyd&Woo数据集 https://www.nature.com/articles/s41586-019-1798-7
- QMOF https://www.cell.com/matter/fulltext/S2590-2385(2100070-9?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2590238521000709%3Fshowall%3Dtrue) 和 https://www.nature.com/articles/s41524-022-00796-6
- hMOF https://www.nature.com/articles/nchem.1192

#### 关于代码的问题
参与论文工作的研究生已从CMU毕业。我们定期监控github仓库，如有关于代码的问题或疑虑，请随时提出github issues。这使我们更容易处理代码请求。
