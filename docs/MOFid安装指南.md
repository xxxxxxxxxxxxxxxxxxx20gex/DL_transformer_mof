# MOFid 安装和使用指南

## 概述

MOFid是一个用于快速识别和分析金属有机框架(MOF)的系统。本指南将帮助您在Linux环境下安装和配置MOFid，并生成MOFormer所需的id_prop.npy文件。

## 系统要求

### 必需软件
- **C++编译器**: gcc/g++ (推荐gcc-11)
- **CMake**: 3.0或更高版本
- **Make**: GNU make
- **Java运行时环境**: OpenJDK 11或更高版本
- **Python**: 3.6或更高版本

### 检查依赖
```bash
# 检查C++编译器
which gcc
which g++

# 检查CMake
which cmake

# 检查Make
which make

# 检查Java
which java
```

## 安装步骤

### 1. 安装Java运行时环境
如果Java未安装，请执行：
```bash
apt update
apt install -y openjdk-11-jre
```

### 2. 编译MOFid
```bash
# 进入MOFid目录
cd mofid

# 编译项目（如果内存不足，使用-j1减少并行进程）
make init
```

### 3. 生成路径配置文件
```bash
# 生成paths.py文件
python set_paths.py
```

### 4. 修复Python模块导入问题
由于MOFid的Python模块结构，需要修改导入语句以支持相对导入：

#### 修改 mofid/Python/run_mofid.py
```python
# 将
from mofid.id_constructor import (extract_fragments, extract_topology,
    assemble_mofkey, assemble_mofid, parse_mofid)
from mofid.cpp_cheminformatics import openbabel_GetSpacedFormula

# 改为
from .id_constructor import (extract_fragments, extract_topology,
    assemble_mofkey, assemble_mofid, parse_mofid)
from .cpp_cheminformatics import openbabel_GetSpacedFormula
```

#### 修改 mofid/Python/id_constructor.py
```python
# 将
from mofid.paths import resources_path, bin_path

# 改为
from .paths import resources_path, bin_path
```

## 使用MOFid生成id_prop.npy文件

### 创建生成脚本
创建 `generate_id_prop.py` 文件：

```python
import os
import sys
import numpy as np
import glob

# 添加mofid目录到Python路径，确保优先使用本地版本
mofid_path = os.path.join(os.path.dirname(__file__), 'mofid')
sys.path.insert(0, mofid_path)
from Python.run_mofid import cif2mofid

def generate_id_prop_npy(cif_dir, output_file='id_prop.npy'):
    """
    为CIF文件夹生成id_prop.npy文件
    
    Args:
        cif_dir: CIF文件目录
        output_file: 输出的npy文件名
    """
    cif_files = glob.glob(os.path.join(cif_dir, '*.cif'))
    print(f"找到 {len(cif_files)} 个CIF文件")
    
    id_prop_data = []
    
    for cif_file in cif_files:
        try:
            # 获取CIF文件名（不含扩展名）
            cif_id = os.path.splitext(os.path.basename(cif_file))[0]
            
            # 使用MOFid生成MOFid字符串
            mofid_result = cif2mofid(cif_file)
            mofid_string = mofid_result['mofid']
            
            # 添加到数据列表
            id_prop_data.append([cif_id, mofid_string])
            print(f"处理: {cif_id} -> {mofid_string[:50]}...")
            
        except Exception as e:
            print(f"处理 {cif_file} 时出错: {e}")
            continue

    # 保存为npy文件
    id_prop_array = np.array(id_prop_data, dtype=object)
    np.save(output_file, id_prop_array)
    print(f"成功生成 {output_file}，包含 {len(id_prop_data)} 个条目")
    
    return id_prop_array

if __name__ == "__main__":
    # 为cif_all/cif文件夹生成id_prop.npy
    cif_directory = "./cif_all/cif"  # CIF文件夹路径
    output_file = "./cif_all/cif/id_prop.npy"  # 输出文件路径

    print(f"开始处理目录: {cif_directory}")
    print(f"输出文件: {output_file}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 生成id_prop.npy文件
    result = generate_id_prop_npy(cif_directory, output_file)
    
    print(f"处理完成！共处理了 {len(result)} 个CIF文件")
```

### 运行生成脚本
```bash
# 运行脚本生成id_prop.npy文件
python generate_id_prop.py
```

## 输出文件格式

生成的 `id_prop.npy` 文件包含以下格式的数据：
- 每行包含两个元素：[CIF文件名, MOFid字符串]
- CIF文件名不包含扩展名
- MOFid字符串是MOF的唯一标识符

## 常见问题解决

### 2. 模块导入错误
如果出现 "No module named 'mofid.paths'" 错误：
- 确保已运行 `python set_paths.py`
- 检查是否已修改导入语句为相对导入

### 3. Java未找到
如果出现 "You must have Java in your path!" 错误：
```bash
apt install -y openjdk-11-jre
```

## 验证安装

运行以下命令验证安装是否成功：
```bash
# 测试单个CIF文件
python generate_id_prop.py
```

如果看到类似以下输出，说明安装成功：
```
开始处理目录: ./cif_all/cif
输出文件: ./cif_all/cif/id_prop.npy
找到 X 个CIF文件
处理: filename -> MOFid字符串...
成功生成 ./cif_all/cif/id_prop.npy，包含 X 个条目
处理完成！共处理了 X 个CIF文件
```

## 注意事项

1. **内存要求**: 编译过程需要足够的内存，建议至少4GB RAM
2. **磁盘空间**: 编译过程会生成大量临时文件，确保有足够的磁盘空间
3. **网络连接**: 首次编译可能需要下载依赖包
4. **权限**: 确保有足够的权限安装软件包和编译代码

## 参考资料

- [MOFid GitHub仓库](https://github.com/snurr-group/mofid)
- [MOFid论文](https://pubs.acs.org/doi/abs/10.1021/acs.cgd.9b01050)
- [MOFormer项目](https://github.com/snurr-group/MOFormer) 