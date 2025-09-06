import numpy as np
import sys
import os

def check_npy_format(npy_file):
    """
    检查npy文件内容，打印其基本信息和前几个条目
    Args:
        npy_file: npy文件路径
    """
    if not os.path.exists(npy_file):
        print(f"文件不存在: {npy_file}")
        return

    try:
        data = np.load(npy_file, allow_pickle=True)
        print(f"成功加载: {npy_file}")
        print(f"数据类型: {type(data)}")
        print(f"数据shape: {data.shape}")
        print(f"数据dtype: {data.dtype}")

        # 打印前5个条目
        print("\n前5个条目:")
        for i, item in enumerate(data[:5]):
            print(f"{i}: {item}")
    except Exception as e:
        print(f"加载npy文件失败: {e}")

if __name__ == "__main__":
    npy_file = '/root/autodl-tmp/MOFormer/cif_all/id_prop.npy'
    # /mofid_string.npy
    # /cif_id.npy

    check_npy_format(npy_file)
