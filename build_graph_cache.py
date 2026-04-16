"""一次性构建并落盘 crystal graph 缓存，供 multiview 训练复用。

作用：
- 读取 YAML 配置中的 `graph_dataset` 参数（如根目录、半径、邻居数等）。
- 将原始结构数据预处理为图缓存文件，避免训练阶段重复构图。
- 输出缓存清单（manifest），便于核对缓存结果与统计信息。

使用方式（在项目根目录执行）：
- 默认配置运行：
  `python build_graph_cache.py`
- 指定配置与缓存目录：
  `python build_graph_cache.py --config config_multiview.yaml --cache-dir /path/to/cache`
- 仅处理前 N 条做快速检查：
  `python build_graph_cache.py --limit 1000 --log-every 100`
"""

import argparse
import json
import os

import yaml

from dataset.dataset_multiview import build_graph_cache


def parse_args():
    parser = argparse.ArgumentParser(description='Build one-time crystal graph cache for multiview pretraining.')
    parser.add_argument('--config', default='config_multiview.yaml', help='Path to YAML config file.')
    parser.add_argument('--root-dir', default=None, help='Override graph_dataset.root_dir from config.')
    parser.add_argument('--cache-dir', default=None, help='Override graph_dataset.cache_dir from config.')
    parser.add_argument('--limit', type=int, default=None, help='Only cache the first N samples for smoke testing.')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild cache files even if they already exist.')
    parser.add_argument('--log-every', type=int, default=1000, help='Print progress every N samples.')
    parser.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1),
                        help='Number of worker processes used during one-time graph caching.')
    parser.add_argument('--chunksize', type=int, default=16,
                        help='Task chunksize for multiprocessing graph cache build.')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    graph_config = dict(config['graph_dataset'])
    root_dir = args.root_dir or graph_config['root_dir']
    cache_dir = args.cache_dir or graph_config.get('cache_dir')
    if not cache_dir:
        raise ValueError('graph_dataset.cache_dir is not configured. Please set it in config_multiview.yaml or pass --cache-dir.')

    manifest = build_graph_cache(
        root_dir=root_dir,
        cache_dir=cache_dir,
        max_num_nbr=graph_config['max_num_nbr'],
        radius=graph_config['radius'],
        dmin=graph_config['dmin'],
        step=graph_config['step'],
        limit=args.limit,
        overwrite=args.overwrite,
        log_every=args.log_every,
        num_workers=args.workers,
        chunksize=args.chunksize,
    )

    print('graph cache ready:')
    print(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == '__main__':
    main()
