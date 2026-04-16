from __future__ import print_function, division

import csv
import functools
import  json
#import  you
import  random
import warnings
import math
import  numpy  as  np
import torch
import os
import time
from multiprocessing import Pool
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from dataset.augmentation import RotationTransformation, PerturbStructureTransformation, RemoveSitesTransformation


class CORE_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio = 1, which_label = 'void_fraction'):
            label_dict = {
                'void_fraction':2,
                'pld':3,
                'lcd':4
            }
            self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data[:, 1].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, label_dict[which_label]].astype(float)
            # self.label = self.label/np.max(self.label)
            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))
            y = torch.from_numpy(np.asarray(self.label[index])).view(-1,1)

            return X, y.float()

class MOF_ID_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio = 1):

            self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data[:, 1].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, 2].astype(float)

            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))
            y = torch.from_numpy(np.asarray(self.label[index])).view(-1,1)

            return X, y.float()


class MOF_pretrain_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio = 1):

            self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data.astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.mofid)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))

            return X.type(torch.LongTensor)


class MOF_tsne_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio = 1):

            self.data = data[:int(len(data)*use_ratio)]
            self.mofname = self.data[:, 0].astype(str)
            self.mofid = self.data[:, 1].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, 2].astype(float)

            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))

            return X, self.label[index], self.mofname[index], self.mofid[index]





def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, val_ratio=0.1, random_seed = 11, num_workers=1, 
                              pin_memory=False, persistent_workers=False, prefetch_factor=2, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.
    !!! The dataset needs to be shuffled before using the function !!!
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool
    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    train_ratio = 1 - val_ratio
    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    indices = list(range(total_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[train_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers, drop_last=True,
                              collate_fn=collate_fn, pin_memory=pin_memory,
                              persistent_workers=persistent_workers and num_workers > 0,
                              prefetch_factor=prefetch_factor if num_workers > 0 else 2)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers, drop_last=True,
                            collate_fn=collate_fn, pin_memory=pin_memory,
                            persistent_workers=persistent_workers and num_workers > 0,
                            prefetch_factor=prefetch_factor if num_workers > 0 else 2)
    return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    Parameters
    ----------
    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)
      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int
    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_tokens = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), tokens, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_tokens.append(tokens)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.cat(batch_tokens, dim=0),\
        batch_cif_ids


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class  AtomCustomJSONInitializer ( AtomInitializer ):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def _normalize_cif_id(cif_id):
    cif_id = str(cif_id)
    if cif_id.endswith('.cif'):
        return cif_id[:-4]
    return cif_id


def _cif_filename(cif_id):
    cif_id = str(cif_id)
    if cif_id.endswith('.cif'):
        return cif_id
    return cif_id + '.cif'


def _load_id_prop_data(root_dir):
    id_prop_file = os.path.join(root_dir, 'id_prop.npy')
    assert os.path.exists(id_prop_file), 'id_prop.npy does not exist!'
    return np.load(id_prop_file, allow_pickle=True)


def _resolve_cif_root_dir(root_dir, id_prop_data):
    candidate_dirs = [
        root_dir,
        os.path.join(root_dir, 'cif'),
    ]
    for candidate_dir in candidate_dirs:
        if not os.path.isdir(candidate_dir):
            continue
        for sample_cif_id, _ in id_prop_data[:100]:
            sample_name = _cif_filename(sample_cif_id)
            if os.path.exists(os.path.join(candidate_dir, sample_name)):
                return candidate_dir
    raise FileNotFoundError(
        f'Cannot find CIF files under {root_dir}. '
        'Expected either root_dir/*.cif or root_dir/cif/*.cif.'
    )


def _encode_mofid(tokenizer, mofid):
    tokens = tokenizer.encode(mofid, max_length=512, truncation=True, padding='max_length')
    tokens = np.asarray([tokens], dtype=np.int64)
    return torch.from_numpy(tokens)


def _build_graph_arrays(cif_root_dir, cif_id, atom_initializer, gaussian_distance, max_num_nbr, radius):
    crystal = Structure.from_file(os.path.join(cif_root_dir, _cif_filename(cif_id))).copy()

    atom_fea = np.vstack([
        atom_initializer.get_atom_fea(crystal[i].specie.number)
        for i in range(len(crystal))
    ]).astype(np.float32)

    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            warnings.warn('{} not find enough neighbors to build graph. '
                          'If it happens frequently, consider increase '
                          'radius.'.format(cif_id))
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                               [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                           [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))

    nbr_fea_idx = np.asarray(nbr_fea_idx, dtype=np.int64)
    nbr_fea = np.asarray(nbr_fea, dtype=np.float32)
    nbr_fea = gaussian_distance.expand(nbr_fea).astype(np.float32)
    return atom_fea, nbr_fea, nbr_fea_idx


def _graph_cache_file(cache_dir, cif_id):
    return os.path.join(cache_dir, _normalize_cif_id(cif_id) + '.npz')


def _graph_cache_manifest(cache_dir):
    return os.path.join(cache_dir, 'cache_meta.json')


_GRAPH_CACHE_WORKER_STATE = {}


def _init_graph_cache_worker(cif_root_dir, atom_init_file, dmin, radius, step, max_num_nbr, cache_dir, overwrite):
    global _GRAPH_CACHE_WORKER_STATE
    _GRAPH_CACHE_WORKER_STATE = {
        'cif_root_dir': cif_root_dir,
        'atom_initializer': AtomCustomJSONInitializer(atom_init_file),
        'gaussian_distance': GaussianDistance(dmin=dmin, dmax=radius, step=step),
        'max_num_nbr': max_num_nbr,
        'radius': radius,
        'cache_dir': cache_dir,
        'overwrite': overwrite,
    }


def _build_graph_cache_worker(cif_id):
    state = _GRAPH_CACHE_WORKER_STATE
    cache_file = _graph_cache_file(state['cache_dir'], cif_id)
    if not state['overwrite'] and os.path.exists(cache_file):
        return 'skipped'

    atom_fea, nbr_fea, nbr_fea_idx = _build_graph_arrays(
        cif_root_dir=state['cif_root_dir'],
        cif_id=cif_id,
        atom_initializer=state['atom_initializer'],
        gaussian_distance=state['gaussian_distance'],
        max_num_nbr=state['max_num_nbr'],
        radius=state['radius'],
    )
    np.savez(cache_file, atom_fea=atom_fea, nbr_fea=nbr_fea, nbr_fea_idx=nbr_fea_idx)
    return 'built'


def build_graph_cache(root_dir, cache_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                      limit=None, overwrite=False, log_every=1000, num_workers=1, chunksize=16):
    root_dir = os.path.abspath(root_dir)
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    id_prop_data = _load_id_prop_data(root_dir)
    if limit is not None:
        id_prop_data = id_prop_data[:int(limit)]
    cif_root_dir = _resolve_cif_root_dir(root_dir, id_prop_data)

    atom_init_file = os.path.join('benchmark_datasets/atom_init.json')
    assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
    atom_initializer = AtomCustomJSONInitializer(atom_init_file)
    gaussian_distance = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    total = len(id_prop_data)
    processed = 0
    skipped = 0
    start_time = time.time()

    cif_ids = [str(cif_id) for cif_id, _ in id_prop_data]
    if num_workers and num_workers > 1:
        with Pool(
            processes=num_workers,
            initializer=_init_graph_cache_worker,
            initargs=(cif_root_dir, atom_init_file, dmin, radius, step, max_num_nbr, cache_dir, overwrite),
        ) as pool:
            for index, status in enumerate(pool.imap_unordered(_build_graph_cache_worker, cif_ids, chunksize=chunksize), start=1):
                if status == 'built':
                    processed += 1
                else:
                    skipped += 1

                if index == 1 or index % log_every == 0 or index == total:
                    elapsed = time.time() - start_time
                    speed = index / elapsed if elapsed > 0 else 0
                    print(
                        f'cache progress: {index}/{total} '
                        f'(new={processed}, skipped={skipped}, speed={speed:.2f} samples/s)'
                    )
    else:
        _init_graph_cache_worker(cif_root_dir, atom_init_file, dmin, radius, step, max_num_nbr, cache_dir, overwrite)
        for index, cif_id in enumerate(cif_ids, start=1):
            status = _build_graph_cache_worker(cif_id)
            if status == 'built':
                processed += 1
            else:
                skipped += 1

            if index == 1 or index % log_every == 0 or index == total:
                elapsed = time.time() - start_time
                speed = index / elapsed if elapsed > 0 else 0
                print(
                    f'cache progress: {index}/{total} '
                    f'(new={processed}, skipped={skipped}, speed={speed:.2f} samples/s)'
                )

    manifest = {
        'version': 1,
        'complete': limit is None,
        'source_root_dir': root_dir,
        'cif_root_dir': cif_root_dir,
        'num_items': int(len(_load_id_prop_data(root_dir))),
        'cached_items': int(total),
        'max_num_nbr': int(max_num_nbr),
        'radius': float(radius),
        'dmin': float(dmin),
        'step': float(step),
        'num_workers': int(num_workers),
    }
    with open(_graph_cache_manifest(cache_dir), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True, sort_keys=True)

    return manifest


class _BaseMultiviewDataset(Dataset):
    def __init__(self, root_dir, tokenizer, random_seed=123):
        self.tokenizer = tokenizer
        self.root_dir = root_dir
        self.random_seed = random_seed
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        self.id_prop_data = _load_id_prop_data(self.root_dir)
        self.cif_root_dir = _resolve_cif_root_dir(self.root_dir, self.id_prop_data)

    def __len__(self):
        return len(self.id_prop_data)

    def _get_metadata(self, idx):
        cif_id, mofid = self.id_prop_data[idx]
        return str(cif_id), str(mofid)

    def _get_tokens(self, mofid):
        return _encode_mofid(self.tokenizer, mofid)


class CIFData(Dataset):
    """
CIFData数据集是对以CIF文件形式存储晶体结构的数据集的封装。该数据集应具有以下目录结构：
    root_dir
    ├── id_prop.npy
    ├── id0.cif
    ├── id1.cif
    ├── ...
  或者：
    root_dir
    ├── id_prop.npy
    └── cif/
        ├── id0.cif
        ├── id1.cif
        ├── ...
  id_prop.npy：一个包含两列的numpy数组。第一列记录每个晶体的唯一ID，第二列记录对应的MOFid字符串。
  atom_init.json：一个存储每个元素初始化向量的JSON文件。
  ID.cif：一个记录晶体结构的CIF文件，其中ID为该晶体的唯一标识符。
  Parameters
    ----------
    root_dir: str
        数据集根目录的路径
    max_num_nbr: int
        构建晶体图时，搜索邻居的最大数量
    radius: float
        搜索邻居的截断半径
    dmin: float
        构建高斯距离时，最小距离
    step: float
        构建高斯距离时，步长
    random_seed: int
        打乱数据集的随机种子
    Returns
    -------
    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, tokenizer, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        super().__init__()
        self.metadata = _BaseMultiviewDataset(root_dir=root_dir, tokenizer=tokenizer, random_seed=random_seed)
        self.tokenizer = self.metadata.tokenizer
        self.root_dir = self.metadata.root_dir
        self.id_prop_data = self.metadata.id_prop_data
        self.cif_root_dir = self.metadata.cif_root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.uses_graph_cache = False
        atom_init_file = os.path.join('benchmark_datasets/atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=5000)  # Cache loaded structures - 启用缓存减少重复计算，4090 24G支持更大缓存
    def __getitem__(self, idx):
        cif_id, mofid = self.metadata._get_metadata(idx)
        atom_fea, nbr_fea, nbr_fea_idx = _build_graph_arrays(
            cif_root_dir=self.cif_root_dir,
            cif_id=cif_id,
            atom_initializer=self.ari,
            gaussian_distance=self.gdf,
            max_num_nbr=self.max_num_nbr,
            radius=self.radius,
        )
        tokens = self.metadata._get_tokens(mofid)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        return (atom_fea, nbr_fea, nbr_fea_idx), tokens, cif_id


class CachedCIFData(Dataset):
    def __init__(self, root_dir, tokenizer, cache_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        super().__init__()
        self.metadata = _BaseMultiviewDataset(root_dir=root_dir, tokenizer=tokenizer, random_seed=random_seed)
        self.tokenizer = self.metadata.tokenizer
        self.root_dir = self.metadata.root_dir
        self.id_prop_data = self.metadata.id_prop_data
        self.cif_root_dir = self.metadata.cif_root_dir
        self.cache_dir = cache_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.uses_graph_cache = True
        self._validate_cache()

    def _validate_cache(self):
        assert os.path.isdir(self.cache_dir), f'cache_dir does not exist: {self.cache_dir}'
        manifest_path = _graph_cache_manifest(self.cache_dir)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f'Graph cache manifest not found: {manifest_path}')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        if not manifest.get('complete', False):
            raise ValueError(
                f'Graph cache at {self.cache_dir} is incomplete. '
                'Please finish the one-time cache build before using it for training.'
            )
        if os.path.abspath(self.root_dir) != manifest.get('source_root_dir'):
            raise ValueError('Graph cache source_root_dir does not match current dataset root_dir.')
        if len(self.id_prop_data) != manifest.get('num_items'):
            raise ValueError('Graph cache item count does not match current id_prop.npy.')
        for key, expected in (
            ('max_num_nbr', self.max_num_nbr),
            ('radius', self.radius),
            ('dmin', self.dmin),
            ('step', self.step),
        ):
            cached_value = manifest.get(key)
            if float(cached_value) != float(expected):
                raise ValueError(f'Graph cache parameter mismatch for {key}: cache={cached_value}, current={expected}')

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=5000)
    def __getitem__(self, idx):
        cif_id, mofid = self.metadata._get_metadata(idx)
        cache_file = _graph_cache_file(self.cache_dir, cif_id)
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f'Graph cache file missing for {cif_id}: {cache_file}. '
                'Please rebuild the graph cache or disable cached loading.'
            )
        crystal_graph = np.load(cache_file)
        atom_fea = torch.Tensor(crystal_graph['atom_fea'])
        nbr_fea = torch.Tensor(crystal_graph['nbr_fea'])
        nbr_fea_idx = torch.LongTensor(crystal_graph['nbr_fea_idx'])
        tokens = self.metadata._get_tokens(mofid)
        return (atom_fea, nbr_fea, nbr_fea_idx), tokens, cif_id


def build_multiview_dataset(root_dir, tokenizer, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                            random_seed=123, cache_dir=None, use_graph_cache=False):
    dataset_kwargs = dict(
        root_dir=root_dir,
        tokenizer=tokenizer,
        max_num_nbr=max_num_nbr,
        radius=radius,
        dmin=dmin,
        step=step,
        random_seed=random_seed,
    )
    if use_graph_cache:
        if not cache_dir:
            raise ValueError('use_graph_cache=True requires graph_dataset.cache_dir to be configured.')
        manifest_path = _graph_cache_manifest(cache_dir)
        if os.path.exists(manifest_path):
            return CachedCIFData(cache_dir=cache_dir, **dataset_kwargs)
        warnings.warn(
            f'Graph cache requested but manifest was not found at {manifest_path}. '
            'Falling back to on-the-fly CIF parsing.'
        )
    return CIFData(**dataset_kwargs)
