import json
import os
from torch.utils.data import Dataset
import kara_storage
from kara_storage.pytorch.base import KaraPytorchDatasetBase
from kara_storage.row import RowDataset
import bmtrain as bmt

def make_torch_dataset(ds : RowDataset, shuffle=False, auto_distributed=True, **kwargs) -> 'KaraPytorchDatasetBase':
    
    import torch
    import torch.distributed
    from kara_storage.pytorch.base import KaraPytorchDatasetBase
    from kara_storage.pytorch.iter import SequentialIterator
    from kara_storage.pytorch.shuffle import ShuffleIterator

    if auto_distributed:
        rank = bmt.rank()
        size = bmt.world_size()

        total_length = ds.size()

        ds.slice_(total_length * rank // size, total_length // size)
    if shuffle:
        ret = KaraPytorchDatasetBase(ds,  ShuffleIterator, seed=2333, **kwargs)
    else:
        ret = KaraPytorchDatasetBase(ds, SequentialIterator, seed=2333, **kwargs)
    return ret

def make_kara_dataset(config, mode, encoding="utf8", *args, **params):
    storage = kara_storage.KaraStorage("file://%s" % config.get("data", "%s_data_path" % mode))

    dataset = storage.open_dataset(config.get("data", "%s_kara_namespace" % mode), config.get("data", "%s_kara_dataset" % mode), "r", version=config.get("data", "%s_kara_version" % mode))
    ret = make_torch_dataset(dataset, shuffle=True)
    ret.length = len(dataset)
    return ret

