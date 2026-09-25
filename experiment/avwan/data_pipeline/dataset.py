"""Each loader worker owns whole shards, and yields already assembled batches."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class ShardDataset(IterableDataset):
    def __init__(self, cache, split, batch_size, seed=0, shuffle=True):
        self.cache = Path(cache)
        self.manifest = json.loads((self.cache / 'manifest.json').read_text())
        if self.manifest['identity'].get('purpose') != 'avwan_foundation_spatial':
            raise ValueError('Not a foundation cache; prepare a new cache directory')
        self.shards = self.manifest['splits'][split]
        self.batch_size, self.seed, self.shuffle = batch_size, seed, shuffle
        # Shared CPU tensor lets persistent workers see the epoch update.
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def __len__(self):
        return sum(math.ceil(s['samples'] / self.batch_size) for s in self.shards)

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = np.random.default_rng(self.seed + int(self.epoch))
        order = rng.permutation(len(self.shards)) if self.shuffle else np.arange(len(self.shards))
        fields = ['rgb', 'depth', 'spectrogram', 'gm', 'am', 'previous_index', 'action', 'action_mask', 'angle_label']
        for shard_id in order[worker_id::workers]:
            shard = self.shards[shard_id]
            arrays = {key: np.load(self.cache / shard['path'] / f'{key}.npy',
                                   mmap_mode='r', allow_pickle=False) for key in fields}
            # Reproducible sample order regardless of which worker owns this shard.
            local_rng = np.random.default_rng(self.seed + int(self.epoch) * 1000003 + int(shard_id))
            indices = (local_rng.permutation(shard['samples']) if self.shuffle
                       else np.arange(shard['samples']))
            try:
                for offset in range(0, len(indices), self.batch_size):
                    # Sorted gather improves storage locality within each randomized batch.
                    selected = np.sort(indices[offset:offset + self.batch_size])
                    # Advanced indexing already returns an owned, writable copy.
                    batch = {key: torch.from_numpy(array[selected]) for key, array in arrays.items()
                             if key != 'previous_index'}
                    previous = arrays['previous_index'][selected]
                    for key in ('rgb', 'depth'):
                        batch[f'prev_{key}'] = torch.from_numpy(arrays[key][previous])
                    yield batch
            finally:
                for array in arrays.values():
                    array._mmap.close()


def make_loader(config, split):
    cfg = config['training']
    dataset = ShardDataset(config['data']['cache'], split, cfg['batch_size'], config['seed'],
                           shuffle=split == 'train')
    workers = min(cfg['workers'], len(dataset.shards))
    kwargs = dict(batch_size=None, num_workers=workers, pin_memory=cfg['pin_memory'],
                  generator=torch.Generator().manual_seed(config['seed']))
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=cfg['prefetch_factor'],
                      multiprocessing_context='spawn')
    return DataLoader(dataset, **kwargs)
