import os
import hashlib
import random

import torch.utils.data
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from pathlib import Path
from skimage.measure import find_contours
import yaml
from tqdm import tqdm
from skimage.transform import resize
from lib.utils import fnot


def md5(obj):
    obj = str(obj).encode('utf8')
    return hashlib.md5(obj).hexdigest()[:16]


def _isval(gt_path):
    year = int(gt_path.stem[:4])
    return year >= 2020


class DeterministicShuffle(torch.utils.data.Sampler):
    def __init__(self, length, rng_key, repetitions=1):
        self.rng_key = rng_key
        self.length = length
        self.repetitions = repetitions

    def __iter__(self):
        self.rng_key, *subkeys = jax.random.split(self.rng_key, self.repetitions+1)
        permutations = jnp.concatenate([
            jax.random.permutation(subkey, self.length) for subkey in subkeys
        ])
        return permutations.__iter__()

    def __len__(self):
        return self.length * self.repetitions


def numpy_collate(batch):
    """Collate tensors as numpy arrays, taken from
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html"""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_loader(batch_size, data_threads, mode, config, rng_key=None, drop_last=True, subtiles=True):
    data = GlacierFrontDataset(mode=mode, config=config, subtiles=subtiles)

    kwargs = dict(
        batch_size = batch_size,
        num_workers = data_threads,
        collate_fn = numpy_collate,
        drop_last = drop_last
    )
    if mode == 'train':
        kwargs['sampler'] = DeterministicShuffle(len(data), rng_key)

    return torch.utils.data.DataLoader(data, **kwargs)


def snakify(gt, vertices):
    contours = find_contours(gt, 0.5)

    out_contours = []
    for contour in contours:
        # filter our tiny contourlets
        if contour.shape[0] < 12: continue

        contour = contour.astype(np.float32)
        contour = contour.view(np.complex64)[:, 0]
        C_space = np.linspace(0, 1, len(contour), dtype=np.float32)
        S_space = np.linspace(0, 1, vertices, dtype=np.float32)

        snake = np.interp(S_space, C_space, contour)
        snake = snake[:, np.newaxis].view(np.float64).astype(np.float32)

        out_contours.append(snake)

    return out_contours


class GlacierFrontDataset(torch.utils.data.Dataset):
    def __init__(self, mode, config, subtiles=True):
        super().__init__()
        self.mode = mode
        self.subtiles = subtiles

        self.channels = config['data_channels']
        self.tilesize = config['tile_size']
        self.vertices = config['vertices']

        self.root = Path(config['data_root'])
        if (self.root / 'ground_truth').exists() and (self.root / 'reference_data').exists():
            self.data_source = 'TUD'
            ## Loading TUD dataset
            gts = sorted(self.root.glob('ground_truth/*/*/*_30m.tif'))

            istest = lambda gt: gt.parent.name.startswith('202') 
            isval = lambda gt: gt.parent.name.endswith('3')

            if mode == 'test':
                # Use data from 2020 onwards for testing
                gts = filter(istest, gts)
            else:
                gts = filter(fnot(istest), gts)
                if mode == 'train':
                    gts = filter(fnot(isval), gts)
                else:
                    gts = filter(isval, gts)
            self.gts = list(gts)
        else:
            self.data_source = 'CALFIN'
            self.gts = sorted(self.root.glob(f'{mode}/*_mask.png'))

        self.cachedir = self.root / 'cache'
        self.cachedir.mkdir(exist_ok=True)

        confighash = md5((
            self.tilesize,
            self.vertices,
            self.channels,
            self.subtiles))
        prefix = f'{self.data_source}_{mode}_{confighash}'
        self.tile_cache_path  = self.cachedir / f'{prefix}_tile.npy'
        self.mask_cache_path  = self.cachedir / f'{prefix}_mask.npy'
        self.snake_cache_path = self.cachedir / f'{prefix}_snake.npy'

        self.assert_cache()

    def assert_cache(self):
        if not self.snake_cache_path.exists() or not self.tile_cache_path.exists():
            tiles  = []
            masks  = []
            snakes = []

            for tile, mask, snake in self.generate_tiles():
                snakes.append(snake)
                masks.append(mask)
                tiles.append(tile)

            np.save(self.snake_cache_path, np.stack(snakes))
            np.save(self.mask_cache_path, np.stack(masks))
            np.save(self.tile_cache_path, tiles)

        self.snake_cache = np.load(self.snake_cache_path, mmap_mode='r')
        self.mask_cache  = np.load(self.mask_cache_path,  mmap_mode='r')
        self.tile_cache  = np.load(self.tile_cache_path,  mmap_mode='r')

    def generate_raw(self):
        if self.data_source == 'TUD':
            for gtpath in self.gts:
                try:
                    *_, loc, date, _ = gtpath.parts
                    ref_root = self.root / 'reference_data' / loc / date / '30m'
                    channels = []
                    for channel_name in self.channels:
                        channel = np.asarray(Image.open(ref_root / f'{channel_name}.tif'))
                        channels.append(channel.astype(np.uint8))
                    tile = np.stack(channels, axis=-1)
                    # mask = np.asarray(Image.open(gtpath)) > 127
                    mask = np.asarray(Image.open(gtpath)) > 0
                    yield tile, mask
                except FileNotFoundError:
                    # Some of the data are not complete, so we skip them
                    pass
        elif self.data_source == 'CALFIN':
            for gtpath in self.gts:
                tilepath = str(gtpath).replace('_mask.png', '.png')
                tile = np.asarray(Image.open(tilepath))[..., self.channels]
                mask = np.asarray(Image.open(gtpath)) > 127
                yield tile, mask

    def generate_tiles(self):
        prog  = tqdm(self.generate_raw(), total=len(self.gts))
        count = 0
        zeros = 0
        taken = 0
        for tile, mask in prog:
            T = self.tilesize
            H, W, C = tile.shape
            tile  = resize(tile, [2*T, 2*T, C], order=1, anti_aliasing=True, preserve_range=True).astype(np.uint8)
            mask  = resize(mask, [2*T, 2*T], order=0, anti_aliasing=False, preserve_range=True).astype(bool)
            H, W, C = tile.shape

            full_tile  = resize(tile, [T, T, C], order=1, anti_aliasing=True, preserve_range=True).astype(np.uint8)
            full_mask  = resize(mask, [T, T], order=0, anti_aliasing=False, preserve_range=True).astype(bool)

            full_snake = snakify(full_mask, self.vertices)
            if len(full_snake) == 1:
                taken += 1
                yield(full_tile, full_mask, full_snake[0])

            if not self.subtiles:
                continue

            for y in np.linspace(0, H-T, 4).astype(np.int32):
                for x in np.linspace(0, W-T, 4).astype(np.int32):
                    patch      = tile[y:y+T, x:x+T]
                    patch_mask = mask[y:y+T, x:x+T]

                    useful = patch_mask.mean()
                    invalid = np.all(patch == 0, axis=-1).mean()

                    if useful < 0.3 or useful > 0.7 or invalid > 0.2:
                        continue

                    snakes = snakify(patch_mask, self.vertices)
                    count += 1

                    if len(snakes) == 1:
                        taken += 1
                        yield(patch, patch_mask, snakes[0])
                    else:
                        lens = [s.shape[0] for s in snakes]
                        if len(snakes) == 0:
                            zeros += 1
            prog.set_description(f'{taken:5d} tiles')

            # print(f'Overall: {count}, '
            #       f'Extracted: {taken}, '
            #       f'No Border: {zeros}, '
            #       f'Funky: {count - taken - zeros}')

    def __getitem__(self, idx):
        ref   = self.tile_cache[idx]
        mask  = self.mask_cache[idx]
        snake = self.snake_cache[idx]

        ref   = np.expand_dims(ref, -1)

        return ref, mask, snake

    def __len__(self):
        return len(self.snake_cache)


if __name__ == '__main__':
    config = {
        'data_root': '../aicore/uc1/data/',
        'data_channels': ['SPECTRAL/BANDS/NORM_B8_8b'],
        # 'data_root': '../CALFIN/training/data',
        # 'data_channels': [0],
        'vertices': 64,
        'tile_size': 256,
    }
    ds = GlacierFrontDataset('validation', config, subtiles=False)
    print(len(ds))
    for x in ds[0]:
        print(x.shape, x.dtype)
