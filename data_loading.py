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


def get_loader(batch_size, data_threads, mode, rng_key):
    data = CalfinDataset(mode=mode)

    kwargs = dict(
        batch_size = batch_size,
        num_workers = data_threads,
        collate_fn = numpy_collate,
        drop_last = True
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


class CalfinDataset(torch.utils.data.Dataset):
    def __init__(self, mode):
        super().__init__()
        self.config = yaml.load(open('config.yml'), Loader=yaml.SafeLoader)

        self.mode = mode
        self.root = Path(self.config['data_root']) / mode

        self.cachedir = self.root.parent / 'cache'
        self.cachedir.mkdir(exist_ok=True)
        self.confighash = md5((self.config['tile_size'], self.config['vertices']))

        self.tile_cache_path   = self.cachedir / f'{mode}_tile_{self.confighash}.npy'
        self.mask_cache_path  = self.cachedir / f'{mode}_mask_{self.confighash}.npy'
        self.snake_cache_path = self.cachedir / f'{mode}_snake_{self.confighash}.npy'

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

    def generate_tiles(self):
        prog = tqdm(list(self.root.glob('*_mask.png')))
        count = 0
        zeros = 0
        taken = 0
        for maskpath in prog:
            tilesize = self.config['tile_size']
            tilepath = str(maskpath).replace('_mask.png', '.png')
            tile = np.asarray(Image.open(tilepath))
            mask = np.asarray(Image.open(maskpath)) > 127

            tile  = resize(tile, [512, 512, 3], order=1, anti_aliasing=True, preserve_range=True).astype(np.uint8)
            mask  = resize(mask, [512, 512], order=0, anti_aliasing=False, preserve_range=True).astype(bool)
            H, W, C = tile.shape

            full_tile  = resize(tile, [256, 256, 3], order=1, anti_aliasing=True, preserve_range=True).astype(np.uint8)
            full_mask  = resize(mask, [256, 256], order=0, anti_aliasing=False, preserve_range=True).astype(bool)

            full_snake = snakify(full_mask, self.config['vertices'])
            if len(full_snake) == 1:
                taken += 1
                yield(full_tile, full_mask, full_snake[0])

            for y in np.linspace(0, H-tilesize, 4).astype(np.int32):
                for x in np.linspace(0, W-tilesize, 4).astype(np.int32):
                    patch = tile[y:y+tilesize, x:x+tilesize]
                    patch_mask = mask[y:y+tilesize, x:x+tilesize]

                    useful = patch_mask.mean()
                    invalid = np.all(patch == 0, axis=-1).mean()

                    if useful < 0.3 or useful > 0.7 or invalid > 0.2:
                        continue

                    snakes = snakify(patch_mask, self.config['vertices'])
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
        refchannel = 2
        if self.mode == 'train':
            refchannel = random.randint(0, 2)

        ref   = self.tile_cache[idx, ..., refchannel]
        mask  = self.mask_cache[idx]
        snake = self.snake_cache[idx]

        ref   = np.expand_dims(ref, -1)

        return ref, mask, snake

    def __len__(self):
        return len(self.snake_cache)


if __name__ == '__main__':
    ds = CalfinDataset('validation_zhang')
    print(len(ds))
    for a in tqdm(ds):
        for x in a:
            print(x.shape, x.dtype)
        break
