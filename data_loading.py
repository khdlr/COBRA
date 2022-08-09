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
    obj = str(obj).encode("utf8")
    return hashlib.md5(obj).hexdigest()[:16]


def _isval(gt_path):
    year = int(gt_path.stem[:4])
    return year >= 2020


def numpy_collate(batch):
    """Collate tensors as numpy arrays, taken from
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html"""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_loader(
    batch_size, data_threads, mode, config, rng_key=None, drop_last=True, subtiles=True
):
    data = GlacierFrontDataset(mode=mode, config=config, subtiles=subtiles)

    kwargs = dict(
        batch_size=batch_size,
        num_workers=data_threads,
        collate_fn=numpy_collate,
        drop_last=drop_last,
    )
    if mode == "train":
        kwargs["shuffle"] = True

    return torch.utils.data.DataLoader(data, **kwargs)


def snakify(gt, vertices):
    contours = find_contours(gt, 0.5)

    out_contours = []
    for contour in contours:
        # filter our tiny contourlets
        if contour.shape[0] < 12:
            continue

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

        self.tilesize = config["tile_size"]
        self.vertices = config["vertices"]
        self.root = Path(config["data_root"])
        self.data_source = config["dataset"]
        if "data_channels" in config:
            self.channels = config["data_channels"]

        if self.data_source == "TUD":
            gts = sorted(self.root.glob("ground_truth/*/*/*_30m.tif"))

            istest = lambda gt: gt.parent.name.startswith("202")
            isval = lambda gt: gt.parent.name.endswith("3")

            if mode == "test":
                # Use data from 2020 onwards for testing
                gts = filter(istest, gts)
            else:
                gts = filter(fnot(istest), gts)
                if mode == "train":
                    gts = filter(fnot(isval), gts)
                else:
                    gts = filter(isval, gts)
            self.gts = list(gts)
        elif self.data_source == "TUD-MS":
            if mode in ("train", "validation"):
                ref_root = self.root / "TRAIN"
            elif mode == "test":
                ref_root = self.root / "TEST"
            elif mode == "test_esa":
                ref_root = self.root / "TEST_ESA-CCI"
            elif mode == "test_calfin":
                ref_root = self.root / "TEST_CALFIN"
            else:
                raise ValueError(f"Cannot provide data for dataset mode {mode}")

            refs = sorted(ref_root.glob("*/*"))
            isval = lambda gt: gt.name.endswith("3")
            if mode == "validation":
                refs = filter(isval, refs)
            elif mode == "train":
                refs = filter(fnot(isval), refs)

            self.gts = []
            for ref in refs:
                *_, site, date = ref.parts
                gt = self.root / "ground_truth" / site / date / f"{date}_30m.tif"
                assert gt.exists()
                self.gts.append(gt)

        elif self.data_source == "CALFIN":
            self.data_source = "CALFIN"
            self.gts = sorted(self.root.glob(f"{mode}/*_mask.png"))

        self.cachedir = self.root / "cache"
        self.cachedir.mkdir(exist_ok=True)

        confighash = md5(
            (
                self.tilesize,
                self.vertices,
                # self.data_source,
                self.subtiles,
            )
        )
        prefix = f"{self.data_source}_{mode}_{confighash}"
        self.tile_cache_path = self.cachedir / f"{prefix}_tile.npy"
        self.mask_cache_path = self.cachedir / f"{prefix}_mask.npy"
        self.snake_cache_path = self.cachedir / f"{prefix}_snake.npy"

        self.assert_cache()

    def assert_cache(self):
        if not self.snake_cache_path.exists() or not self.tile_cache_path.exists():
            tiles = []
            masks = []
            snakes = []

            for tile, mask, snake in self.generate_tiles():
                snakes.append(snake)
                masks.append(mask)
                tiles.append(tile)

            np.save(self.snake_cache_path, np.stack(snakes))
            np.save(self.mask_cache_path, np.stack(masks))
            np.save(self.tile_cache_path, tiles)

        self.snake_cache = np.load(self.snake_cache_path, mmap_mode="r")
        self.mask_cache = np.load(self.mask_cache_path, mmap_mode="r")
        self.tile_cache = np.load(self.tile_cache_path, mmap_mode="r")

    def generate_CALFIN(self):
        for gtpath in self.gts:
            tilepath = str(gtpath).replace("_mask.png", ".png")
            tile = np.asarray(Image.open(tilepath))[..., 1:2]
            mask = np.asarray(Image.open(gtpath)) > 127
            yield tile, mask

    def generate_TUD(self):
        for gtpath in self.gts:
            try:
                *_, site, date, _ = gtpath.parts
                ref_root = self.root / "reference_data" / site / date / "30m"
                channels = []
                for channel_name in self.channels:
                    channel = np.asarray(Image.open(ref_root / f"{channel_name}.tif"))
                    channels.append(channel.astype(np.uint8))
                tile = np.stack(channels, axis=-1)
                # mask = np.asarray(Image.open(gtpath)) > 127
                mask = np.asarray(Image.open(gtpath)) > 0
                yield tile, mask
            except FileNotFoundError:
                # Some of the data are not complete, so we skip them
                pass

    def generate_TUD_MS(self):
        if self.mode in ("train", "validation"):
            ref_root = self.root / "TRAIN"
        elif self.mode == "test":
            ref_root = self.root / "TEST"

        for gtpath in self.gts:
            try:
                *_, site, date, _ = gtpath.parts
                img_ref_root = ref_root / site / date / "30m" / "SPECTRAL" / "BANDS"
                channels = []
                for channel_name in [
                    "B1",
                    "B2",
                    "B3",
                    "B4",
                    "B5",
                    "B6",
                    "B7",
                    "B8",
                    "B10",
                    "B11",
                ]:
                    channel = np.asarray(
                        Image.open(img_ref_root / f"{channel_name}.tif")
                    )
                    # Scale by two sigma
                    mu = channel.mean()
                    sigma = channel.std()

                    a = 1 / (4 * sigma)
                    b = 0.5 - mu * a

                    channel = np.clip(a * channel + b, 0, 1)
                    channel = (255 * channel).astype(np.uint8)
                    channels.append(channel)
                tile = np.stack(channels, axis=-1)
                mask = np.asarray(Image.open(gtpath)) > 0
                yield tile, mask
            except FileNotFoundError:
                print(f"FileNotFoundError for {gtpath}")
                pass

    def generate_tiles(self):
        if self.data_source == "CALFIN":
            generator = self.generate_CALFIN()
        elif self.data_source == "TUD":
            generator = self.generate_TUD()
        elif self.data_source == "TUD-MS":
            generator = self.generate_TUD_MS()

        prog = tqdm(generator, total=len(self.gts))
        count = 0
        zeros = 0
        taken = 0
        for tile, mask in prog:
            T = self.tilesize
            H, W, C = tile.shape
            tile = resize(
                tile,
                [2 * T, 2 * T, C],
                order=1,
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.uint8)
            mask = resize(
                mask,
                [2 * T, 2 * T],
                order=0,
                anti_aliasing=False,
                preserve_range=True,
            ).astype(bool)
            H, W, C = tile.shape

            full_tile = resize(
                tile, [T, T, C], order=1, anti_aliasing=True, preserve_range=True
            ).astype(np.uint8)
            full_mask = resize(
                mask, [T, T], order=0, anti_aliasing=False, preserve_range=True
            ).astype(bool)

            full_snake = snakify(full_mask, self.vertices)
            if len(full_snake) == 1:
                taken += 1
                yield (full_tile, full_mask, full_snake[0])

            if not self.subtiles:
                continue

            for y in np.linspace(0, H - T, 4).astype(np.int32):
                for x in np.linspace(0, W - T, 4).astype(np.int32):
                    patch = tile[y : y + T, x : x + T]
                    patch_mask = mask[y : y + T, x : x + T]

                    useful = patch_mask.mean()
                    invalid = np.all(patch == 0, axis=-1).mean()

                    if useful < 0.3 or useful > 0.7 or invalid > 0.2:
                        continue

                    snakes = snakify(patch_mask, self.vertices)
                    count += 1

                    if len(snakes) == 1:
                        taken += 1
                        yield (patch, patch_mask, snakes[0])
                    else:
                        lens = [s.shape[0] for s in snakes]
                        if len(snakes) == 0:
                            zeros += 1
            prog.set_description(f"{taken:5d} tiles")

            # print(f'Overall: {count}, '
            #       f'Extracted: {taken}, '
            #       f'No Border: {zeros}, '
            #       f'Funky: {count - taken - zeros}')

    def __getitem__(self, idx):
        ref = self.tile_cache[idx]
        mask = self.mask_cache[idx]
        snake = self.snake_cache[idx]

        return ref, mask, snake

    def __len__(self):
        return len(self.snake_cache)


if __name__ == "__main__":
    config = {
        "dataset": "TUD-MS",
        "data_root": "../aicore/uc1_new/",
        "vertices": 64,
        "tile_size": 256,
    }
    ds = GlacierFrontDataset("test_esa", config, subtiles=False)
    print(len(ds))
    for x in ds[0]:
        print(x.shape, x.dtype)
