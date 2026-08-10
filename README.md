# COBRA
Charting Outlines by Recurrent Adaptation – A Deep Contour Detector developed for Charting Greenland's Glacier Calving Fronts.

This repository contains the code for training a deep active contour model
that can be used for calving front detection.

## Setup & Running

The python environment is managed through [`uv`](https://docs.astral.sh/uv/) and defaults to Python 3.12.
While possible to run on a CPU-only machine, it is highly recommended to use a machine with CUDA 12 compatible GPU with at least `16GB` of VRAM. `uv sync` will automatically pull CUDA 12 wheels for `jax`.

Run training as follows (for the [CALFIN](https://github.com/daniel-cheng/CALFIN) dataset):

1. Copy `sample_config.yml` to `config.yml`, change config according to desired experiment.
1. Check out the CALFIN repo to a sibling directory of the project (or adapt `data_root` accordingly in `config.yml`)
1. If experiment tracking through [wandb](https://wandb.ai/) is desired, make sure wandb is authenticated (`uv run wandb login`). Otherwise, turn it off using `uv run wandb off`.
1. Start training by calling `uv run train.py`

## Links
* [Project Page](https://khdlr.github.io/COBRA)
* [Inference Results Map](https://khdlr.github.io/COBRA/map.html)
