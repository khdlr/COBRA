import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import wandb
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw

from io import BytesIO
from PIL import Image
import numpy as np
import base64

from argparse import ArgumentParser
from einops import rearrange

from .jump_flood import jump_flood
from .utils import min_pool


def log_metrics(metrics, prefix, epoch, do_print=True, do_wandb=True):
    metrics = {m: np.mean(metrics[m]) for m in metrics}

    if do_wandb:
        wandb.log({f"{prefix}/{m}": metrics[m] for m in metrics}, step=epoch)
    if do_print:
        print(f"{prefix}/metrics")
        print(", ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))


def get_rgb(data):
    img = data["imagery"]
    if img.shape[-1] == 1:
        rgb = np.concatenate([img] * 3, axis=-1)
    elif img.shape[-1] == 10:
        rgb = img[..., 3:0:-1]
    rgb = np.clip(255 * rgb, 0, 255).astype(np.uint8)
    return rgb


def log_segmentation(data, tag, step):
    H, W, C = data["imagery"].shape

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(get_rgb(data))
    axs[1].imshow(
        np.asarray(data["segmentation"][:, :, 0]), cmap="gray", vmin=-1, vmax=1
    )
    axs[2].imshow(np.asarray(data["mask"]), cmap="gray", vmin=0, vmax=1)

    wandb.log({tag: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_edge(data, tag, step):
    H, W, C = data["imagery"].shape
    mask = data["mask"]
    true_edge = hk.max_pool(mask, [3, 3], [1, 1], "SAME") != min_pool(
        mask, [3, 3], [1, 1], "SAME"
    )

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(get_rgb(data))
    axs[1].imshow(np.asarray(data["edge"][:, :, 0]), cmap="binary", vmin=0, vmax=1)
    axs[2].imshow(np.asarray(true_edge), cmap="binary", vmin=0, vmax=1)

    wandb.log({tag: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_anim(data, tag, step):
    img = get_rgb(data)
    H, W, C = img.shape
    img = Image.fromarray(np.asarray(img))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    imgbase64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    truth = data["contour"]
    gtpath = make_path_string(truth)

    pred = list(data["snake_steps"])
    pred = pred + [pred[-1], pred[-1]]
    path_html = animated_path(pred)

    html = f"""
    <!DOCTYPE html>
    <html>
    <meta charset = "UTF-8">
    <body>
      <svg xmlns="http://www.w3.org/2000/svg" height="100%" viewBox="0 0 256 256">
        <image href="data:image/jpeg;charset=utf-8;base64,{imgbase64}" width="256px" height="256px"/>
        <path fill="none" stroke="rgb(0, 0, 255)" stroke-width="3"
            d="{gtpath}" />
        </path>
        {path_html}
      </svg>
    </body>
    </html>
    """

    wandb.log({tag: wandb.Html(html, inject=False)}, step=step)


def make_path_string(vertices):
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for y, x in vertices)


def animated_path(paths):
    pathvalues = ";".join(make_path_string(path) for path in paths)
    keytimes = ";".join(f"{x:.2f}" for x in np.linspace(0, 1, len(paths)))
    return f"""<path fill="none" stroke="rgb(255, 0, 0)" stroke-width="1">
          <animate attributeName="d" values="{pathvalues}" keyTimes="{keytimes}" dur="3s" repeatCount="indefinite" />
          </path>
          """


def log_offset_field(data, tag, step):
    img = data["imagery"]
    offsets = data["offsets"]
    mask = data["mask"]
    contour = data["contour"]
    true_offsets = jump_flood(mask)

    offsets = jax.image.resize(offsets, [16, 16, 2], "nearest")
    true_offsets = jax.image.resize(true_offsets, [16, 16, 2], "nearest")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis("off")
    ax.imshow(get_rgb(data))

    H, W, C = img.shape

    ry = np.linspace(0, 1, true_offsets.shape[0]) * (H - 1)
    rx = np.linspace(0, 1, true_offsets.shape[1]) * (W - 1)
    x, y = np.meshgrid(rx, ry)
    true_dy = true_offsets[..., 0]
    true_dx = true_offsets[..., 1]
    ax.quiver(x, y, true_dx, true_dy, scale=1, scale_units="xy", angles="xy", color="b")

    ry = np.linspace(0, 1, offsets.shape[0]) * (H - 1)
    rx = np.linspace(0, 1, offsets.shape[1]) * (W - 1)
    x, y = np.meshgrid(rx, ry)
    dy = offsets[..., 0] * H / 2
    dx = offsets[..., 1] * H / 2
    ax.quiver(x, y, dx, dy, scale=1, scale_units="xy", angles="xy", color="red")

    cy, cx = data["contour"].T
    ax.plot(cx, cy, c="b")

    wandb.log({tag: wandb.Image(fig)}, step=step)
    plt.close(fig)


def draw_snake(draw, snake, dashed=False, **kwargs):
    if dashed:
        for (y0, x0, y1, x1) in snake.reshape((-1, 4)):
            draw.line((x0, y0, x1, y1), **kwargs)
    else:
        for (y0, x0), (y1, x1) in zip(snake, snake[1:]):
            draw.line((x0, y0, x1, y1), **kwargs)


def draw_image(img, contour, snake, out_path):
    H, W, C = img.shape
    assert H == 256 and W == 256

    # img = np.asarray(jax.image.resize(img, (256, 256, C), method='linear'))
    img = np.asarray(img)
    img = (255 * img).astype(np.uint8)
    img = np.concatenate([img] * 3, axis=-1)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.imshow(img)
    _plot(ax, contour, c="r", linewidth=3)
    _plot(ax, snake, c="b", linewidth=3)

    ax.set_xlim([-0.5, 255.5])
    ax.set_ylim([255.5, -0.5])
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def draw_steps(img, contour, steps, out_path):
    H, W, C = img.shape
    assert H == 256 and W == 256

    # img = np.asarray(jax.image.resize(img, (256, 256, C), method='linear'))
    img = np.asarray(img)
    img = (255 * img).astype(np.uint8)
    img = np.concatenate([img] * 3, axis=-1)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.imshow(img)
    _plot(ax, contour, c="r", linewidth=3)

    for i, step in enumerate(steps):
        if i == 0 and len(steps) > 1:
            continue
        if i + 1 == len(steps):
            _plot(ax, step, c="b", linewidth=3)
        else:
            _plot(ax, step, c="b", linestyle="dashed", linewidth=1)

    ax.set_xlim([-0.5, 255.5])
    ax.set_ylim([255.5, -0.5])
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def draw_multi(img, contour, snakes, out_path):
    H, W, C = img.shape
    assert H == 256 and W == 256

    # img = np.asarray(jax.image.resize(img, (256, 256, C), method='linear'))
    img = np.asarray(img)
    img = (255 * img).astype(np.uint8)
    img = np.concatenate([img] * 3, axis=-1)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.imshow(img)
    _plot(ax, contour, c="r", linewidth=3)

    for snake in snakes:
        _plot(ax, snake, c="b", linewidth=1)

    ax.set_xlim([-0.5, 255.5])
    ax.set_ylim([255.5, -0.5])
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def draw_uncertainty(img, contour, mean, samples, out_path):
    H, W, C = img.shape
    assert H == 256 and W == 256

    # img = np.asarray(jax.image.resize(img, (256, 256, C), method='linear'))
    img = np.asarray(img)
    img = (255 * img).astype(np.uint8)
    img = np.concatenate([img] * 3, axis=-1)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.imshow(img)
    _plot(ax, contour, c="r", linewidth=3)

    deviation = jnp.stack(samples, axis=0).std()
    # for snake in snakes:
    # _plot(ax, snake, c='b', linewidth=3)

    std = jnp.sqrt(jnp.sum(jnp.stack(samples, axis=0).var(axis=0), axis=-1))
    color = 0.5 * (std[:-1] + std[1:])

    segments = np.stack([mean[:-1, ::-1], mean[1:, ::-1]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, 500 / 30, clip=True)
    lc = LineCollection(segments, cmap="inferno", norm=norm, capstyle="round")
    # Set the values used for colormapping
    lc.set_array(color)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)

    ax.set_xlim([-0.5, 255.5])
    ax.set_ylim([255.5, -0.5])
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _plot(ax, vertices, **kwargs):
    ax.plot(*vertices.T[::-1], **kwargs)
