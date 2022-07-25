import jax
import jax.numpy as jnp
import haiku as hk
from einops import rearrange, repeat
from .transformer import Transformer


class ContourTransformer:
    def __init__(self, model_dim=384, patch_size=16, vertices=64, n_heads=6,
            encoder_layers=6, decoder_layers=6, dim_feedforward=1024):
        # Corresponds to DeiT-S dimensionalities
        self.model_dim       = model_dim
        self.patch_size      = patch_size
        self.n_vertices      = vertices
        self.n_heads         = n_heads
        self.encoder_layers  = encoder_layers
        self.decoder_layers  = decoder_layers
        self.dim_feedforward = dim_feedforward


    def __call__(self, img, is_training=False):
        B, H, W, C = img.shape

        patch_encoder = hk.Conv2D(self.model_dim, self.patch_size, stride=self.patch_size, with_bias=False)
        transformer   = Transformer(d_model=self.model_dim, nheads=self.n_heads,
                encoder_layers=self.encoder_layers, decoder_layers=self.decoder_layers,
                dim_feedforward=self.dim_feedforward)

        pe_patch      = hk.get_parameter("pe_patch",
            shape=(1, H//self.patch_size, W//self.patch_size, self.model_dim),
            init=hk.initializers.TruncatedNormal()
        )
        pe_contour    = hk.get_parameter("pe_contour",
            shape=(self.n_vertices, self.model_dim),
            init=hk.initializers.TruncatedNormal()
        )
        predictor = hk.Sequential([
            hk.Linear(1024), jax.nn.relu,
            hk.Linear(1024), jax.nn.relu,
            hk.Linear(2)
        ])

        patches    = patch_encoder(img) + pe_patch
        in_tokens  = rearrange(patches, 'b h w c -> b (h w) c', b=B, c=self.model_dim)
        out_tokens = repeat(pe_contour, 't c -> b t c', b=B, c=self.model_dim)

        features = transformer(in_tokens, out_tokens, is_training)
        contour  = predictor(features)

        return {'snake': contour}
