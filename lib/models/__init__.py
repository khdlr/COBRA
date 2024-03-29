from .cobra import COBRA
from .unet import UNet
from .calfin import CFM
from .hed_unet import HEDUNet
from .rupprecht_dac import RupprechtDAC, RupprechtUNetDAC
from .dance import DANCE
from .deepsnake import DeepSnake

import jax
import haiku as hk
from inspect import signature

def get_model(config, dummy_in, seed=jax.random.PRNGKey(39)):
    model_args = config['model_args']
    modelclass = globals()[config['model']]
    if 'vertices' in signature(modelclass).parameters:
        model_args['vertices'] = config['vertices']
    model = modelclass(**model_args)
    model = hk.transform_with_state(model)

    params, buffers = model.init(seed, dummy_in[:1], is_training=True)

    return model, params, buffers
