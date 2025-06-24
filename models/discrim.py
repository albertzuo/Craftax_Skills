import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

import distrax


class Discriminator(nn.Module):
    num_skills: int
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh        

        
        discriminator = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        discriminator = activation(discriminator)

        discriminator = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(discriminator)
        discriminator = activation(discriminator)

        discriminator = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(discriminator)
        discriminator = activation(discriminator)

        discriminator = nn.Dense(
            self.num_skills, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(discriminator)
        discriminator = distrax.Categorical(logits=discriminator)

        return discriminator
