import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from tensorflow.keras.datasets import cifar10
import numpy as np

class MLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x
