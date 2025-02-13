import jax
import jax.numpy as jnp
from urdfit import print
from urdfit.param_helper import ParamHelper, Params
from brax import System


class ParamOptimizer:
    def __init__(self, sys: System, config: str) -> None:
        self.params = ParamHelper.parse(sys)
        self.config = config

    def update(self, batch_gradients: jnp.ndarray) -> None:
        if self.config == "simple_gd":
            batch_gradients = jnp.nan_to_num(batch_gradients)
            self.params = self.params - 0.0001 * jnp.mean(batch_gradients, axis=0)
        else:
            raise ValueError("Invalid config")

    def get_params(self) -> Params:
        return self.params
