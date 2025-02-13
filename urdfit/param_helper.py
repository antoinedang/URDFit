from urdfit import print
from brax import System
from jax import numpy as jnp
import jax


class ParamHelper:
    # TODO
    def parse(sys: System) -> jnp.ndarray:
        return jnp.array(sys.link.inertia.mass)

    # TODO
    def make_sys(base_sys: System, params: jnp.ndarray) -> System:
        new_inertia = base_sys.link.inertia.replace(mass=params)
        new_sys = base_sys.replace(link=base_sys.link.replace(inertia=new_inertia))
        return new_sys

    def print(params: jnp.ndarray) -> None:
        jax.debug.print("sys.link.inertia.mass: {}", params)
