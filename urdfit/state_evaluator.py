from jax import numpy as jnp
import jax


class StateEvaluator:
    def __init__(self, config):
        self.config = config
        self.eval = jax.jit(self.eval)

    def eval(self, sim_state, true_state):
        sim_q, sim_qd = sim_state
        true_q, true_qd = true_state
        if self.config == "mse":
            q_loss = jnp.mean((sim_q - true_q) ** 2)
            qd_loss = jnp.mean((sim_qd - true_qd) ** 2)
            return (q_loss + qd_loss) / 2
        else:
            raise ValueError("Invalid config")
