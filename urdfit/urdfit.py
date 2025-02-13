import jax
from brax.generalized import pipeline
from brax.io import mjcf
from urdfit.param_optimizer import ParamOptimizer
from urdfit.param_helper import ParamHelper
from urdfit.state_evaluator import StateEvaluator
from urdfit import print
from jax import numpy as jnp


class URDFit:
    def __init__(
        self,
        mjcf_path: str,
        timestep: float,
        state_evaluator_config: str = "mse",
        param_optimizer_config: str = "simple_gd",
    ) -> None:
        # LOAD URDF
        self.load_from_mjcf(mjcf_path, timestep)
        # JIT FUNCTIONS FOR SPEED
        self.loss_fn = jax.jit(self.loss_fn)
        self.single_step = jax.jit(self.single_step)
        self.loss_grad = jax.jit(jax.grad(self.loss_fn))
        # INIT PARAM OPTIMIZATION
        self.param_optimizer = ParamOptimizer(self.sys, param_optimizer_config)
        # INIT STATE EVALUATOR
        self.state_evaluator = StateEvaluator(state_evaluator_config)

    def get_optimized_params(self) -> jnp.ndarray:
        return self.param_optimizer.get_params()

    def load_from_mjcf(self, mjcf_path: str, timestep: float) -> None:
        try:
            with open(mjcf_path, "r") as f:
                mjcf_str = f.read()
        except:
            print("ERROR: Could not open MJCF file. (invalid path or file)")
        try:
            self.sys = mjcf.loads(mjcf_str)
        except Exception as e:
            print("ERROR: MJCF could not be loaded as a valid Brax system.")
            print(f"Full Traceback: {e}")
        self.sys = self.sys.replace(dt=timestep)

    def loss_fn(self, params: jnp.ndarray) -> float:
        # UPDATE SYSTEM WITH GIVEN PARAMS
        new_sys = ParamHelper.make_sys(self.sys, params)
        # STEP SYSTEM
        state = pipeline.init(new_sys, self.state_q, self.state_qd)
        sim_next_state = pipeline.step(new_sys, state, self.action)
        # COMPUTE LOSS ON STEPPED STATE
        loss = self.state_evaluator.eval(
            (sim_next_state.q, sim_next_state.qd),
            (self.next_state_q, self.next_state_qd),
        )
        return loss

    def single_step(
        self,
        current_params: jnp.ndarray,
        state_q: jnp.ndarray,
        state_qd: jnp.ndarray,
        action: jnp.ndarray,
        next_state_q: jnp.ndarray,
        next_state_qd: jnp.ndarray,
    ) -> None:

        (
            self.state_q,
            self.state_qd,
            self.action,
            self.next_state_q,
            self.next_state_qd,
        ) = (
            state_q,
            state_qd,
            action,
            next_state_q,
            next_state_qd,
        )
        return self.loss_grad(current_params)

    def step(
        self,
        batched_state_q: jnp.ndarray,
        batched_state_qd: jnp.ndarray,
        batched_action: jnp.ndarray,
        batched_next_state_q: jnp.ndarray,
        batched_next_state_qd: jnp.ndarray,
    ) -> None:
        batch_gradients = []
        current_params = self.get_optimized_params()
        for (
            state_q,
            state_qd,
            action,
            next_state_q,
            next_state_qd,
        ) in zip(
            batched_state_q,
            batched_state_qd,
            batched_action,
            batched_next_state_q,
            batched_next_state_qd,
        ):
            gradient = self.single_step(
                current_params, state_q, state_qd, action, next_state_q, next_state_qd
            )
            batch_gradients.append(gradient)

        self.param_optimizer.update(jnp.array(batch_gradients))
