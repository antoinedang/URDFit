import jax.numpy as jnp
import jax
import time
from urdfit import URDFit, ParamHelper

if __name__ == "__main__":
    # Load MJCF system
    path = "experiment/test.xml"

    urdfit = URDFit(path)

    def run_gradient_step(batch_size=1):
        key = jax.random.PRNGKey(0)  # Seed with 0 (or any other integer)
        state_q = jax.random.uniform(
            key,
            shape=(
                batch_size,
                urdfit.sys.q_size(),
            ),
            minval=-1.0,
            maxval=1.0,
        )
        state_qd = jax.random.uniform(
            key,
            shape=(
                batch_size,
                urdfit.sys.qd_size(),
            ),
            minval=-1.0,
            maxval=1.0,
        )
        action = jax.random.uniform(
            key,
            shape=(
                batch_size,
                urdfit.sys.act_size(),
            ),
            minval=-1.0,
            maxval=1.0,
        )
        next_state_q = jax.random.uniform(
            key,
            shape=(
                batch_size,
                urdfit.sys.q_size(),
            ),
            minval=-1.0,
            maxval=1.0,
        )
        next_state_qd = jax.random.uniform(
            key,
            shape=(
                batch_size,
                urdfit.sys.qd_size(),
            ),
            minval=-1.0,
            maxval=1.0,
        )
        urdfit.step(
            state_q,
            state_qd,
            action,
            next_state_q,
            next_state_qd,
        )

    # RUN GRADIENT COMPUTATION ONCE TO JITTIFY FUNCTIONS
    run_gradient_step()
    N = 10000
    start_time = time.time()
    for _ in range(N):
        run_gradient_step()
    print(
        f"Grad comp speed: {(time.time() - start_time)/N:.4f}s ({N/(time.time() - start_time):.4f} Hz)"
    )
    ParamHelper.print(urdfit.get_optimized_params())
