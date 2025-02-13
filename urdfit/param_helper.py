from urdfit import print
from brax import System
from jax import numpy as jnp
import jax

Params = jnp.ndarray


class ParamHelper:
    TARGET_PARAMETERS = [
        "actuator_actrange",
        "actuator_biasprm",
        "actuator_dynprm",
        "actuator_forcerange",
        "actuator_gainprm",
        "actuator_gear",
        "body_inertia",
        "body_mass",
        "body_gravcomp",
        "density",
        "dof_M0",
        "dof_armature",
        "dof_damping",
        "dof_frictionloss",
        "dof_solimp",
        "dof_solref",
        "ang_damping",
        "jnt_actfrcrange",
        "jnt_axis",
        "jnt_range",
        "jnt_solimp",
        "jnt_solref",
        "jnt_stiffness",
        "joint_scale_ang",
        "joint_scale_pos",
        "jnt_margin",
        "baumgarte_erp",
        "collide_scale",
        "elasticity",
        "eq_solimp",
        "eq_solref",
        "spring_inertia_scale",
        "spring_mass_scale",
        "geom_friction",
        "geom_gap",
        "geom_margin",
        "geom_size",
        "geom_solimp",
        "geom_solmix",
        "geom_solref",
        "gravity",
        "viscosity",
    ]
    DECODE_KEYS = {}

    @staticmethod
    @jax.jit
    def parse(sys: System) -> Params:
        all_params = []
        for param_name in ParamHelper.TARGET_PARAMETERS:
            raw_val = jnp.array(getattr(sys, param_name))
            len_before_extend = len(all_params)
            all_params.extend(raw_val.flatten())
            ParamHelper.DECODE_KEYS[param_name] = {
                "idx": slice(len_before_extend, len(all_params)),
                "shape": raw_val.shape,
            }

        all_params = jnp.array(all_params)

        return all_params

    @staticmethod
    @jax.jit
    def make_sys(base_sys: System, params: Params) -> System:
        new_sys = base_sys.replace(
            **{
                param_name: params[decode_key["idx"]].reshape(decode_key["shape"])
                for param_name, decode_key in ParamHelper.DECODE_KEYS.items()
            }
        )
        return new_sys

    @staticmethod
    def print(params: Params) -> None:
        for param_name, decode_key in ParamHelper.DECODE_KEYS.items():
            jax.debug.print(
                param_name + ": {}",
                params[decode_key["idx"]].reshape(decode_key["shape"]),
            )
