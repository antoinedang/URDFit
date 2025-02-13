import jax


def print(*args):
    jax_print_args = []
    for arg in args:
        if type(arg) == str:
            jax_print_args.append(arg)
        else:
            jax_print_args.append("{}")
            jax_print_args.append(arg)
    jax.debug.print(*jax_print_args)


from urdfit.urdfit import *
from urdfit.param_optimizer import *
from urdfit.param_helper import *
