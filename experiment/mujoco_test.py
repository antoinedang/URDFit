import time
from urdfit import URDFit, ParamHelper
from mujoco_utils import *

if __name__ == "__main__":
    xml_path = "_archive/mujoco_mpc/mjpc/tasks/manipulation/task_panda_bring.xml"
    task_id = "PickAndPlace"
    timestep = 0.01
    RENDER = False

    mj_model, mj_data, renderer, agent = get_mujoco_setup(xml_path, task_id, timestep)
    urdfit = URDFit(xml_path)

    # rollout
    mujoco.mj_resetData(mj_model, mj_data)
    while True:
        start_time = time.time()

        # run planner for num_steps
        action = compute_action(agent, mj_data, 10)
        mj_data.ctrl = action

        state_q = mj_data.qpos
        state_qd = mj_data.qvel
        mujoco.mj_step(mj_model, mj_data)
        next_state_q = mj_data.qpos
        next_state_qd = mj_data.qvel

        urdfit.step(
            state_q,
            state_qd,
            action,
            next_state_q,
            next_state_qd,
        )

        optimized_params = urdfit.get_optimized_params()

        # render
        if RENDER:
            render(renderer, mj_data)
        else:
            end_time = time.time()
            control_time = end_time - start_time
            print(f"Control freq.: {1.0 / control_time}")
