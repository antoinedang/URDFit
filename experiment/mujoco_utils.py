import numpy as np
import time
import cv2
import mujoco
import pathlib
from mujoco_mpc import agent as agent_lib


# Function to render current state of mujoco (for debugging)
def render(renderer, mj_data, display=True):
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    renderer.update_scene(mj_data, camera="track", scene_option=scene_option)
    frame = renderer.render()
    # time.sleep(1 / CONTROL_FREQUENCY)
    if display:
        cv2.imshow("CPU Sim View", frame)
        cv2.waitKey(1)
    else:
        return frame


def get_mujoco_setup(model_path, task_id, timestep):
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_model.opt.timestep = timestep
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, 720, 1080)
    agent = agent_lib.Agent(task_id=task_id, model=mj_model)

    return mj_model, mj_data, renderer, agent


def compute_action(agent, mj_data, planning_horizon):
    agent.set_state(
        time=mj_data.time,
        qpos=mj_data.qpos,
        qvel=mj_data.qvel,
        act=mj_data.act,
        mocap_pos=mj_data.mocap_pos,
        mocap_quat=mj_data.mocap_quat,
        userdata=mj_data.userdata,
    )

    # run planner for planning_horizon
    for _ in range(planning_horizon):
        agent.planner_step()

    # get ctrl from agent policy
    torque_ctrl = agent.get_action()
    return torque_ctrl
