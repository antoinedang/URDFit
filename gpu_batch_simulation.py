import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from simulation_parameters import *
from reward_functions import *
from jax.scipy.spatial.transform import Rotation
import gc
import random

### ADAPTED FROM MUJOCO SOURCE CODE https://github.com/google-deepmind/mujoco/blob/4ad1dc4b84b408ae585f5553fc4e78da7c95e530/mjx/mujoco/mjx/_src/io.py#L218
### ALLOWS US TO AVOID HAVING TO CALL get_data WHICH IS VERY SLOW
def getFootForces(m, pressure_sensor_ids, efc_J, efc_force, dist, geom1, geom2):
  ### ALL CODE BELOW WAS TAKEN FROM MUJOCO CODE, USED TO CALCULATE efc_address VALUES
  ne_connect = (m.eq_type == mujoco.mjtEq.mjEQ_CONNECT).sum()
  ne_weld = (m.eq_type == mujoco.mjtEq.mjEQ_WELD).sum()
  ne_joint = (m.eq_type == mujoco.mjtEq.mjEQ_JOINT).sum()
  ne = ne_connect * 3 + ne_weld * 6 + ne_joint

  nf = 0

  nl = int(m.jnt_limited.sum())

  nc = efc_J.shape[-2] - ne - nf - nl
  
  efc_type = jp.array([
      0,#mujoco.mjtConstraint.mjCNSTR_EQUALITY,
      1,#mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
      2,#mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT,
      3,#mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
  ]).repeat(jp.array([ne, nf, nl, nc]))
  
  efc_active = (efc_J != 0).any(axis=1)
  
  efc_con = efc_type == 3#mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
  
  nefc, nc = efc_active.sum(), (efc_active & efc_con).sum()
  
  efc_start = nefc - nc
    
  ncon = dist.shape[0]
      
  # efc_address = efc_start + jp.arange(0, ncon * 4, 4)
  efc_address = efc_start + jp.arange(0, ncon * 4, 4)
  ##### END OF MUJOCO SOURCE CODE ADAPTATION
  
  # using efc_address, we can count up the forces on the different pressure sensor geometries
  ordered_efc_forces = efc_force[efc_address]
  pressure_values = []
  for i in range(len(pressure_sensor_ids)):
    # makes the assumption there will be no more than _ contacts per pressure sensor per frame (size=_ argument)
    geom1_forces = ordered_efc_forces * jp.where(geom1 == pressure_sensor_ids[i], 1, 0)
    geom2_forces = ordered_efc_forces * jp.where(geom2 == pressure_sensor_ids[i], 1, 0)
    pressure_values.append(jp.sum(jp.abs(geom1_forces) + jp.abs(geom2_forces)))

  return jp.array(pressure_values)
    

def applyExternalForces(sim_batch):
  # check which simulations need new random force times/durations/magnitudes/directions/target bodies
  should_update_force = sim_batch.data_batch.time > (sim_batch.next_force_start_times + sim_batch.next_force_durations)
  
  # for the simulations which need to be updated, randomly generate new values
  if jp.any(should_update_force):
    N = jp.sum(should_update_force)
    updated_next_force_start_times = sim_batch.data_batch.time[should_update_force] + jax.random.uniform(key=sim_batch.rng_key, shape=(N,), minval=MIN_EXTERNAL_FORCE_INTERVAL, maxval=MAX_EXTERNAL_FORCE_INTERVAL)
    updated_next_force_durations = jax.random.uniform(key=sim_batch.rng_key, shape=(N,), minval=MIN_EXTERNAL_FORCE_DURATION, maxval=MAX_EXTERNAL_FORCE_DURATION)
    updated_next_force_magnitudes = jax.random.uniform(key=sim_batch.rng_key, shape=(N,), minval=MIN_EXTERNAL_FORCE_MAGNITUDE*sim_batch.randomization_factor, maxval=MAX_EXTERNAL_FORCE_MAGNITUDE*sim_batch.randomization_factor)
    updated_next_force_bodies = jax.random.randint(key=sim_batch.rng_key, shape=(N,), minval=1, maxval=len(sim_batch.data_batch.xfrc_applied) - 1)
    updated_next_force_directions = jax.random.ball(key=sim_batch.rng_key, d=2, shape=(N,))
    
    sim_batch.next_force_start_times.at[should_update_force].set(updated_next_force_start_times)
    sim_batch.next_force_durations.at[should_update_force].set(updated_next_force_durations)
    sim_batch.next_force_magnitudes.at[should_update_force].set(updated_next_force_magnitudes)
    sim_batch.next_force_bodies.at[should_update_force].set(updated_next_force_bodies)
    sim_batch.next_force_directions.at[should_update_force].set(updated_next_force_directions)

  # apply force values (times/durations/etc.) to corresponding simulations
  should_apply_force = jp.logical_and((sim_batch.data_batch.time > sim_batch.next_force_start_times), (sim_batch.data_batch.time < (sim_batch.next_force_start_times + sim_batch.next_force_durations)))
  
  if jp.any(should_apply_force):
    xfrc_applied = jp.zeros(sim_batch.data_batch.xfrc_applied.shape)
    applied_forces_x = sim_batch.next_force_directions[should_apply_force][0] * sim_batch.next_force_magnitudes[should_apply_force]
    applied_forces_y = sim_batch.next_force_directions[should_apply_force][1] * sim_batch.next_force_magnitudes[should_apply_force]
    xfrc_applied.at[should_apply_force][sim_batch.next_force_bodies[should_apply_force]][0].set(applied_forces_x)
    xfrc_applied.at[should_apply_force][sim_batch.next_force_bodies[should_apply_force]][1].set(applied_forces_y)
  
    sim_batch.data_batch.replace(xfrc_applied=xfrc_applied)

inverseRotateVectors = jax.jit(jax.vmap(lambda q, v : Rotation.from_quat(q).inv().apply(v)))

class GPUBatchSimulation:
  def __init__(self, count, xml_path, reward_fn, physics_steps_per_control_step=5, timestep=0.001, randomization_factor=0, verbose=False):
    if jax.default_backend() != 'gpu':
      print("ERROR: failed to find GPU device.")
      exit()
      
    self.xml_path = xml_path
    self.randomization_factor = randomization_factor
    self.timestep = timestep
    self.count = count
    self.reward_fn = jax.jit(jax.vmap(lambda v, z, q, jt : reward_fn(v, z, q, jt)))
    self.physics_steps_per_control_step = physics_steps_per_control_step
    self.rng_key = jax.random.PRNGKey(42)
    self.rng = jax.random.split(self.rng_key, self.count)
    self.verbose = verbose
    
    self.reset()
    
  def reset(self):    
    if self.verbose: print("Creating new simulations...")
    
    #load model from XML
    self.model = mujoco.MjModel.from_xml_path(self.xml_path)
    self.model.opt.timestep = self.timestep
    self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    self.model.opt.iterations = 3
    self.model.opt.ls_iterations = 5
    self.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

    #initialize instance parameters
    self.next_force_start_times = jp.zeros((self.count))
    self.next_force_durations = jp.zeros((self.count))
    self.next_force_magnitudes = jp.zeros((self.count))
    self.next_force_bodies = jp.zeros((self.count))
    self.next_force_directions = jp.zeros((self.count, 2))
    self.previous_torso_local_velocity = jp.zeros((self.count, 3))
    # save joint addresses
    self.joint_qpos_idx = []
    self.joint_torque_idx = []
    for joint in JOINT_NAMES:
      self.joint_torque_idx.append(self.model.jnt_dofadr[self.model.joint(joint).id])
      self.joint_qpos_idx.append(self.model.jnt_qposadr[self.model.joint(joint).id])
    self.joint_qpos_idx = jp.array(self.joint_qpos_idx)
    self.joint_torque_idx = jp.array(self.joint_torque_idx)
    # save gravity vector
    self.gravity_vector = self.model.opt.gravity
    # save torso body index
    self.torso_idx = self.model.body(TORSO_BODY_NAME).id
    # get pressure sensor geom ids
    self.pressure_sensor_ids = [self.model.geom(pressure_sensor_geom).id for pressure_sensor_geom in PRESSURE_GEOM_NAMES]
    
    # RANDOMIZATION
    # floor friction (0.5 to 1.0)
    self.model.geom('floor').friction = [coef * (1.0*(1.0-self.randomization_factor) + random.uniform(FLOOR_FRICTION_MIN_MULTIPLIER, FLOOR_FRICTION_MAX_MULTIPLIER)*self.randomization_factor) for coef in self.model.geom('floor').friction]    
    #delays in actions and observations (10ms to 50ms)
    self.action_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.observation_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    #round delays to be multiples of the timestep
    actual_timestep = self.timestep * self.physics_steps_per_control_step
    self.observation_delay = round(self.observation_delay / actual_timestep) * actual_timestep
    self.action_delay = round(self.action_delay / actual_timestep) * actual_timestep
    #make buffers for observations and actions
    self.observation_buffer = [None] * (int)(self.observation_delay/actual_timestep)
    self.action_buffer = [None] * (int)(self.action_delay/actual_timestep)
    # vary the mass of all limbs randomly
    for i in range(self.model.nbody-1): self.model.body(i+1).mass[0] += random.uniform(-MAX_MASS_CHANGE_PER_LIMB*self.randomization_factor, MAX_MASS_CHANGE_PER_LIMB*self.randomization_factor)
    # attach a random external mass (up to 0.1 kg) to a randomly chosen limb
    self.model.body(random.randint(1, self.model.nbody - 1)).mass[0] += random.uniform(0, MAX_EXTERNAL_MASS_ADDED*self.randomization_factor)
    # randomize IMU Z
    self.imu_z_offset = jax.random.uniform(key=self.rng_key, shape=(self.count,), minval=-IMU_Z_OFFSET_MAX, maxval=IMU_Z_OFFSET_MAX)
    # randomize joint properties  
    for joint in JOINT_NAMES:
      self.model.joint(joint).damping[0] += random.uniform(-JOINT_DAMPING_MAX_CHANGE, JOINT_DAMPING_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).armature[0] += random.uniform(0, JOINT_ARMATURE_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).stiffness[0] += random.uniform(0, JOINT_STIFFNESS_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).margin[0] += random.uniform(0, JOINT_MARGIN_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).range[0] += random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).range[1] += random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)*self.randomization_factor
    for joint in JOINT_ACTUATOR_NAMES:
      self.model.actuator(joint).forcerange[0] += random.uniform(-JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE)*self.randomization_factor
      self.model.actuator(joint).forcerange[1] += random.uniform(-JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE)*self.randomization_factor
      kp = max(0, JOINT_PID_P_GAIN + random.uniform(-JOINT_PID_GAIN_MAX_CHANGE, JOINT_PID_GAIN_MAX_CHANGE)*self.randomization_factor)
      kv = max(0, JOINT_PID_V_GAIN + random.uniform(-JOINT_PID_GAIN_MAX_CHANGE, JOINT_PID_GAIN_MAX_CHANGE)*self.randomization_factor)
      self.model.actuator(joint).gainprm[0] = kp
      self.model.actuator(joint).biasprm[1] = -kp
      self.model.actuator(joint).biasprm[2] = -kv

    # create MJX model/data from CPU model/data
    self.cpu_model = self.model
    self.model = mjx.put_model(self.cpu_model)
    mjx_data = mjx.put_data(self.cpu_model, mujoco.MjData(self.cpu_model))
    
    #define contact force function
    self.getFootForces = jax.vmap(lambda efc_J, efc_force, dist, geom1, geom2 : getFootForces(self.model, self.pressure_sensor_ids, efc_J, efc_force, dist, geom1, geom2))
    
    # define step function (rollout)
    def rollout(m, d):
      for _ in range(self.physics_steps_per_control_step):
        d = mjx.step(m, d)
      return d
    jax_rollout = jax.jit(rollout)
    self.jax_step = jax.vmap(jax_rollout, in_axes=(None, 0))
    
    # randomize joint initial states (GPU)
    self.data_batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, mjx_data.qpos.shape, minval=-JOINT_INITIAL_STATE_OFFSET_MAX/180.0*jp.pi, maxval=JOINT_INITIAL_STATE_OFFSET_MAX/180.0*jp.pi)))(self.rng)

    # step sim (to populate self.data)
    self.step()
    
    # clean up any unreferenced variables
    gc.collect()
    
    if self.verbose: print("Simulations initialized.")

  def computeReward(self):
    if self.verbose: print("Computing rewards...")
    
    torso_global_velocity = self.data_batch.cvel[:, self.torso_idx][:, 3:]
    torso_z_pos = self.data_batch.xpos[:, self.torso_idx, 2]
    torso_z_pos += self.imu_z_offset
    torso_quat = self.data_batch.xquat[:, self.torso_idx]
    joint_torques = self.data_batch.qfrc_constraint[:, self.joint_torque_idx] + self.data_batch.qfrc_smooth[:, self.joint_torque_idx]
    
    rewards = self.reward_fn(torso_global_velocity, torso_z_pos, torso_quat, joint_torques)
    
    if self.verbose: print("Rewards computed.")

    return rewards
    
  def getObs(self):
    if self.verbose: print("Collecting observations...")
    
    torso_quat = self.data_batch.xquat[:, self.torso_idx]
    torso_global_vel = self.data_batch.cvel[:, self.torso_idx]
    
    # joint positions     20          Joint positions in radians
    joint_angles = self.data_batch.qpos[:, self.joint_qpos_idx] + ((JOINT_ANGLE_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=(self.count, len(self.joint_qpos_idx))))
    
    # angular velocity    3           Angular velocity (roll, pitch, yaw) from IMU
    torso_global_ang_vel = torso_global_vel[:, 0:3]
    local_ang_vel = inverseRotateVectors(torso_quat, torso_global_ang_vel) + (GYRO_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.count, 3)))
    # agent velocity      2           X and Y velocity of robot torso
    torso_global_velocity = torso_global_vel[:, 3:] + (VELOCIMETER_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.count, 3)))
    # linear acceleration 3           Linear acceleration from IMU
    torso_local_velocity = inverseRotateVectors(torso_quat, torso_global_velocity)
    torso_local_accel = ((torso_local_velocity - self.previous_torso_local_velocity)/(self.timestep * self.physics_steps_per_control_step)) + (ACCELEROMETER_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.count, 3)))
    self.previous_torso_local_velocity = torso_local_velocity
    # gravity             3           Gravity direction, derived from angular velocity using Madgwick filter
    noisy_torso_quat = torso_quat + ((IMU_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=(self.count, 4)))
    local_gravity_vector = inverseRotateVectors(noisy_torso_quat, jp.array([self.gravity_vector]*self.count))
    # foot pressure       8           Pressure values from foot sensors
    pressure_values = self.getFootForces(self.data_batch.efc_J, self.data_batch.efc_force, self.data_batch.contact.dist, self.data_batch.contact.geom1, self.data_batch.contact.geom2)

    observations = jp.hstack((joint_angles, local_ang_vel, torso_global_velocity[:, 0:2], torso_local_accel, local_gravity_vector, pressure_values))
  
    # cycle observation through observation buffer
    self.observation_buffer.append(observations)
    delayed_observations = self.observation_buffer.pop(0)
    
    if self.verbose: print("Observations collected.")
    
    return delayed_observations
  
  def step(self, action=None):
    if self.verbose: print("Stepping simulations...")
    
    # cycle action through action buffer
    self.action_buffer.append(action)
    action_to_take = self.action_buffer.pop(0)
    if action_to_take is not None:
      self.data_batch.replace(ctrl=jp.array(action_to_take))
    
    # apply forces to the robot to destabilise it
    applyExternalForces(self)

    # step sims, update data batch
    self.data_batch = self.jax_step(self.model, self.data_batch)
    
    if self.verbose: print("Simulations stepped.")

if __name__ == "__main__":
    sim_batch = GPUBatchSimulation(count=512,
                                   xml_path="assets/world.xml",
                                   reward_fn=standingRewardFn,
                                   physics_steps_per_control_step=5,
                                   timestep=0.005,
                                   randomization_factor=1,
                                   verbose=True)

    while True:
      while all(sim_batch.data_batch.time < 2):
        observations = sim_batch.getObs()
        actions = [[0]*4]*sim_batch.count
        sim_batch.step(actions)
        rewards = sim_batch.computeReward()
        print(rewards[0])
      sim_batch.reset()