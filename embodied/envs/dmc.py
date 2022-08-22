import os

import embodied
import numpy as np


class DMC(embodied.Env):

  # TODO: Simplify using DMEnv.

  DEFAULT_CAMERAS = dict(
      locom_rodent=1,
      quadruped=2,
  )

  def __init__(self, name, repeat=1, size=(64, 64), camera=-1):
    # TODO: This env variable is necessary when running on a headless GPU but
    # breaks when running on a CPU machine.
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    if not isinstance(name, str):
      self._env = name
    else:
      domain, task = name.split('_', 1)
      if task.endswith('hz'):
        task, freq = task.rsplit('_', 1)
        freq = float(freq.strip('hz'))
        patch_control_frequency(freq)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        from dm_control import manipulation
        self._env = manipulation.load(task + '_vision')
      elif domain == 'locom':
        from dm_control.locomotion.examples import basic_rodent_2020
        self._env = getattr(basic_rodent_2020, task)()
      else:
        from dm_control import suite
        self._env = suite.load(domain, task)
    self._repeat = repeat
    self._size = size
    self._camera = camera
    self._ignored_keys = []
    for key, value in self._env.observation_spec().items():
      if value.shape == (0,):
        print(f"Ignoring empty observation key '{key}'.")
        self._ignored_keys.append(key)
    self._done = True

  @property
  def obs_space(self):
    spaces = {
        'image': embodied.Space(np.uint8, self._size + (3,)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }
    for key, value in self._env.observation_spec().items():
      if key in self._ignored_keys:
        continue
      shape = (1,) if value.shape == () else value.shape
      if np.issubdtype(value.dtype, np.floating):
        spaces[key] = embodied.Space(np.float32, shape)
      elif np.issubdtype(value.dtype, np.uint8):
        spaces[key] = embodied.Space(np.uint8, shape)
      else:
        raise NotImplementedError(value.dtype)
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    return {
        'action': embodied.Space(np.float32, None, spec.minimum, spec.maximum),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      time_step = self._env.reset()
      self._done = False
      return self._obs(time_step, 0.0)
    assert np.isfinite(action['action']).all(), action['action']
    reward = 0.0
    for _ in range(self._repeat):
      time_step = self._env.step(action['action'])
      reward += time_step.reward or 0.0
      if time_step.last():
        break
    assert time_step.discount in (0, 1)
    self._done = time_step.last()
    return self._obs(time_step, reward)

  def _obs(self, time_step, reward):
    obs = {
        k: v[None] if v.shape == () else v
        for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys}
    return dict(
        reward=reward,
        is_first=time_step.first(),
        is_last=time_step.last(),
        is_terminal=time_step.discount == 0,
        image=self.render(),
        **obs,
    )

  def render(self):
    return self._env.physics.render(*self._size, camera_id=self._camera)


def patch_control_frequency(freq):
  print(f'Warning: Overwriting control frequency of all DMC envs to {freq}.')
  from dm_control import mujoco
  from dm_control.rl import control

  class Physics(mujoco.Physics):
    @classmethod
    def from_xml_string(cls, xml_string, assets=None):
      import re
      xml_string = re.sub(
          r'timestep="[0-9.]+"',
          r'timestep="0.001"',
          xml_string.decode('utf-8'),
      ).encode('utf-8')
      assert '<option timestep="0.001"' in xml_string.decode('utf-8'), (
          xml_string)
      return super().from_xml_string(xml_string, assets)
  mujoco.Physics = Physics

  class Environment(control.Environment):
    def __init__(
        self, physics, task, time_limit=float('inf'),
        control_timestep=None, n_sub_steps=None,
        flat_observation=False):
      control_timestep = 1 / freq
      time_limit = 1000 * control_timestep
      super().__init__(
          physics, task, time_limit, control_timestep, n_sub_steps,
          flat_observation)
  control.Environment = Environment
