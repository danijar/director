import functools

import embodied


def load_env(
    task, amount=1, parallel='none', daemon=False, restart=False, seed=None,
    **kwargs):
  ctors = []
  for index in range(amount):
    ctor = functools.partial(load_single_env, task, **kwargs)
    if seed is not None:
      ctor = functools.partial(ctor, seed=hash((seed, index)) % (2 ** 31 - 1))
    if parallel != 'none':
      ctor = functools.partial(embodied.Parallel, ctor, parallel, daemon)
    if restart:
      ctor = functools.partial(embodied.wrappers.RestartOnException, ctor)
    ctors.append(ctor)
  envs = [ctor() for ctor in ctors]
  return embodied.BatchEnv(envs, parallel=(parallel != 'none'))


def load_single_env(
    task, size=(64, 64), repeat=1, mode='train', camera=-1, gray=False,
    length=0, logdir='/dev/null', discretize=0, sticky=True, lives=False,
    episodic=True, again=False, termination=False, weaker=1.0, checks=False,
    seed=None):
  suite, task = task.split('_', 1)
  if suite == 'dummy':
    from . import dummy
    env = dummy.Dummy(task, size, length or 100)
  elif suite == 'gym':
    from . import gym
    env = gym.Gym(task)
  elif suite == 'bsuite':
    import bsuite
    from . import dmenv
    env = bsuite.load_from_id(task)
    env = dmenv.DMEnv(env)
    env = embodied.wrappers.FlattenTwoDimObs(env)
  elif suite == 'dmc':
    from . import dmc
    env = dmc.DMC(task, repeat, size, camera)
  elif suite == 'atari':
    from . import atari
    env = atari.Atari(task, repeat, size, gray, lives=lives, sticky=sticky)
  elif suite == 'crafter':
    from . import crafter
    assert repeat == 1
    # outdir = embodied.Path(logdir) / 'crafter' if mode == 'train' else None
    outdir = None
    env = crafter.Crafter(task, size, outdir)
  elif suite == 'dmlab':
    from . import dmlab
    env = dmlab.DMLab(task, repeat, size, mode, seed=seed, episodic=episodic)
  elif suite == 'robodesk':
    from . import robodesk
    env = robodesk.RoboDesk(task, mode, repeat, length or 2000)
  elif suite == 'minecraft':
    from . import minecraft
    env = minecraft.Minecraft(task, repeat, size, length or 24000)
  elif suite == 'loconav':
    from . import loconav
    env = loconav.LocoNav(
        task, repeat, size, camera,
        again=again, termination=termination, weaker=weaker)
  elif suite == 'pinpad':
    from . import pinpad
    assert repeat == 1
    assert size == (64, 64)
    env = pinpad.PinPad(task, length or 2000)
  else:
    raise NotImplementedError(suite)
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    if space.discrete:
      env = embodied.wrappers.OneHotAction(env, name)
    elif discretize:
      env = embodied.wrappers.DiscretizeAction(env, name, discretize)
    else:
      env = embodied.wrappers.NormalizeAction(env, name)
  if length:
    env = embodied.wrappers.TimeLimit(env, length)
  env = embodied.wrappers.ExpandScalars(env)
  if checks:
    env = embodied.wrappers.CheckSpaces(env)
  return env


__all__ = [
    k for k, v in list(locals().items())
    if type(v).__name__ in ('type', 'function') and not k.startswith('_')]
