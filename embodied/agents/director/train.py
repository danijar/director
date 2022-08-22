import pathlib
import sys
import warnings

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')

directory = pathlib.Path(__file__)
try:
  import google3  # noqa
except ImportError:
  directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied


def main(argv=None):
  from . import agent as agnt
  from . import train_with_viz

  parsed, other = embodied.Flags(
      configs=['defaults'], actor_id=0, actors=0,
  ).parse_known(argv)
  config = embodied.Config(agnt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agnt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)

  config = config.update(logdir=str(embodied.Path(config.logdir)))
  args = embodied.Config(logdir=config.logdir, **config.train)
  args = args.update(expl_until=args.expl_until // config.env.repeat)
  print(config)

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  cleanup = []

  if config.run == 'acting':
    actordir = logdir / f'actor{parsed.actor_id}'
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(actordir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(actordir),
    ], multiplier=config.env.repeat * parsed.actors)
  else:
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
    ], multiplier=config.env.repeat)

  chunk = config.replay_chunk
  if config.replay == 'fixed':
    def make_replay(name, capacity):
      directory = logdir / name
      store = embodied.replay.CkptRAMStore(directory, capacity, parallel=True)
      cleanup.append(store)
      return embodied.replay.FixedLength(store, chunk, **config.replay_fixed)
  elif config.replay == 'consec':
    def make_replay(name, capacity):
      directory = logdir / name
      store = embodied.replay.CkptRAMStore(directory, capacity, parallel=True)
      cleanup.append(store)
      return embodied.replay.Consecutive(store, chunk, **config.replay_consec)
  elif config.replay == 'prio':
    def make_replay(name, capacity):
      directory = logdir / name
      store = embodied.replay.CkptRAMStore(directory, capacity, parallel=True)
      cleanup.append(store)
      return embodied.replay.Prioritized(store, chunk, **config.replay_prio)
  else:
    raise NotImplementedError(config.replay)

  try:
    config = config.update({'env.seed': hash((config.seed, parsed.actor_id))})
    env = embodied.envs.load_env(
        config.task, mode='train', logdir=logdir, **config.env)
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    if config.run == 'train':
      replay = make_replay('episodes', config.replay_size)
      embodied.run.train(agent, env, replay, logger, args)
    elif config.run == 'train_eval':
      eval_env = embodied.envs.load_env(
          config.task, mode='eval', logdir=logdir, **config.env)
      eval_replay = make_replay('eval_episodes', config.replay_size // 10)
      replay = make_replay('episodes', config.replay_size)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args)
      cleanup.append(eval_env)
    elif config.run == 'train_with_viz':
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config.eval_dir, config.replay_size // 10)
      else:
        assert config.train.eval_fill
        eval_replay = make_replay('eval_episodes', config.replay_size // 10)
      replay = make_replay('episodes', config.replay_size)
      train_with_viz.train_with_viz(
          agent, env, replay, eval_replay, logger, args)
    elif config.run == 'learning':
      assert config.replay.sync
      env.close()
      replay = make_replay('episodes', config.replay_size)
      eval_replay = (
          make_replay(config.eval_dir, config.replay_size // 10)
          if config.eval_dir else replay)
      embodied.run.learning(agent, replay, eval_replay, logger, args)
    elif config.run == 'acting':
      replay = make_replay('episodes', args.train_fill)
      embodied.run.acting(agent, env, replay, logger, actordir, args)
    else:
      raise NotImplementedError(config.run)
  finally:
    for obj in cleanup:
      obj.close()


if __name__ == '__main__':
  main()
