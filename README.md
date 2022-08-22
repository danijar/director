Deep Hierarchical Planning from Pixels
======================================

Official implementation of the [Director][project] algorithm in TensorFlow 2.

[project]: https://danijar.com/director/

![Director Internal Goals](https://github.com/danijar/director/raw/main/media/header.gif)

If you find this code useful, please reference in your paper:

```
@article{hafner2022director,
  title={Deep Hierarchical Planning from Pixels},
  author={Hafner, Danijar and Lee, Kuang-Huei and Fischer, Ian and Abbeel, Pieter},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

How does Director work?
-----------------------

Director is a practical and robust algorithm for hierarchical reinforcement
learning. To solve long horizon tasks end-to-end from sparse rewards, Director
learns to break down tasks into internal subgoals. Its manager policy selects
subgoals that trade off exploratory and extrinsic value, its worker policy
learns to achieve the goals through low-level actions. Both policies are
trained from imagined trajectories predicted by a learned world model. To
support the manager in choosing realistic goals, a goal autoencoder compresses
and quantizes previously encountered representations. The manager chooses its
goals in this compact space. All components are trained concurrently.

![Director Method Diagram](https://github.com/danijar/director/raw/main/media/method.png)

For more information:

- [Google AI Blog](https://ai.googleblog.com/2022/07/deep-hierarchical-planning-from-pixels.html)
- [Project website](https://danijar.com/project/director/)
- [Research paper](https://arxiv.org/pdf/2206.04114.pdf)

Running the Agent
-----------------

Either use `embodied/Dockerfile` or follow the manual instructions below.

Install dependencies:

```sh
pip install -r requirements.txt
```

Train agent:

```sh
python embodied/agents/director/train.py \
  --logdir ~/logdir/$(date +%Y%m%d-%H%M%S) \
  --configs dmc_vision \
  --task dmc_walker_walk
```

See `agents/director/configs.yaml` for available flags and
`embodied/envs/__init__.py` for available envs.

Using the Tasks
---------------

The HRL environments are in `embodied/envs/pinpad.py` and
`embodied/envs/loconav.py`.
