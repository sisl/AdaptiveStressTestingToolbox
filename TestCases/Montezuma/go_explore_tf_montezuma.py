#!/usr/bin/env python3

import gym


from garage.np.baselines import LinearFeatureBaseline
from mylab.algos.go_explore import GoExplore
from garage.tf.envs import TfEnv
from  mylab.envs.go_explore_env import Pixel_GoExploreEnv, Ram_GoExploreEnv
from mylab.policies.go_explore_policy import GoExplorePolicy
from mylab.samplers.batch_sampler import BatchSampler

import fire
import os
import numpy as np
from skimage.measure import block_reduce
from garage.misc.overrides import overrides
from garage.experiment import LocalRunner, run_experiment, SnapshotConfig
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
import garage.tf.core.layers as L
from garage.tf.envs import TfEnv
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import CategoricalLSTMPolicy

# class Pixel_GoExploreEnv(GoExploreTfEnv):
#     @overrides
#     def downsample(self, obs):
#         # import pdb; pdb.set_trace()
#         obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
#         obs = block_reduce(obs, block_size=(20, 20), func=np.mean)
#         obs= obs.astype(np.uint8) // 32
#         return obs.flatten()
#
# class Ram_GoExploreEnv(GoExploreTfEnv):
#     @overrides
#     def downsample(self, obs):
#         # import pdb; pdb.set_trace()
#         return obs // 32

def runner(exp_name='montezuma',
           use_ram=False,
           db_filename='/home/mkoren/Scratch/cellpool-shelf.dat',
           max_db_size=150,
           overwrite_db=True,
           n_parallel=2,
           snapshot_mode='last',
           snapshot_gap=1,
           log_dir=None,
           max_path_length=100,
           discount=0.99,
           n_itr=100,
           max_kl_step=0.01):

    if overwrite_db and os.path.exists(db_filename):
        os.remove(db_filename)

    batch_size = max_path_length * n_parallel

    def run_task(snapshot_config, *_):
        with LocalRunner(snapshot_config=snapshot_config) as runner:
            # gym_env=gym.make('MontezumaRevenge-ram-v0')
            if use_ram:
                # gym_env = gym.make('MontezumaRevenge-ram-v0')
                # import pdb; pdb.set_trace()
                env = Ram_GoExploreEnv(env_name='MontezumaRevenge-ram-v0')
                # env = GoExploreTfEnv(env=gym_env)
                # pool=CellPool())
                # setattr(env, 'downsampler', ram_downsampler)
            else:
                # gym_env = gym.make('MontezumaRevenge-v0')
                # import pdb; pdb.set_trace()
                env = Pixel_GoExploreEnv(env_name='MontezumaRevenge-v0')
                # env = GoExploreTfEnv(env=gym_env)
                #                      # pool=CellPool())
                # setattr(env, 'downsampler',pixel_downsampler)

            policy = GoExplorePolicy(
                env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)



            algo = GoExplore(
                db_filename=db_filename,
                max_db_size=max_db_size,
                env=env,
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=max_path_length,
                discount=discount,
                )
            # algo.train()
            #setup(self, algo, env, sampler_cls=None, sampler_args=None):
            sampler_cls = BatchSampler
            sampler_args = {'n_envs': n_parallel}

            runner.setup(algo=algo,
                         env=env,
                         sampler_cls=sampler_cls,
                         sampler_args=sampler_args)
            runner.train(n_epochs=n_itr, batch_size=batch_size)

            # runner.setup(algo, env, sampler_args={'n_envs': 1})
            # runner.train(n_epochs=120, batch_size=5000,store_paths=False)


    run_experiment(
        run_task,
        snapshot_mode=snapshot_mode,
        log_dir=log_dir,
        exp_name=exp_name,
        snapshot_gap=snapshot_gap,
        seed=1,
        n_parallel=n_parallel,
    )


if __name__ == '__main__':
  fire.Fire()
