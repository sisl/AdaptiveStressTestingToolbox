import pytest

# Import the example classes
from mylab.simulators.example_av_simulator import ExampleAVSimulator
from mylab.rewards.example_av_reward import ExampleAVReward
from mylab.spaces.example_av_spaces import ExampleAVSpaces
from garage.experiment import LocalRunner, run_experiment

# Import the AST classes
from mylab.envs.ast_env import ASTEnv
from mylab.samplers.ast_vectorized_sampler import ASTVectorizedSampler

# Import the necessary garage classes
from garage.tf.algos.trpo import TRPO
from garage.tf.envs.base import TfEnv
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.envs.normalized_env import normalize

# Useful imports
import os.path as osp
import argparse
from example_save_trials import *
import tensorflow as tf


batch_size = 4000
max_path_length = 50
n_envs = batch_size // max_path_length


def run_task(snapshot_config, *_):

    with LocalRunner(
            snapshot_config=snapshot_config, max_cpus=2) as runner:

        # Instantiate the example classes
        sim = ExampleAVSimulator()
        reward_function = ExampleAVReward()
        spaces = ExampleAVSpaces()

        # Create the environment
        env = TfEnv(normalize(ASTEnv(blackbox_sim_state=True,
                                     fixed_init_state=True,
                                     s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
                                     simulator=sim,
                                     reward_function=reward_function,
                                     spaces=spaces
                                     )))

        # Instantiate the garage objects
        policy = GaussianLSTMPolicy(name='lstm_policy',
                                    env_spec=env.spec,
                                    hidden_dim=64,
                                    use_peepholes=True)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=max_path_length,
            discount=0.99,
            kl_constraint='soft',
            max_kl_step=0.01)

        sampler_cls = ASTVectorizedSampler

        runner.setup(
            algo=algo,
            env=env,
            sampler_cls=sampler_cls,
            sampler_args={"sim": sim,
                          "reward_function": reward_function})

        runner.train(n_epochs=1, batch_size=4000, plot=False)

        print("Installation successfully validated")

def validate_parallel():
    run_experiment(run_task, snapshot_mode='last', seed=1, n_parallel=2)
    return True

if __name__ == '__main__':
    validate_parallel()


