# Import the example classes
import os

import fire
# Useful imports
import tensorflow as tf
from garage.envs.normalized_env import normalize
from garage.experiment import run_experiment
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
# Import the necessary garage classes
from garage.tf.algos.ppo import PPO
from garage.tf.envs.base import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from garage.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
# from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.policies import GaussianLSTMPolicy

# Import the AST classes
from ast_toolbox.envs import ASTEnv
from ast_toolbox.rewards import ExampleAVReward
from ast_toolbox.samplers import ASTVectorizedSampler
from ast_toolbox.simulators import ExampleAVSimulator
from ast_toolbox.spaces import ExampleAVSpaces
from ast_toolbox.utils.go_explore_utils import load_convert_and_save_drl_expert_trajectory


def runner(
    env_args=None,
    run_experiment_args=None,
    sim_args=None,
    reward_args=None,
    spaces_args=None,
    policy_args=None,
    baseline_args=None,
    algo_args=None,
    runner_args=None,
    sampler_args=None,
    save_expert_trajectory=False,
):

    if env_args is None:
        env_args = {}

    if run_experiment_args is None:
        run_experiment_args = {}

    if sim_args is None:
        sim_args = {}

    if reward_args is None:
        reward_args = {}

    if spaces_args is None:
        spaces_args = {}

    if policy_args is None:
        policy_args = {}

    if baseline_args is None:
        baseline_args = {}

    if algo_args is None:
        algo_args = {}

    if runner_args is None:
        runner_args = {'n_epochs': 1}

    if sampler_args is None:
        sampler_args = {}

    if 'n_parallel' in run_experiment_args:
        n_parallel = run_experiment_args['n_parallel']
    else:
        n_parallel = 1
        run_experiment_args['n_parallel'] = n_parallel

    if 'max_path_length' in sim_args:
        max_path_length = sim_args['max_path_length']
    else:
        max_path_length = 50
        sim_args['max_path_length'] = max_path_length

    if 'batch_size' in runner_args:
        batch_size = runner_args['batch_size']
    else:
        batch_size = max_path_length * n_parallel
        runner_args['batch_size'] = batch_size

    def run_task(snapshot_config, *_):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.variable_scope('AST', reuse=tf.AUTO_REUSE):

                with LocalTFRunner(
                        snapshot_config=snapshot_config, max_cpus=4, sess=sess) as local_runner:
                    # Instantiate the example classes
                    sim = ExampleAVSimulator(**sim_args)
                    reward_function = ExampleAVReward(**reward_args)
                    spaces = ExampleAVSpaces(**spaces_args)

                    # Create the environment
                    if 'id' in env_args:
                        env_args.pop('id')
                    env = TfEnv(normalize(ASTEnv(simulator=sim,
                                                 reward_function=reward_function,
                                                 spaces=spaces,
                                                 **env_args
                                                 )))

                    # Instantiate the garage objects
                    policy = GaussianLSTMPolicy(env_spec=env.spec, **policy_args)

                    baseline = LinearFeatureBaseline(env_spec=env.spec, **baseline_args)

                    optimizer = ConjugateGradientOptimizer
                    optimizer_args = {'hvp_approach': FiniteDifferenceHvp(base_eps=1e-5)}

                    algo = PPO(env_spec=env.spec,
                               policy=policy,
                               baseline=baseline,
                               optimizer=optimizer,
                               optimizer_args=optimizer_args,
                               **algo_args)

                    sampler_cls = ASTVectorizedSampler
                    sampler_args['sim'] = sim
                    sampler_args['reward_function'] = reward_function

                    local_runner.setup(
                        algo=algo,
                        env=env,
                        sampler_cls=sampler_cls,
                        sampler_args=sampler_args)

                    # Run the experiment
                    local_runner.train(**runner_args)

                    if save_expert_trajectory:
                        load_convert_and_save_drl_expert_trajectory(last_iter_filename=os.path.join(run_experiment_args['log_dir'],
                                                                                                    'itr_' +
                                                                                                    str(runner_args['n_epochs'] - 1) +
                                                                                                    '.pkl'),
                                                                    expert_trajectory_filename=os.path.join(run_experiment_args['log_dir'],
                                                                                                            'expert_trajectory.pkl'))

                    print('done!')

    run_experiment(
        run_task,
        **run_experiment_args,
    )


if __name__ == '__main__':
    fire.Fire()
