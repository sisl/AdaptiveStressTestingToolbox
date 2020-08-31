import fire
import numpy as np
# Useful imports
# from example_save_trials import *
import tensorflow as tf
# Import the necessary garage classes
from garage.experiment import run_experiment
from garage.np.baselines.zero_baseline import ZeroBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies.continuous_mlp_policy import ContinuousMLPPolicy

# Import the AST classes
import ast_toolbox.mcts.BoundedPriorityQueues as BPQ
from ast_toolbox.algos import GA
from ast_toolbox.algos import GASM
from ast_toolbox.envs import ASTEnv
from ast_toolbox.rewards import ExampleAVReward
from ast_toolbox.samplers import ASTVectorizedSampler
from ast_toolbox.simulators import ExampleAVSimulator
from ast_toolbox.spaces import ExampleAVSpaces


def runner(
    ga_type=None,
    env_args=None,
    run_experiment_args=None,
    sim_args=None,
    reward_args=None,
    spaces_args=None,
    policy_args=None,
    algo_args=None,
    runner_args=None,
    bpq_args=None,
    sampler_args=None,
    # log_dir='.',
):
    if ga_type is None:
        pass

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

    if algo_args is None:
        algo_args = {}

    if runner_args is None:
        runner_args = {}

    if sampler_args is None:
        sampler_args = {}

    if bpq_args is None:
        bpq_args = {}

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

        seed = 0
        # top_k = 10
        np.random.seed(seed)

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
                    env = ASTEnv(simulator=sim,
                                 reward_function=reward_function,
                                 spaces=spaces,
                                 **env_args
                                 )
                    env = TfEnv(env)

                    policy = ContinuousMLPPolicy(
                        name='ast_agent',
                        env_spec=env.spec,
                        **policy_args
                    )

                    params = policy.get_params()
                    sess.run(tf.variables_initializer(params))

                    # Instantiate the garage objects
                    baseline = ZeroBaseline(env_spec=env.spec)

                    top_paths = BPQ.BoundedPriorityQueue(**bpq_args)

                    sampler_cls = ASTVectorizedSampler
                    sampler_args['sim'] = sim
                    sampler_args['reward_function'] = reward_function

                    if ga_type == 'ga':
                        print('ga')
                        algo = GA(
                            env_spec=env.spec,
                            policy=policy,
                            baseline=baseline,
                            top_paths=top_paths,
                            **algo_args
                        )
                    elif ga_type == 'gasm':
                        print('gasm')
                        algo = GASM(
                            env_spec=env.spec,
                            policy=policy,
                            baseline=baseline,
                            top_paths=top_paths,
                            **algo_args
                        )
                    else:
                        raise NotImplementedError

                    local_runner.setup(algo=algo,
                                       env=env,
                                       sampler_cls=sampler_cls,
                                       sampler_args=sampler_args)

                    # Run the experiment
                    local_runner.train(**runner_args)

    # from garage.experiment.experiment import AttrDict
    # import os
    # tabular_log_file = os.path.join("./", "test.csv")
    # from dowel import logger, tabular
    # import dowel
    # logger.add_output(dowel.CsvOutput(tabular_log_file))
    # run_task(AttrDict(run_experiment_args))
    run_experiment(
        run_task,
        **run_experiment_args,
    )


if __name__ == '__main__':
    fire.Fire()
