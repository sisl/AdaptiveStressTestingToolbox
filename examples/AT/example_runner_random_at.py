# Import the example classes
import fire
# Useful imports
import tensorflow as tf
from garage.envs.normalized_env import normalize
from garage.experiment import run_experiment
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
# Import the necessary garage classes
from garage.tf.envs.base import TfEnv
from garage.tf.experiment import LocalTFRunner

# Import the AST classes
import ast_toolbox.mcts.BoundedPriorityQueues as BPQ
from ast_toolbox.algos.random_search import RandomSearch
from ast_toolbox.policies.random_policy import RandomPolicy
from ast_toolbox.envs import ASTEnv
from ast_toolbox.rewards import ExampleATReward
from ast_toolbox.samplers import ASTVectorizedSampler
from ast_toolbox.simulators import ExampleATSimulator
from ast_toolbox.spaces import ExampleATSpaces


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
    bpq_args=None,
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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.variable_scope('AST', reuse=tf.AUTO_REUSE):

                with LocalTFRunner(
                        snapshot_config=snapshot_config, max_cpus=4, sess=sess) as local_runner:
                    # Instantiate the example classes
                    sim = ExampleATSimulator(**sim_args)
                    reward_function = ExampleATReward(**reward_args)
                    spaces = ExampleATSpaces(**spaces_args)

                    # Create the environment
                    if 'id' in env_args:
                        env_args.pop('id')
                    env = TfEnv(ASTEnv(simulator=sim,
                                       reward_function=reward_function,
                                       spaces=spaces,
                                       **env_args
                                       ))

                    # Instantiate the garage objects
                    policy = RandomPolicy(
                                env_spec=env.spec,
                                name='ast_agent',
                                **policy_args
                                )

                    baseline = LinearFeatureBaseline(env_spec=env.spec, **baseline_args)

                    top_paths = BPQ.BoundedPriorityQueue(**bpq_args)
                    algo = RandomSearch(
                        # env=env,
                        env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        # batch_size=batch_size,
                        # store_paths=True,
                        # top_paths=top_paths,
                        # plot=False,
                        **algo_args
                    )

                    sampler_cls = ASTVectorizedSampler

                    local_runner.setup(
                        algo=algo,
                        env=env,
                        sampler_cls=sampler_cls,
                        sampler_args={"open_loop": False,
                                      "sim": sim,
                                      "reward_function": reward_function,
                                      'n_envs': n_parallel})

                    # Run the experiment
                    local_runner.train(**runner_args)

    run_experiment(
        run_task,
        **run_experiment_args,
    )


if __name__ == '__main__':
    fire.Fire()

