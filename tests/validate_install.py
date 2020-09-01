# Import the example classes
from garage.envs.normalized_env import normalize
from garage.experiment import run_experiment
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
# Import the necessary garage classes
from garage.tf.algos.trpo import TRPO
from garage.tf.envs.base import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy

# Import the AST classes
from ast_toolbox.envs import ASTEnv
from ast_toolbox.rewards import ExampleAVReward
from ast_toolbox.samplers import ASTVectorizedSampler
from ast_toolbox.simulators import ExampleAVSimulator
from ast_toolbox.spaces import ExampleAVSpaces

# Useful imports


batch_size = 4000
max_path_length = 50
n_envs = batch_size // max_path_length


def run_task(snapshot_config, *_):

    with LocalTFRunner(
            snapshot_config=snapshot_config, max_cpus=1) as runner:

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
                                    hidden_dim=64)

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


def validate_install():
    run_experiment(run_task, snapshot_mode='last', seed=1, n_parallel=1, log_dir='./data/install')
    return True


if __name__ == '__main__':
    validate_install()
