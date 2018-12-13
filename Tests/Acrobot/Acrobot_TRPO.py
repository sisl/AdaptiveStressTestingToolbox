import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from garage.tf.algos.trpo import TRPO
from garage.baselines.linear_feature_baseline import LinearFeatureBaseline
from Acrobot.acrobot import AcrobotEnv
from garage.tf.envs.base import TfEnv
from garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy

from garage.misc import logger
import os.path as osp
import tensorflow as tf
import joblib

log_dir = "Data/Train"

tabular_log_file = osp.join(log_dir, "progress.csv")
text_log_file = osp.join(log_dir, "debug.log")
params_log_file = osp.join(log_dir, "params.json")
pkl_file = osp.join(log_dir, "params.pkl")

logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode("gap")
logger.set_snapshot_gap(10)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % "Acrobot-RL")

max_path_length = 400
env = TfEnv(AcrobotEnv(success_reward = max_path_length,
                        success_threshhold = 1.9999,))

# policy = CategoricalMLPPolicy(
#     name='protagonist',
#     env_spec=env.spec,
#     hidden_sizes=(64, 32),
# )

policy = GaussianMLPPolicy(
    name='protagonist',
    env_spec=env.spec,
    hidden_sizes=(128, 64, 32),
    output_nonlinearity=tf.nn.tanh,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=max_path_length,
    n_itr=51,
    discount=0.99,
    step_size=0.01,
    plot=True,
)
algo.train()