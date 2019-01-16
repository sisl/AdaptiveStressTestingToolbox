import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from CartPoleAST.CartPole.cartpole import CartPoleEnv
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

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
logger.push_prefix("[%s] " % "Carpole-RL")

env = TfEnv(CartPoleEnv(use_seed=False))

policy = CategoricalMLPPolicy(
    name='protagonist',
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = ZeroBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=51,
    discount=0.99,
    step_size=0.01,
)
algo.train()