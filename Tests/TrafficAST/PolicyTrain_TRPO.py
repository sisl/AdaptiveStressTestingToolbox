import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from garage.misc import logger
import os.path as osp
import tensorflow as tf
import numpy as np
import joblib

seed = 1
log_dir = "Data/Train/TRPO/seed{}".format(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

tabular_log_file = osp.join(log_dir, "progress.csv")
# text_log_file = osp.join(log_dir, "debug.log")
params_log_file = osp.join(log_dir, "params.json")
pkl_file = osp.join(log_dir, "params.pkl")

# logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode("gap")
logger.set_snapshot_gap(100)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % "Traffic-RL")

from traffic.make_env import make_env
from mylab.envs.tfenv import TfEnv
env = TfEnv(make_env(env_name='highway',
                    driver_sigma=2.,
                    x_des_sigma=2.,
                    v0_sigma=0.,))

from garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
policy = CategoricalMLPPolicy(
    name='Policy',
    env_spec=env.spec,
    hidden_sizes=(32, 32),
    hidden_nonlinearity=tf.nn.tanh,
)

# from garage.baselines.zero_baseline import ZeroBaseline
# baseline = ZeroBaseline(env_spec=env.spec)
from garage.baselines.linear_feature_baseline import LinearFeatureBaseline
baseline = LinearFeatureBaseline(env_spec=env.spec)
# from garage.tf.baselines.deterministic_mlp_baseline import DeterministicMLPBaseline
# baseline = DeterministicMLPBaseline(env_spec=env.spec,
#                                 regressor_args={
#                                     'hidden_sizes':(32, 32),
#                                     'hidden_nonlinearity':tf.nn.tanh,
#                                 }) # unstable

from garage.tf.algos.trpo import TRPO
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=2048,
    max_path_length=100,
    n_itr=1001,
    discount=0.99,
    step_size=0.01,
)
algo.train()