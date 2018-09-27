from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
import rllab.misc.logger as logger

from mylab.crosswalk_env_2 import CrosswalkEnv
from mylab.crosswalk_sensor_env_2 import CrosswalkSensorEnv
from mylab.myalgo import GA

import os.path as osp
import argparse
from save_trials import *
import pdb
import tensorflow as tf

from simulators.example_av_simulator import ExampleAVSimulator
from mylab.example_av_reward import ExampleAVReward
from mylab.ast_env import ASTEnv
from mylab.ast_vectorized_sampler import ASTVectorizedSampler

parser = argparse.ArgumentParser()
#Algo Params
parser.add_argument('--iters', type=int, default=101)
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--store_paths', type=bool, default=True)
parser.add_argument('--action_only', type=bool, default=True)
parser.add_argument('--mlp_baseline', type=bool, default=False)
parser.add_argument('--lambda', type=float, default = 0.999)
parser.add_argument('--fit_f', type=str, default='max')
parser.add_argument('--pop_size', type=int, default=1000)
parser.add_argument('--elites', type=int, default=20)

# Logger Params
parser.add_argument('--exp_name', type=str, default='crosswalk_exp')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='../../../../scratch/mkoren/run1')
parser.add_argument('--args_data', type=str, default=None)

#Policy Params
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--policy', type=str, default="LSTM")
parser.add_argument('--use_peepholes', type=bool, default=False)

#Optimizer Params
parser.add_argument('--optimizer', type=str, default="CGO")


#Environement Params
parser.add_argument('--interactive', type=bool, default=False)
parser.add_argument('--dt', type=float, default=0.1)
parser.add_argument('--num_peds', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.85)
parser.add_argument('--beta', type=float, default=0.005)
parser.add_argument('--v_des', type=float, default=11.17)
parser.add_argument('--delta', type=float, default=4.0)
parser.add_argument('--t_headway', type=float, default=1.5)
parser.add_argument('--a_max', type=float, default=3.0)
parser.add_argument('--s_min', type=float, default=4.0)
parser.add_argument('--d_cmf', type=float, default=2.0)
parser.add_argument('--d_max', type=float, default=9.0)
parser.add_argument('--min_dist_x', type=float, default=2.5)
parser.add_argument('--min_dist_y', type=float, default=1.4)
parser.add_argument('--car_init_x', type=float, default=-35.0)
parser.add_argument('--car_init_y', type=float, default=0.0)
parser.add_argument('--x_accel_low', type=float, default=-1.0)
parser.add_argument('--y_accel_low', type=float, default=-1.0)
parser.add_argument('--x_accel_high', type=float, default=1.0)
parser.add_argument('--y_accel_high', type=float, default=1.0)
parser.add_argument('--x_boundary_low', type=float, default=-10.0)
parser.add_argument('--y_boundary_low', type=float, default=-10.0)
parser.add_argument('--x_boundary_high', type=float, default=10.0)
parser.add_argument('--y_boundary_high', type=float, default=10.0)
parser.add_argument('--x_v_low', type=float, default=-10.0)
parser.add_argument('--y_v_low', type=float, default=-10.0)
parser.add_argument('--x_v_high', type=float, default=10.0)
parser.add_argument('--y_v_high', type=float, default=10.0)
parser.add_argument('--mean_x', type=float, default=0.0)
parser.add_argument('--mean_y', type=float, default=0.0)
parser.add_argument('--cov_x', type=float, default=0.1)
parser.add_argument('--cov_y', type=float, default=0.01)


args = parser.parse_args()
log_dir = args.log_dir

tabular_log_file = osp.join(log_dir, args.tabular_log_file)
text_log_file = osp.join(log_dir, args.text_log_file)
params_log_file = osp.join(log_dir, args.params_log_file)

logger.log_parameters_lite(params_log_file, args)
logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(args.log_tabular_only)
logger.push_prefix("[%s] " % args.exp_name)

sim = ExampleAVSimulator(ego=None,
                           num_peds=args.num_peds,
                           dt=args.dt,
                           alpha = args.alpha,
                           beta = args.beta,
                           v_des=args.v_des,
                           delta=args.delta,
                           t_headway=args.t_headway,
                           a_max=args.a_max,
                           s_min=args.s_min,
                           d_cmf=args.d_cmf,
                           d_max=args.d_max,
                           min_dist_x=args.min_dist_x,
                           min_dist_y=args.min_dist_y,
                           x_accel_low=args.x_accel_low,
                           y_accel_low=args.y_accel_low,
                           x_accel_high=args.x_accel_high,
                           y_accel_high=args.y_accel_high,
                           x_boundary_low=args.x_boundary_low,
                           y_boundary_low=args.y_boundary_low,
                           x_boundary_high=args.x_boundary_high,
                           y_boundary_high=args.y_boundary_high,
                           x_v_low=args.x_v_low,
                           y_v_low=args.y_v_low,
                           x_v_high=args.x_v_high,
                           y_v_high=args.y_v_high,
                           mean_x=args.mean_x,
                           mean_y=args.mean_y,
                           cov_x=args.cov_x,
                           cov_y=args.cov_y,
                           car_init_x=args.car_init_x,
                           car_init_y=args.car_init_y,
                           mean_sensor_noise = 0.0,
                           cov_sensor_noise = 0.1,
                           action_only = args.action_only,
                         max_path_length=50)

reward_function = ExampleAVReward(num_peds=args.num_peds,
                                  cov_x=args.cov_x,
                                  cov_y=args.cov_y,
                                  cov_sensor_noise=0.1)

env = TfEnv(normalize(ASTEnv(num_peds=args.num_peds,
                             max_path_length=50,
                                v_des=args.v_des,
                                x_accel_low=args.x_accel_low,
                                y_accel_low=args.y_accel_low,
                                x_accel_high=args.x_accel_high,
                                y_accel_high=args.y_accel_high,
                                x_boundary_low=args.x_boundary_low,
                                y_boundary_low=args.y_boundary_low,
                                x_boundary_high=args.x_boundary_high,
                                y_boundary_high=args.y_boundary_high,
                                x_v_low=args.x_v_low,
                                y_v_low=args.y_v_low,
                                x_v_high=args.x_v_high,
                                y_v_high=args.y_v_high,
                                car_init_x=args.car_init_x,
                                car_init_y=args.car_init_y,
                                action_only = args.action_only,
                                sample_init_state=True,
                                s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
                                simulator=sim,
                                reward_function=reward_function,
                             )))

if args.policy == "LSTM":
    policy = GaussianLSTMPolicy(name='lstm_policy',
                               env_spec=env.spec,
                                hidden_dim=args.hidden_dim,
                                use_peepholes=args.use_peepholes)
elif args.policy == "GRU":
    policy = GaussianGRUPolicy(name='lstm_policy',
                                env_spec=env.spec,
                                hidden_dim=args.hidden_dim)
else:
    policy = GaussianMLPPolicy(name='mlp_policy',
                               env_spec=env.spec,
                               hidden_sizes=(512, 256, 128, 64, 32))
# policy = DeterministicMLPPolicy(name='mlp_policy',
#                                 env_spec=env.spec,
#                                 hidden_sizes=(512, 256, 128, 64, 32))
if args.mlp_baseline:
    baseline = GaussianMLPBaseline(env_spec=env.spec)
else:
    baseline = LinearFeatureBaseline(env_spec=env.spec)

# parallel_sampler.initialize(n_parallel=4)
# singleton_pool.initialize(n_parallel=4)

if args.optimizer == "CGO":
    optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
else:
    optimizer = PenaltyLbfgsOptimizer(name="LBFGS")

sampler_cls = ASTVectorizedSampler

algo = TRPO(
    env=env,
    policy=policy,
    baseline=LinearFeatureBaseline(env_spec=env.spec),
    batch_size=args.batch_size,
    step_size=args.step_size,
    n_itr=args.iters,
    store_paths=True,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
    max_path_length=50,
    sampler_cls=sampler_cls,
    sampler_args= {"sim": sim,
                    "reward_function": reward_function}
)
# algo = GA(
#     env=env,
#     policy=policy,
#     baseline=baseline,
#     batch_size=args.batch_size,
#     step_size=args.step_size,
#     gae_lambda=0.999,
#     n_itr=args.iters,
#     store_paths=True,
#     optimizer=optimizer,
#     pop_size=args.pop_size,
#     elites=args.elites,
#     fit_f=args.fit_f
# )
saver = tf.train.Saver(save_relative_paths=True)
with tf.Session() as sess:
    algo.train(sess=sess)

    header = 'trial, step, ' + 'v_x_car, v_y_car, x_car, y_car, '
    for i in range(0,args.num_peds):
        header += 'v_x_ped_' + str(i) + ','
        header += 'v_y_ped_' + str(i) + ','
        header += 'x_ped_' + str(i) + ','
        header += 'y_ped_' + str(i) + ','

    for i in range(0,args.num_peds):
        header += 'a_x_'  + str(i) + ','
        header += 'a_y_' + str(i) + ','
        header += 'noise_v_x_' + str(i) + ','
        header += 'noise_v_y_' + str(i) + ','
        header += 'noise_x_' + str(i) + ','
        header += 'noise_y_' + str(i) + ','

    header += 'reward'
    save_trials(args.iters, args.log_dir, header, sess, save_every_n=args.snapshot_gap)
    saver.save(sess, args.log_dir + '/' + args.exp_name)


