import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from garage.misc import logger

from mylab.rewards.ast_reward_standard import ASTRewardS
from mylab.envs.ast_env import ASTEnv
from Cartpole.cartpole_simulator import CartpoleSimulator



from mylab.algos.gasm import GASM

import os.path as osp
import argparse
import tensorflow as tf
import joblib
import math
import numpy as np

import mylab.mcts.BoundedPriorityQueues as BPQ
import csv
# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="cartpole")
parser.add_argument('--n_trial', type=int, default=5)
parser.add_argument('--n_itr', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--snapshot_mode', type=str, default="none")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='./Data/AST/GASM')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

top_k = 10
max_path_length = args.batch_size
open_loop = False

pop_size = 100
elites = 20
keep_best = 3
step_size=0.01

tf.set_random_seed(0)
sess = tf.Session()
sess.__enter__()

# Instantiate the env
data = joblib.load("../Cartpole/control_policy.pkl")
sut = data['policy']
reward_function = ASTRewardS()

simulator = CartpoleSimulator(sut=sut,max_path_length=max_path_length,use_seed=False)
env = TfEnv(ASTEnv(open_loop=open_loop,
				   simulator=simulator,
				   fixed_init_state=True,
				   s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
				   reward_function=reward_function,
				   ))

# Create policy
policy = DeterministicMLPPolicy(
	name='ast_agent',
	env_spec=env.spec,
	hidden_sizes=(64, 32),
	output_nonlinearity=tf.nn.tanh,
)

# Training
with open(osp.join(args.log_dir, 'total_result.csv'), mode='w') as csv_file:
	fieldnames = ['step_count']
	for i in range(top_k):
		fieldnames.append('reward '+str(i))
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
	writer.writeheader()

	for trial in range(args.n_trial):
		# Create the logger
		log_dir = args.log_dir+'/'+str(trial)

		tabular_log_file = osp.join(log_dir, 'process.csv')
		text_log_file = osp.join(log_dir, 'text.txt')
		params_log_file = osp.join(log_dir, 'args.txt')

		logger.set_snapshot_dir(log_dir)
		logger.set_snapshot_mode(args.snapshot_mode)
		logger.set_snapshot_gap(args.snapshot_gap)
		logger.log_parameters_lite(params_log_file, args)
		if trial > 0:
			old_log_dir = args.log_dir+'/'+str(trial-1)
			logger.pop_prefix()
			logger.remove_text_output(osp.join(old_log_dir, 'text.txt'))
			logger.remove_tabular_output(osp.join(old_log_dir, 'process.csv'))
		logger.add_text_output(text_log_file)
		logger.add_tabular_output(tabular_log_file)
		logger.push_prefix("["+args.exp_name+'_trial '+str(trial)+"]")

		np.random.seed(trial)

		params = policy.get_params()
		sess.run(tf.variables_initializer(params))

		# Instantiate the garage objects
		baseline = ZeroBaseline(env_spec=env.spec)
		top_paths = BPQ.BoundedPriorityQueue(top_k)
		algo = GASM(
			env=env,
			policy=policy,
			baseline=baseline,
			batch_size=args.batch_size,
			pop_size=pop_size,
			elites=elites,
			keep_best=keep_best,
			step_size=step_size,
			n_itr=args.n_itr,
			store_paths=False,
			max_path_length=max_path_length,
			top_paths = top_paths,
			fit_f = "max",
			plot=False,
			)

		algo.train(sess=sess, init_var=False)

		row_content = dict()
		row_content['step_count'] = args.n_itr*args.batch_size*pop_size
		i = 0
		for (r,action_seq) in algo.top_paths:
			row_content['reward '+str(i)] = r
			i += 1
		writer.writerow(row_content)