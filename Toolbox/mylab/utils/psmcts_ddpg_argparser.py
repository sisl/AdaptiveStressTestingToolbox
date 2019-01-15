import argparse
import numpy as np

def get_psmcts_ddpg_parser(log_dir='./'):
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, default="cartpole")
	parser.add_argument('--n_trial', type=int, default=5)
	parser.add_argument('--n_itr', type=int, default=20000)
	parser.add_argument('--batch_size', type=int, default=1000)
	parser.add_argument('--snapshot_mode', type=str, default="none")
	parser.add_argument('--snapshot_gap', type=int, default=10)
	parser.add_argument('--log_dir', type=str, default=log_dir)
	parser.add_argument('--step_size', type=float, default=1.0)
	parser.add_argument('--step_size_anneal', type=float, default=1.0)
	parser.add_argument('--args_data', type=str, default=None)
	parser.add_argument('--f_F',type=str, default="mean")
	parser.add_argument('--ec',type=float, default=1.414)
	parser.add_argument('--k',type=float, default=0.5)
	parser.add_argument('--alpha',type=float, default=0.5)
	parser.add_argument('--n_ca',type=int, default=4)
	parser.add_argument('--initial_pop',type=int, default=0)
	parser.add_argument('--log_interval', type=int, default=4000)
	parser.add_argument('--plot_tree', type=bool, default=False)
	parser.add_argument('--f_Q', type=str, default='max')
	parser.add_argument('--batch_size2',type=int,default=200)
	parser.add_argument('--n_itr2', type=int, default=500)
	parser.add_argument('--n_epoch_cycles', type=int, default=20)
	parser.add_argument('--step_size2', type=float, default=0.1)
	args = parser.parse_args()
	args.log_dir += ('Step'+str(args.step_size)+'Ec'+str(args.ec)+'K'+str(args.k)+'A'+str(args.alpha)+'SStep'+str(args.step_size2))
	if args.initial_pop > 0:
		args.log_dir += ('InitP'+str(args.initial_pop))
	args.log_dir += 'Q'+args.f_Q
	args.log_dir += 'F'+args.f_F
	return args