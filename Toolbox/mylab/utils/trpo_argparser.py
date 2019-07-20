import argparse

def get_trpo_parser(log_dir='./'):
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, default="cartpole")
	parser.add_argument('--n_trial', type=int, default=10)
	parser.add_argument('--trial_start', type=int, default=0)
	parser.add_argument('--n_itr', type=int, default=5001)
	parser.add_argument('--batch_size', type=int, default=1000)
	parser.add_argument('--step_size', type=float, default=0.1)
	parser.add_argument('--snapshot_mode', type=str, default="gap")
	parser.add_argument('--snapshot_gap', type=int, default=1000)
	parser.add_argument('--log_dir', type=str, default=log_dir)
	parser.add_argument('--args_data', type=str, default=None)
	args = parser.parse_args()
	args.log_dir += ('Step'+str(args.step_size))
	return args