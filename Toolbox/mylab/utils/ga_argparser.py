import argparse

def get_ga_parser(log_dir='./'):
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, default="cartpole")
	parser.add_argument('--n_trial', type=int, default=5)
	parser.add_argument('--n_itr', type=int, default=25)
	parser.add_argument('--batch_size', type=int, default=4000)
	parser.add_argument('--snapshot_mode', type=str, default="gap")
	parser.add_argument('--snapshot_gap', type=int, default=10)
	parser.add_argument('--log_dir', type=str, default=log_dir)
	parser.add_argument('--step_size', type=float, default=0.01)
	parser.add_argument('--step_size_anneal', type=float, default=1.0)
	parser.add_argument('--args_data', type=str, default=None)
	parser.add_argument('--fit_f',type=str, default="max")
	parser.add_argument('--pop_size', type=int, default=100)
	parser.add_argument('--elites', type=int, default=20)
	parser.add_argument('--keep_best', type=int, default=3)
	parser.add_argument('--log_interval', type=int, default=4000)
	args = parser.parse_args()
	if args.step_size_anneal == 1.0:
		args.log_dir += ('Step'+str(args.step_size)+'F'+args.fit_f)
	else:
		args.log_dir += ('Step'+str(args.step_size)+'Anneal'+str(args.step_size_anneal)+'F'+args.fit_f)
	return args