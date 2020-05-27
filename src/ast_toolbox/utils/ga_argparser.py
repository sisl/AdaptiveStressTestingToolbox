import argparse


def get_ga_parser(log_dir='./'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="cartpole")
    parser.add_argument('--nd', type=int, default=1)
    parser.add_argument('--sut_itr', type=int, default=5)
    parser.add_argument('--n_trial', type=int, default=5)
    parser.add_argument('--trial_start', type=int, default=0)
    parser.add_argument('--n_itr', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--snapshot_mode', type=str, default="none")
    parser.add_argument('--snapshot_gap', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default=log_dir)
    parser.add_argument('--init_step', type=float, default=0.4)
    parser.add_argument('--step_size', type=float, default=1.0)
    parser.add_argument('--step_size_anneal', type=float, default=1.0)
    parser.add_argument('--args_data', type=str, default=None)
    parser.add_argument('--f_F', type=str, default="mean")  # fitness function
    parser.add_argument('--pop_size', type=int, default=100)
    parser.add_argument('--truncation_size', type=int, default=20)
    parser.add_argument('--keep_best', type=int, default=3)
    parser.add_argument('--log_interval', type=int, default=1000)
    args = parser.parse_args()
    args.log_dir += ('P' + str(args.pop_size))
    args.log_dir += ('T' + str(args.truncation_size))
    args.log_dir += ('K' + str(args.keep_best))
    args.log_dir += ('Step' + str(args.step_size))
    if args.step_size_anneal != 1.0:
        args.log_dir += ('Anneal' + str(args.step_size_anneal))
    args.log_dir += ('F' + args.f_F)
    return args
