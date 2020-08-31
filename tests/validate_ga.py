import tensorflow as tf

from examples.AV.example_runner_ga_av import runner as ga_runner


def validate_ga():
    # Overall settings
    max_path_length = 50
    s_0 = [0.0, -4.0, 1.0, 11.17, -35.0]
    base_log_dir = './data'
    # experiment settings
    run_experiment_args = {'snapshot_mode': 'last',
                           'snapshot_gap': 1,
                           'log_dir': None,
                           'exp_name': None,
                           'seed': 0,
                           'n_parallel': 1,
                           'tabular_log_file': 'progress.csv'
                           }

    # runner settings
    runner_args = {'n_epochs': 2,
                   'batch_size': 500,
                   'plot': False
                   }

    # env settings
    env_args = {'id': 'ast_toolbox:GoExploreAST-v1',
                'blackbox_sim_state': False,
                'open_loop': False,
                'fixed_init_state': True,
                's_0': s_0,
                }

    # simulation settings
    sim_args = {'blackbox_sim_state': False,
                'open_loop': False,
                'fixed_initial_state': True,
                'max_path_length': max_path_length
                }

    # reward settings
    reward_args = {'use_heuristic': True}

    # spaces settings
    spaces_args = {'open_loop': False}

    # policy settings
    policy_args = {
        'hidden_sizes': (64, 32),
        'output_nonlinearity': tf.nn.tanh,
    }

    sampler_args = {'n_envs': 1,
                    'open_loop': False}

    # GA Settings

    ga_bpq_args = {'N': 10}

    exp_log_dir = base_log_dir
    max_path_length = 50
    s_0 = [0.0, -4.0, 1.0, 11.17, -35.0]
    env_args['s_0'] = s_0
    reward_args['use_heuristic'] = True
    sim_args['max_path_length'] = max_path_length

    for ga_type in ['ga', 'gasm']:
        run_experiment_args['log_dir'] = exp_log_dir + '/' + ga_type
        run_experiment_args['exp_name'] = ga_type
        ga_algo_args = {'max_path_length': max_path_length,
                        'batch_size': 100,
                        'pop_size': 5,
                        'truncation_size': 3,
                        'keep_best': 1,
                        'step_size': 0.01,
                        'n_itr': 2,
                        'log_interval': 100,
                        }
        ga_runner(
            ga_type=ga_type,
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            policy_args=policy_args,
            algo_args=ga_algo_args,
            bpq_args=ga_bpq_args,
            sampler_args=sampler_args,
            runner_args=runner_args,
        )

    return True


if __name__ == '__main__':
    validate_ga()
