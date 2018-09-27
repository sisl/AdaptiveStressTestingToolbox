import rllab
import joblib
import numpy as np
import sandbox
import pdb
import tensorflow as tf


def save_trials(iters, path, header, sess, save_every_n = 100):
    #sess.run(tf.global_variables_initializer())
    for i in range(0, iters):
        if (np.mod(i, save_every_n) != 0):
            continue
        with tf.variable_scope('Loader' + str(i)):
            data = joblib.load(path + '/itr_' + str(i) + '.pkl')
            # pdb.set_trace()
            paths = data['paths']

            trials = np.array([]).reshape(0, paths[0]['env_infos']['info']['cache'].shape[1])
            crashes = np.array([]).reshape(0, paths[0]['env_infos']['info']['cache'].shape[1])
            for n, a_path in enumerate(paths):
                cache = a_path['env_infos']['info']['cache']
                # pdb.set_trace()
                cache[:, 0] = n
                trials = np.concatenate((trials, cache), axis=0)
                if cache[-1,-1] == 0.0:
                    crashes = np.concatenate((crashes, cache), axis=0)

            np.savetxt(fname=path + '/trials_' + str(i) + '.csv',
                       X=trials,
                       delimiter=',',
                       header=header)

            np.savetxt(fname=path + '/crashes_' + str(i) + '.csv',
                       X=crashes,
                       delimiter=',',
                       header=header)


        # pdb.set_trace()
        # env = data['env']
        # w_env = env.wrapped_env
        # out = np.array(w_env.get_cache_list())
        # if out[-1,9] == 0.0:
        #     crashes = out
        # else:
        #     crashes = np.array([]).reshape(0,out.shape[1])
        # for i in range(1, iters):
        #     data = joblib.load(path + '/itr_' + str(i) + '.pkl')
        #     env = data['env']
        #     w_env = env.wrapped_env
        #     iter_array = np.array(w_env.get_cache_list())
        #     iter_array[:, 0] = i
        #     out = np.concatenate((out, iter_array), axis=0)
        #     if iter_array[-1, 9] == 0.0:
        #         crashes = np.concatenate((crashes, iter_array), axis=0)







