import joblib
import numpy as np
import tensorflow as tf


def example_save_trials(iters, path, header, sess, save_every_n = 100):
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






