# export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

import roboschool
import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv
from mylab.envs.seed_env import SeedEnv
import numpy as np
import time
import argparse
import joblib
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='Acrobot')
parser.add_argument('--itr', type=int, default=50000)
# parser.add_argument('--trial', type=int, default=0)
args = parser.parse_args()

max_path_length = 100
# policy_name = 'PSMCTSTRCK0.5A0.5Ec1.414Step1.0FmeanQmax'
# policy_name = 'PSMCTSTRCK0.3A0.3Ec1.414Step1.0FmeanQmax'
# policy_name = 'GATRDP100T20K3Step1.0Fmean'
policy_name = 'TRPOStep0.1'

path0 = '../'
path1 = '/Data/'

log_dir = path0+args.exp_name+path1+policy_name
import os
if not os.path.isdir(log_dir + '/Videos'):
    os.mkdir(log_dir + '/Videos')
for trial in range(9):
    print(trial)
    np.random.seed(trial)
    tf.reset_default_graph()
    with tf.Session() as sess:
        data_path = log_dir+'/'+str(trial)+'/itr_'+str(args.itr)+'.pkl'
        data = joblib.load(data_path)
        env = data['env']
        o = env.reset()

        from gym.wrappers.monitoring.video_recorder import VideoRecorder
        video_path = log_dir+'/Videos/'+str(trial)+'_itr_'+str(args.itr)
        video_recorder = VideoRecorder(env=env, base_path=video_path, enabled=True)

        policy = data['policy']
        path_length = 0
        done = False
        c_r = 0.0
        while (path_length < max_path_length) and (not done):
            path_length += 1
            a, _ = policy.get_action(o)
            o, r, done, _ = env.step(a)
            c_r += r
            video_recorder.capture_frame()
            # print("step: ",path_length)
            # print("o: ",o)
            # print('r: ',r)
            # print(done)
            time.sleep(0.1)
        print('c_r: ',c_r)

        video_recorder.close()