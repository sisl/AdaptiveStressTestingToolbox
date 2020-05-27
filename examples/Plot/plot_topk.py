import csv
import os.path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 15})

n_trial = 5
top_k = 1
batch_size = 4000
max_step = np.inf
max_reward = np.inf
min_reward = -np.inf

exp_name = 'CartpoleNd'
exp_param = 'D1K05A05Ec10'
extra_name = ''
prepath = "../" + exp_name + "/Data/AST/Lexington/" + exp_param
plot_path = "../" + exp_name + "/Data/Plot/top" + str(top_k) + "/"
policies = [
    # "TRPO",\
    "MCTSRS",\
    # "MCTSAS",\
    # "MCTSBV",\
    # "GAP100T20K3Step1.0Fmean","GASMP100T20K3Step1.0Fmean",\
    # "GAP500T20K3Step1.0Fmean","GASMP500T20K3Step1.0Fmean",\
]
plot_name = exp_name + '_' + exp_param + 'avgtop' + str(top_k) + 'trial' + str(n_trial) + extra_name


plts = []
legends = []
fig = plt.figure(figsize=(10, 10))

for (policy_index, policy) in enumerate(policies):
    print(policy)

    for trial in range(n_trial):
        file_path = prepath + '/' + policy + '/' + str(trial) + '/process.csv'
        if os.path.exists(file_path):
            print(trial)
            steps = []
            rewards = []
            with open(file_path) as csv_file:
                if '\0' in open(file_path).read():
                    print("you have null bytes in your input file")
                    csv_reader = csv.reader(x.replace('\0', '') for x in csv_file)
                else:
                    csv_reader = csv.reader(csv_file, delimiter=',')

                for (i, row) in enumerate(csv_reader):
                    if i == 0:
                        entry_dict = {}
                        for index in range(len(row)):
                            entry_dict[row[index]] = index
                    else:
                        # print(row[entry_dict["StepNum"]])
                        if int(row[entry_dict["StepNum"]]) > max_step:
                            break
                        if int(row[entry_dict["StepNum"]]) % batch_size == 0:
                            steps.append(int(row[entry_dict["StepNum"]]))
                            avg_top = 0.0
                            for k in range(top_k):
                                avg_top += np.clip(float(row[entry_dict["reward " + str(k)]]), min_reward, max_reward)
                            avg_top /= top_k
                            rewards.append(avg_top)
            plot, = plt.plot(steps, rewards)
            # plot, = plt.plot(steps,np.mean(np.exp(Rewards),0))
            # plot,_,_ = plt.errorbar(steps,np.mean(Rewards,0),yerr=np.std(Rewards,0)/np.sqrt(n_trial),errorevery=10)
            plts.append(plot)
            legends.append(policy + '_' + str(trial))

plt.legend(plts, legends)
plt.xlabel('Step Number')
plt.ylabel('Top ' + str(top_k) + ' Reward')
fig.savefig(plot_path + plot_name)
plt.close(fig)
