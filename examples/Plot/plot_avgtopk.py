import csv
import os.path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 15})

n_trial = 10
top_k = 1
batch_size = 1000
max_step = np.inf
max_reward = np.inf
min_reward = -np.inf

exp_name = 'CartpoleNd'
exp_param = 'D1SUT5'
extra_name = 'MCTS'
prepath = "../" + exp_name + "/Data/AST/Lexington/" + exp_param
plot_path = "../" + exp_name + "/Data/Plot/avgtop" + str(top_k) + "/"
policies = [
    # "TRPOB2000Step0.1","TRPOB2000Step1.0","TRPOB2000Step5.0",\
    # "TRPOB50000Step0.1","TRPOB50000Step1.0","TRPOB50000Step5.0",\
    # "TRPOLSTMB2000Step0.1","TRPOLSTMB2000Step1.0","TRPOLSTMB2000Step5.0",\
    # "TRPOLSTMB50000Step0.1","TRPOLSTMB50000Step1.0","TRPOLSTMB50000Step5.0",\
    "MCTSRSEc10.0K0.3A0.3", "MCTSRSEc10.0K0.5A0.5", "MCTSRSEc10.0K0.8A0.8",\
    "MCTSASEc10.0K0.3A0.3", "MCTSASEc10.0K0.5A0.5", "MCTSASEc10.0K0.8A0.8",\
    "MCTSBVEc10.0K0.3A0.3", "MCTSBVEc10.0K0.5A0.5", "MCTSBVEc10.0K0.8A0.8",\
]
plot_name = exp_name + '_' + exp_param + 'avgtop' + str(top_k) + 'trial' + str(n_trial) + extra_name


plts = []
legends = []
fig = plt.figure(figsize=(10, 10))

for (policy_index, policy) in enumerate(policies):
    print(policy)
    Rewards = []
    min_array_length = np.inf
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
            if len(rewards) < min_array_length:
                min_array_length = len(rewards)
            Rewards.append(rewards)
            # print(len(rewards))
            # print(steps[-1])
            # print(min_array_length)
    steps = steps[:min_array_length]
    Rewards = [rewards[:min_array_length] for rewards in Rewards]
    plot, = plt.plot(steps, np.mean(Rewards, 0))
    # plot, = plt.plot(steps,np.mean(np.exp(Rewards),0))
    # plot,_,_ = plt.errorbar(steps,np.mean(Rewards,0),yerr=np.std(Rewards,0)/np.sqrt(n_trial),errorevery=10)
    plts.append(plot)
    legends.append(policy + ' ' + str(np.mean(Rewards, 0)[-1]))

plt.legend(plts, legends)
plt.xlabel('Step Number')
plt.ylabel('Average Top ' + str(top_k) + ' Reward')
fig.savefig(plot_path + plot_name)
plt.close(fig)
