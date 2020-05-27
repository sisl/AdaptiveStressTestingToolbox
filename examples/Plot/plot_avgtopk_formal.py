import csv
import os.path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save

matplotlib.rcParams.update({'font.size': 10, 'font.family': 'Times New Roman',
                            'text.usetex': True})

n_trial = 10
top_k = 1
batch_size = 1000
max_step = np.inf
max_reward = np.inf
min_reward = -np.inf

exp_name = 'CartpoleNd'
exp_param = 'D1SUT5'
extra_name = ''  # 'hyper'
prepath = "../" + exp_name + "/Data/AST/Lexington/" + exp_param
plot_path = "../" + exp_name + "/Data/Plot/avgtop" + str(top_k) + "/"
# plot_path = "/Users/xiaobaima/Dropbox/SISL/ASTToolbox/IROS/plots/"

policy_groups = [
                [
                    "RandomSearchB2000",
                ],
    [
                    # "TRPOB2000Step0.1","TRPOB2000Step1.0",
                    "TRPOB2000Step5.0",\
                    # "TRPOB50000Step0.1","TRPOB50000Step1.0","TRPOB50000Step5.0",\
                ],\
    [
                    # "TRPOLSTMB2000Step0.1",
                    "TRPOLSTMB2000Step1.0",
                    # "TRPOLSTMB2000Step5.0",\
                    # "TRPOLSTMB50000Step0.1","TRPOLSTMB50000Step1.0","TRPOLSTMB50000Step5.0",\
                ],\
    [
                    # "MCTSRSEc10.0K0.3A0.3",
                    "MCTSRSEc10.0K0.5A0.5",
                    # "MCTSRSEc10.0K0.8A0.8",\
                    # "MCTSASEc10.0K0.3A0.3","MCTSASEc10.0K0.5A0.5","MCTSASEc10.0K0.8A0.8",\
                    # "MCTSBVEc10.0K0.3A0.3","MCTSBVEc10.0K0.5A0.5","MCTSBVEc10.0K0.8A0.8",\
                ]\
]

algos = ["Random Search", "TRPO, Closed-Loop", "TRPO, Open-Loop", "MCTS, Open-Loop"]
parameters = [[""],
              ["step size 0.1"],
              ['$k=\\alpha=0.5,ec=10.0$', '$k=\\alpha=0.5,ec=10.0$', '$k=\\alpha=0.5,ec=10.0$'],
              ]

exp_name_f = 'Cartpole'

exp_param_f = ''

colors = ["black", "blue", "green", "red"]
plot_name = exp_name + '_' + exp_param + 'avgtop' + str(top_k) + 'trial' + str(n_trial) + extra_name


plts = []
legends = []
fig = plt.figure(figsize=(10, 10))

for (group_index, policies) in enumerate(policy_groups):
    best_policy_index = -1
    best_rewards = []
    best_steps = []
    best_last_reward = -np.inf
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
        steps = steps[:min_array_length]
        Rewards = [rewards[:min_array_length] for rewards in Rewards]
        if np.mean(Rewards, 0)[-1] > best_last_reward:
            best_last_reward = np.mean(Rewards, 0)[-1]
            best_policy_index = policy_index
            best_rewards = Rewards
            best_steps = steps
    Rewards = best_rewards
    steps = best_steps
    # y = np.mean(Rewards,0)
    # yerr = np.std(Rewards,0)/np.sqrt(n_trial)
    plot, = plt.plot(steps, np.mean(Rewards, 0), color=colors[group_index])
    print(np.max(Rewards))
    # plt.fill_between(steps,y+yerr,y-yerr,linewidth=0,facecolor=colors[group_index],alpha=0.3)
    plts.append(plot)
    # legends.append(algos[group_index]+' '+parameters[group_index][best_policy_index])
    legends.append(algos[group_index])

axes = plt.gca()
# start, end = axes.get_xlim()
# axes.set_xlim([0,2e6])
# axes.xaxis.set_ticks(np.arange(0, 2e6, int(2e6/4)))
# axes.set_yscale('log')
# axes.set_xlim([xmin,xmax])
# axes.set_ylim([-0.15,1.0])
# plt.title(exp_name_f+' '+exp_param_f)
plt.legend(plts, legends)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Step Number')
# plt.ylabel('Average Top '+str(top_k) +' Reward')
plt.ylabel('Average Best Return')
fig.savefig(plot_path + plot_name + '.pdf', bbox_inches='tight')
tikz_save(plot_path + plot_name + '.tex')
plt.close(fig)
