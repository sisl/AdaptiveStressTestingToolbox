import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 15})
from matplotlib import pyplot as plt
import numpy as np

n_trial = 5
top_k = 10
batch_size = 4000
max_step = np.inf
max_reward = np.inf
min_reward = -np.inf

prepath = "../CartPole/Data/AST/Lexington/l100th0612i01"
exp = "CartPole_l100th0612i01"
plot_path = "../CartPole/Data/Plot/"
policies = ["TRPO",\
        "GATRDP100T20K3Step1.0Fmean","GATRDP100T20K3Step0.1Fmean","GATRDP100T20K3Step0.01Fmean",\
        "PSMCTSTRCK0.5A0.5Ec1.414Step1.0FmeanQmax","PSMCTSTRCK0.5A0.5Ec1.414Step0.1FmeanQmax","PSMCTSTRCK0.5A0.5Ec1.414Step0.01FmeanQmax"]
        # "PSMCTSTRCK0.5A0.5Ec100.0Step1.0FmeanQmax","PSMCTSTRCK0.5A0.5Ec100.0Step0.1FmeanQmax","PSMCTSTRCK0.5A0.5Ec100.0Step0.01FmeanQmax"]
plot_name = exp


plts = []
legends = []
fig = plt.figure(figsize=(10, 10))

for (policy_index,policy) in enumerate(policies):
    print(policy)
    Rewards = []
    min_array_length = np.inf
    for trial in range(n_trial):
        file_path = prepath+'/'+policy+'/'+str(trial)+'/process.csv'
        if os.path.exists(file_path):
            print(trial)
            steps = []
            rewards = []
            with open(file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for (i,row) in enumerate(csv_reader):
                    if i == 0:
                        entry_dict = {}
                        for index in range(len(row)):
                            entry_dict[row[index]] = index
                    else:
                        if int(row[entry_dict["StepNum"]]) > max_step:
                            break
                        if int(row[entry_dict["StepNum"]])%batch_size == 0:
                            steps.append(int(row[entry_dict["StepNum"]]))
                            avg_top = 0.0
                            for k in range(top_k):
                                avg_top += np.clip(float(row[entry_dict["reward "+str(k)]]),min_reward,max_reward)
                            avg_top /= top_k
                            rewards.append(avg_top)
            ### if process2 exists
            step1 = steps[-1]
            print(step1)
            file_path2 = prepath+'/'+policy+'/'+str(trial)+'/process2.csv'
            if os.path.exists(file_path2):
                print(str(trial)+"_2")
                with open(file_path2) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for (i,row) in enumerate(csv_reader):
                        if i == 0:
                            entry_dict = {}
                            for index in range(len(row)):
                                entry_dict[row[index]] = index
                        else:
                            if step1+int(row[entry_dict["StepNum"]]) > max_step:
                                break
                            if (step1+int(row[entry_dict["StepNum"]]))%batch_size == 0:
                                steps.append(step1+int(row[entry_dict["StepNum"]]))
                                avg_top = 0.0
                                for k in range(top_k):
                                    avg_top += np.clip(float(row[entry_dict["reward "+str(k)]]),min_reward,max_reward)
                                avg_top /= top_k
                                rewards.append(avg_top)
                print(steps[-1])
            ###
            if len(rewards) < min_array_length:
                min_array_length = len(rewards) 
            Rewards.append(rewards)
            # print(len(rewards))
            # print(steps[-1])
            # print(min_array_length)
    steps = steps[:min_array_length]
    Rewards = [rewards[:min_array_length] for rewards in Rewards]
    # plot, = plt.plot(steps,np.mean(Rewards,0),color=colors[policy_index])
    plot, = plt.plot(steps,np.mean(Rewards,0))
    plts.append(plot)
    legends.append(policy)

plt.legend(plts,legends)
plt.xlabel('Step Number')
plt.ylabel('Average Top '+str(top_k) +' Reward')        
fig.savefig(plot_path+plot_name+'_avgtop10.pdf')
plt.close(fig)