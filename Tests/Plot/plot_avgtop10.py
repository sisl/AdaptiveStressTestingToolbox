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

exp_name = 'LunarLander'#'CartPole'#'BipedalWalker'#'Acrobot'
exp_param = 'L100I06'#'L100Th0612I015'#'L100TL25'#'L100Th19'
prepath = "../"+exp_name+"/Data/AST/Lexington/"+exp_param
plot_path = "../"+exp_name+"/Data/Plot/"
policies = ["TRPO",\
        "GATRDP100T20K3Step1.0Fmean","GATRDP100T20K3Step0.1Fmean","GATRDP100T20K3Step0.01Fmean",\
        "PSMCTSTRCK0.5A0.5Ec1.414Step1.0FmeanQmax","PSMCTSTRCK0.5A0.5Ec1.414Step0.1FmeanQmax","PSMCTSTRCK0.5A0.5Ec1.414Step0.01FmeanQmax"]
        # "PSMCTSTRCK0.5A0.5Ec100.0Step1.0FmeanQmax","PSMCTSTRCK0.5A0.5Ec100.0Step0.1FmeanQmax","PSMCTSTRCK0.5A0.5Ec100.0Step0.01FmeanQmax"]
plot_name = exp_name+'_'+exp_param


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
                if '\0' in open(file_path).read():
                    print("you have null bytes in your input file")
                    csv_reader = csv.reader(x.replace('\0', '') for x in csv_file)
                else:
                    csv_reader = csv.reader(csv_file, delimiter=',')

                for (i,row) in enumerate(csv_reader):
                    if i == 0:
                        entry_dict = {}
                        for index in range(len(row)):
                            entry_dict[row[index]] = index
                    else:
                        # print(row[entry_dict["StepNum"]])
                        if int(row[entry_dict["StepNum"]]) > max_step:
                            break
                        if int(row[entry_dict["StepNum"]])%batch_size == 0:
                            steps.append(int(row[entry_dict["StepNum"]]))
                            avg_top = 0.0
                            for k in range(top_k):
                                avg_top += np.clip(float(row[entry_dict["reward "+str(k)]]),min_reward,max_reward)
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
    # plot, = plt.plot(steps,np.mean(Rewards,0))
    plot,_,_ = plt.errorbar(steps,np.mean(Rewards,0),yerr=np.std(Rewards,0)/np.sqrt(n_trial),errorevery=10)
    plts.append(plot)
    legends.append(policy)

plt.legend(plts,legends)
plt.xlabel('Step Number')
plt.ylabel('Average Top '+str(top_k) +' Reward')        
fig.savefig(plot_path+plot_name+'_avgtop10.pdf')
plt.close(fig)