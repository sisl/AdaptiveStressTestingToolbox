import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 35, 'font.family':'Times New Roman',\
                            'text.usetex': False})
from matplotlib import pyplot as plt
import numpy as np

n_trial = 10#20
top_k = 1
batch_size = 1e3#4000
max_step = 2e7#5e6
max_reward = np.inf
min_reward = -np.inf

exp_name = "Roboschool"
exp_param = "Ant"#"HalfCheetah"#"Hopper"#
extra_name = ''#'hyper'
prepath = "../"+exp_name+"/Data/Lexington/"+exp_param
# plot_path = "../"+exp_name+"/Data/Plot/avgtop"+str(top_k)+"/"
plot_path = "/Users/xiaobaima/Dropbox/SISL/MCTSPO/IJCAI/presentation/Plots/"

policy_groups = [
                [
                "TRPOStep0.1",\
                "TRPOStep1.0",
                ],\
                [
                "GATRDP100T20K3Step1.0Fmean",\
                "GATRDP500T20K3Step1.0Fmean",\
                "GATRDP1000T20K3Step1.0Fmean",\
                ],\
                [
                "PSMCTSTRCK0.3A0.3Ec10.0Step1.0FmeanQmax",\
                "PSMCTSTRCK0.5A0.5Ec10.0Step1.0FmeanQmax",\
                "PSMCTSTRCK0.8A0.8Ec10.0Step1.0FmeanQmax",\
                ]\
                ]

algos = ["TRPO","Deep GA","MCTSPO"]
parameters = [["step size 0.1","step size 1.0"],\
                ["population size 100","population size 500","population size 1000"],\
                ['$k=\\alpha=0.3$','$k=\\alpha=0.5$','$k=\\alpha=0.8$'],
                ]

exp_name_f = 'Roboschool'

exp_param_f = exp_param
            # 'Ant'
            # 'HalfCheetah'
            # 'Hopper'
            

colors = ["blue","green","red"]
plot_name = exp_name+'_'+exp_param+'avgtop'+str(top_k)+'trial'+str(n_trial)+extra_name


plts = []
legends = []
fig = plt.figure(figsize=(9.5, 10))

for (group_index,policies) in enumerate(policy_groups):
    best_policy_index = -1
    best_rewards = []
    best_steps = []
    best_last_reward = -np.inf
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
        steps = steps[:min_array_length]
        Rewards = [rewards[:min_array_length] for rewards in Rewards]
        if np.mean(Rewards,0)[-1] > best_last_reward:
            best_last_reward = np.mean(Rewards,0)[-1]
            best_policy_index = policy_index
            best_rewards = Rewards
            best_steps = steps
    Rewards = best_rewards
    steps = best_steps
    y = np.mean(Rewards,0)
    yerr = np.std(Rewards,0)/np.sqrt(n_trial)
    plot, = plt.plot(steps,np.mean(Rewards,0),color=colors[group_index])
    plt.fill_between(steps,y+yerr,y-yerr,linewidth=0,facecolor=colors[group_index],alpha=0.3)
    plts.append(plot)
    legends.append(algos[group_index])

axes = plt.gca()
# axes.set_xlim([xmin,xmax])
axes.set_ylim([-50,800])
# plt.title(exp_param_f)
plt.legend(plts,legends, loc='upper left')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Step Number')
# plt.ylabel('Average Top '+str(top_k) +' Reward')    
plt.ylabel('Average Best Return')     
fig.savefig(plot_path+plot_name+'.pdf', bbox_inches='tight')
# from matplotlib2tikz import save as tikz_save
# tikz_save(plot_path+plot_name+'.tex')
plt.close(fig)