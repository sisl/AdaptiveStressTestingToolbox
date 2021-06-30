import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 25, 'font.family':'Times New Roman',\
                            'text.usetex': True})
from matplotlib import pyplot as plt
import numpy as np

import csv

n_trial = 20
top_k = 1
batch_size = 1000
max_step = 5e6#np.inf
max_reward = np.inf
min_reward = -np.inf

exp_names = [
            'Acrobot',\
            'BipedalWalker',\
            'MountainCar'
            ]
exp_params = [
            # ['L100Th19999new','L100Th1999new','L100Th199new','L100Th19new'],\
            # ['L100TL25new','L100TL30new'],\
            # ['L100P0002new','L100P00015new',]
            ['L100Th1999new'],\
            ['L100TL30new'],\
            ['L100P00015new',]
            ]
exp_names_f = [
            'Acrobot',\
            'BipedalWalker',\
            'MountainCar'
            ]
# exp_params_f = [
#             ['$y_\\mathrm{goal} = 1.9999$','$y_\\mathrm{goal} = 1.999$','$y_\\mathrm{goal} = 1.99$','$y_\\mathrm{goal} = 1.9$'],\
#             ['$x_\\mathrm{goal} = 25$','$x_\\mathrm{goal} = 30$'],\
#             ['$p_\\mathrm{car} = 0.002$','$p_\\mathrm{car} = 0.0015$']
#             ]
extra_name = ''#'hyper'
policy_groups = [
                [
                "TRPOStep0.1",\
                "TRPOStep1.0"],\
                [
                "GATRDP100T20K3Step1.0Fmean",\
                "GATRDP500T20K3Step1.0Fmean",\
                "GATRDP1000T20K3Step1.0Fmean"
                ],\
                [
                "PSMCTSTRCK0.3A0.3Ec1.414Step1.0FmeanQmax",\
                "PSMCTSTRCK0.5A0.5Ec1.414Step1.0FmeanQmax",\
                "PSMCTSTRCK0.8A0.8Ec1.414Step1.0FmeanQmax"]\
                ]

algos = ["TRPO","Deep GA","MCTSPO"]
parameters = [["step size 0.1","step size 1.0"],\
                ["population size 100","population size 500","population size 1000"],\
                ['$k=\\alpha=0.3$','$k=\\alpha=0.5$','$k=\\alpha=0.8$'],
                ]
colors = ["blue","green","red"]
    # plot_path = "../"+exp_name+"/Data/Plot/avgtop"+str(top_k)+"/"
plot_path = "/Users/xiaobaima/Dropbox/SISL/MCTSPO/IJCAI/presentation/Plots/ClassicControl/"
csv_path = "/Users/xiaobaima/Dropbox/SISL/MCTSPO/IJCAI/presentation/Plots/ClassicControl/stats.csv"

csv_file = open(csv_path, mode='w')
field_names = ["Evironment"]
for policies in policy_groups:
    for policy in policies:
        field_names.append(policy)
writer = csv.DictWriter(csv_file, fieldnames=field_names)
writer.writeheader()

for (exp_index,exp_name) in enumerate(exp_names):
    for (param_index,exp_param)in enumerate(exp_params[exp_index]): 

        prepath = "../"+exp_name+"/Data/AST/Lexington/"+exp_param
        plot_name = exp_name+'_'+exp_param+'avgtop'+str(top_k)+'trial'+str(n_trial)+extra_name


        plts = []
        legends = []
        fig = plt.figure(figsize=(10, 10))
        fig.set_size_inches(10.0, 10.0)

        row_content = dict()
        row_content["Evironment"] = exp_name+exp_param

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

                row_content[policy] = np.mean(Rewards,0)[-1]

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
            legends.append(algos[group_index]+' '+parameters[group_index][best_policy_index])

        writer.writerow(row_content)

        axes = plt.gca()
        # axes.set_xscale('log')
        # axes.set_xlim([xmin,xmax])
        axes.set_ylim([-0.15,1.0])
        # plt.title(exp_names_f[exp_index]+' '+exp_params_f[exp_index][param_index])
        plt.title(exp_names_f[exp_index])
        # plt.legend(plts,legends)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel('Step Number')
        # plt.ylabel('Average Top '+str(top_k) +' Return')
        # plt.ylabel('Average Best Return')        
        fig.savefig(plot_path+plot_name+'.pdf', bbox_inches='tight')
        # from matplotlib2tikz import save as tikz_save
        # tikz_save(plot_path+plot_name+'.tex')
        plt.close(fig)