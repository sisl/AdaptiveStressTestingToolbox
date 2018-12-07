import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 22})
from matplotlib import pyplot as plt
import numpy as np

n_trial = 5
top_k = 10
batch_size = 4000
# max_step = 5e6

prepath = "../"
exps = ["CartpoleNdRewardt"]
policies = ["GATRDInterStep1.0Fmax","PSMCTSInterStep1.0Ec1.414K0.5A0.5Qmax",\
            "PSMCTSTRInterStep1.0Ec1.414K0.5A0.5Qmax","PSMCTSTRCInterStep1.0Ec1.414K0.5A0.5Qmax"]
plot_name = "GATRD_PSMCTS"

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
colors = []
for i in range(len(policies)):
    colors.append('C'+str(i))

for exp in exps:
    plts = []
    legends = []
    fig = plt.figure(figsize=(10, 10))
    for (policy_index,policy) in enumerate(policies):
        print(policy)
        Rewards = []
        min_array_length = np.inf
        for trial in range(n_trial):
            print(trial)
            steps = []
            rewards = []
            file_path = prepath+exp+'/Data/AST/Lexington/'+policy+'/'+str(trial)+'/process.csv'
            if os.path.exists(file_path):
                with open(file_path) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for (i,row) in enumerate(csv_reader):
                        if i == 0:
                            entry_dict = {}
                            for index in range(len(row)):
                                entry_dict[row[index]] = index
                        else:
                            # if int(row[entry_dict["StepNum"]]) > max_step:
                                # break
                            if int(row[entry_dict["StepNum"]])%batch_size == 0:
                                steps.append(int(row[entry_dict["StepNum"]]))
                                rewards.append(max(0.0,float(row[entry_dict["reward 0"]])))
            if len(rewards) < min_array_length:
                min_array_length = len(rewards) 
            Rewards.append(rewards)
        steps = steps[:min_array_length]
        Rewards = [rewards[:min_array_length] for rewards in Rewards]
        plot, = plt.plot(steps,np.mean(Rewards,0),color=colors[policy_index])
        plts.append(plot)
        legends.append(policy)

    plt.legend(plts,legends)
    plt.xlabel('Step number')
    plt.ylabel('Average Best reward')        
    fig.savefig(prepath+exp+'/Data/Plot/avgtop1/'+plot_name+'_avgtop1.pdf')
    plt.close(fig)