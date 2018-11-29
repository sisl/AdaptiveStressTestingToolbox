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
exps = ["CartpoleNd"]
# plolicies = ["MCTS_RS","MCTS_AS","MCTS_BV","RLInter","GAInter","GAISInter","GAISNInter","GATRInter","GATRISInter"]
# plot_name = "Total"
# plolicies = ["GAInter","GAISInter","GAISNInter","GATRInter",\
#                 "GATRInter_kl01","GATRISInter_kl01","GATRISNInter_kl01",\
#                 "GATRInterstep0.5anneal1.0","GATRISInterstep0.5anneal1.0","GATRISNInterstep0.5anneal1.0"]
# plot_name = "GA"
# plolicies = ["GAInter","GANonInter","GAMeanInter","GAMeanNonInter"]
# plot_name = 'GA_max_mean'
# plolicies = ["GATRInter","GATRInter_kl01","GATRISInter_kl01","GATRISNInter_kl01",\
#                 "GATRInterstep0.5anneal1.0","GATRISInterstep0.5anneal1.0","GATRISNInterstep0.5anneal1.0"]
# plot_name = "GA_kl"
# plolicies = ["GATRInterstep0.5anneal1.0","GATRISInterstep0.5anneal1.0","GATRISNInterstep0.5anneal1.0",\
#                 "GATRInterstep0.5anneal0.95","GATRISInterstep0.5anneal0.95","GATRISNInterstep0.5anneal0.95"]
# plot_name = "GA_kl_anneal"
# plolicies = ["GAInter","GASMInter"]
# plot_name = "GA_SM"

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
colors = []
for i in range(len(plolicies)):
    colors.append('C'+str(i))

for exp in exps:
    plts = []
    legends = []
    fig = plt.figure(figsize=(10, 10))
    for (policy_index,policy) in enumerate(plolicies):
        print(policy)
        Rewards = []
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
            Rewards.append(rewards)
        plot, = plt.plot(steps,np.mean(Rewards,0),color=colors[policy_index])
        plts.append(plot)
        legends.append(policy)

    plt.legend(plts,legends)
    plt.xlabel('Step number')
    plt.ylabel('Average Best reward')        
    fig.savefig(prepath+exp+'/Data/Plot/avgtop1/'+plot_name+'_avgtop1.pdf')
    plt.close(fig)