import csv
import os.path
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
# plolicies = ["GAInter","GAMeanInter","GAISInter","GAISInterStep0.01Fmax",\
#                 "GAISNInter","GAISNInterStep0.01Fmax","GATRISInter","GATRISInterStep0.01Fmax",\
#                 "GATRISNInter","GATRISNInterStep0.01Fmax"]
# plot_name = 'GA_max_mean'
# plolicies = ["GATRInter","GATRInter_kl01","GATRISInter_kl01","GATRISNInter_kl01",\
#                 "GATRInterstep0.5anneal1.0","GATRISInterstep0.5anneal1.0","GATRISNInterstep0.5anneal1.0",\
#                 "GATRInterStep1.0Anneal1.0"]
# plot_name = "GA_kl"
# plolicies = ["GAInterStep0.001Fmax","GAInterStep0.005Fmax","GAInter",\
#             "GAInterStep0.1Anneal1.0","GAInterStep0.5Anneal1.0","GAInterStep1.0Anneal1.0",\
#             "GAInterStep5.0Fmax","GAInterStep10.0Fmax"]
# plot_name = "GA_step"
# plolicies = ["GATRInterStep0.001Fmax","GATRInterStep0.005Fmax","GATRInter",\
#             "GATRInter_kl01","GATRInterstep0.5Fmax","GATRInterStep1.0Fmax",\
#             "GATRInterStep5.0Fmax","GATRInterStep10.0Fmax",\
#             "GATRInterStep50.0Fmax","GATRInterStep100.0Fmax"]
# plot_name = "GATR_step"
# plolicies = ["GATRInterstep0.5anneal1.0","GATRISInterstep0.5anneal1.0","GATRISNInterstep0.5anneal1.0",\
#                 "GATRInterstep0.5anneal0.95","GATRISInterstep0.5anneal0.95","GATRISNInterstep0.5anneal0.95"]
# plot_name = "GA_kl_anneal"
# plolicies = ["GAInter","GAInterStep0.5Anneal1.0","GASMInter","GASMInterStep0.5Anneal1.0"]
# plot_name = "GA_SM"
# plolicies = ["GAInter","GAISNInter","GAISNInterStep0.5Anneal1.0",\
#             "GATRInter","GATRInterStep0.5Anneal1.0","GATRInterStep1.0Anneal1.0"]
# plot_name = "GA_best"
# plolicies = ["GADeterInterStep0.01Fmax","GADeterInterStep0.1Fmax","GADeterInterStep1.0Fmax",\
#                 "GATRDInterStep0.01Fmax","GATRDInterStep0.1Fmax","GATRDInterStep1.0Fmax"]
# plot_name = "GA_deter"
# plolicies = ["PSMCTSInterStep0.01Ec100.0K0.5A0.85","PSMCTSInterStep0.1Ec100.0K0.5A0.85","PSMCTSInterStep1.0Ec100.0K0.5A0.85",\
#                 "PSMCTSTRInterStep0.01Ec100.0K0.5A0.85","PSMCTSTRInterStep0.1Ec100.0K0.5A0.85","PSMCTSTRInterStep1.0Ec100.0K0.5A0.85",\
#                 "PSMCTSTRCInterStep0.01Ec100.0K0.5A0.85","PSMCTSTRCInterStep0.1Ec100.0K0.5A0.85","PSMCTSTRCInterStep1.0Ec100.0K0.5A0.85"]
# plot_name = "PSMCTS"
plolicies = ["PSMCTSInterStep0.01Ec100.0K0.5A0.85","PSMCTSInterStep0.01Ec100.0K0.5A0.85InitP100"]
plot_name = "PSMCTS_initPop"

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# colors = []
# for i in range(len(plolicies)):
#     colors.append('C'+str(i))

for exp in exps:
    plts = []
    legends = []
    fig = plt.figure(figsize=(10, 10))
    for (policy_index,policy) in enumerate(plolicies):
        print(policy)
        Rewards = []
        min_array_length = np.inf
        for trial in range(n_trial):
            steps = []
            rewards = []
            file_path = prepath+exp+'/Data/AST/Lexington/'+policy+'/'+str(trial)+'/process.csv'
            if os.path.exists(file_path):
                print(trial)
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
                                avg_top = 0.0
                                for k in range(top_k):
                                    avg_top += max(0.0,float(row[entry_dict["reward "+str(k)]]))
                                avg_top /= top_k
                                rewards.append(avg_top)
                print(len(rewards))
                print(steps[-1])
                if len(rewards) < min_array_length:
                    min_array_length = len(rewards) 
                Rewards.append(rewards)
        steps = steps[:min_array_length]
        Rewards = [rewards[:min_array_length] for rewards in Rewards]
        # plot, = plt.plot(steps,np.mean(Rewards,0),color=colors[policy_index])
        plot, = plt.plot(steps,np.mean(Rewards,0))
        plts.append(plot)
        legends.append(policy)

    plt.legend(plts,legends)
    plt.xlabel('Step number')
    plt.ylabel('Average Top '+str(top_k) +' reward')        
    fig.savefig(prepath+exp+'/Data/Plot/avgtop10/'+plot_name+'_avgtop10.pdf')
    plt.close(fig)