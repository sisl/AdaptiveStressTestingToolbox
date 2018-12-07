import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 20})
from matplotlib import pyplot as plt
import numpy as np

n_trial = 5
top_k = 10
batch_size = 4000
max_step = 4000000#np.inf

prepath = "../"
exps = ["CartpoleNd"]
# policies = ["MCTS_RS","MCTS_AS","MCTS_BV","RLInter","GAInter"]
# plot_name = "MCTS_TRPO_GA"
# policies = ["GAInter","GAISInterStep0.01Fmax","GAISNInterStep0.01Fmax"]
# plot_name = 'GA_IS'
# policies = ["GAInterStep0.1","GAISInterStep0.01Fmax","GAISNInterStep0.01Fmax"]
# plot_name = 'GA_TR'
# policies = ["GAInter","GAISInter","GAISNInter","GATRInter",\
#                 "GATRInter_kl01","GATRISInter_kl01","GATRISNInter_kl01",\
#                 "GATRInterstep0.5anneal1.0","GATRISInterstep0.5anneal1.0","GATRISNInterstep0.5anneal1.0"]
# plot_name = "GA"
# policies = ["GAInter","GAMeanInter","GAISInter","GAISInterStep0.01Fmax",\
#                 "GAISNInter","GAISNInterStep0.01Fmax","GATRISInter","GATRISInterStep0.01Fmax",\
#                 "GATRISNInter","GATRISNInterStep0.01Fmax"]
# plot_name = 'GA_max_mean'
# policies = ["GAInterStep0.001Fmax","GAInterStep0.005Fmax","GAInter",\
#             "GAInterStep0.1","GAInterStep0.5","GAInterStep1.0",\
#             "GAInterStep5.0Fmax","GAInterStep10.0Fmax"]
# plot_name = "GA_step"
# policies = ["GATRInterStep0.001Fmax","GATRInterStep0.005Fmax","GATRInter",\
#             "GATRInter_kl01","GATRInterstep0.5Fmax","GATRInterStep1.0Fmax",\
#             "GATRInterStep5.0Fmax","GATRInterStep10.0Fmax",\
#             "GATRInterStep50.0Fmax","GATRInterStep100.0Fmax"]
# plot_name = "GATR_step"
# policies = ["GATRISInterStep0.01Fmax","GATRISInterStep0.1Fmax","GATRISInterStep1.0Fmax",\
#             "GATRISInterStep10.0Fmax","GATRISInterStep50.0Fmax",]
# plot_name = "GATRISFmax_step"
# policies = ["GATRISNInterStep0.01Fmax","GATRISNInterStep0.1Fmax","GATRISNInterStep1.0Fmax",\
#             "GATRISNInterStep10.0Fmax","GATRISNInterStep50.0Fmax",]
# plot_name = "GATRISNFmax_step"
# policies = ["GATRInterstep0.5anneal1.0","GATRISInterstep0.5anneal1.0","GATRISNInterstep0.5anneal1.0",\
#                 "GATRInterstep0.5anneal0.95","GATRISInterstep0.5anneal0.95","GATRISNInterstep0.5anneal0.95"]
# plot_name = "GA_kl_anneal"
# policies = ["GAInter","GAInterStep0.5Anneal1.0","GASMInter","GASMInterStep0.5Anneal1.0"]
# plot_name = "GA_SM"
# policies = ["GAInterStep0.1","GATRInterStep100.0Fmax",\
#             "GADeterInterStep0.01Fmax","GADeterInterStep0.1Fmax","GADeterInterStep1.0Fmax"]
# plot_name = "GA_deter"
# policies = ["GADeterInterStep0.01Fmax","GADeterInterStep0.1Fmax","GADeterInterStep1.0Fmax",\
#                 "GADeterInterStep10.0Fmax","GADeterInterStep50.0Fmax",\
#                 "GATRDInterStep0.01Fmax","GATRDInterStep0.1Fmax","GATRDInterStep1.0Fmax",\
#                 "GATRDInterStep10.0Fmax","GATRDInterStep50.0Fmax"]
# plot_name = "GA_TRD"
# policies = ["PSMCTSInterStep0.01Ec100.0K0.5A0.85","PSMCTSInterStep0.1Ec100.0K0.5A0.85","PSMCTSInterStep1.0Ec100.0K0.5A0.85",\
#                 "PSMCTSTRInterStep0.01Ec100.0K0.5A0.85","PSMCTSTRInterStep0.1Ec100.0K0.5A0.85","PSMCTSTRInterStep1.0Ec100.0K0.5A0.85",\
#                 "PSMCTSTRCInterStep0.01Ec100.0K0.5A0.85","PSMCTSTRCInterStep0.1Ec100.0K0.5A0.85","PSMCTSTRCInterStep1.0Ec100.0K0.5A0.85"]
# plot_name = "PSMCTS_Ec100.0K0.5A0.85"
# policies = ["PSMCTSInterStep0.01Ec100.0K0.5A0.85","PSMCTSInterStep0.01Ec100.0K0.5A0.85InitP100"]
# plot_name = "PSMCTS_initPop"
# policies = ["PSMCTSInterStep0.01Ec100.0K0.5A0.85","PSMCTSInterStep0.01Ec10.0K0.5A0.85","PSMCTSInterStep0.01Ec1.0K0.5A0.85",\
#             "PSMCTSTRCInterStep0.01Ec100.0K0.5A0.85","PSMCTSTRCInterStep0.01Ec10.0K0.5A0.85","PSMCTSTRCInterStep0.01Ec1.0K0.5A0.85",]
# plot_name = "PSMCTS_ec"
# policies = ["PSMCTSInterStep0.01Ec100.0K0.5A0.5","PSMCTSInterStep0.01Ec10.0K0.5A0.5","PSMCTSInterStep0.01Ec1.0K0.5A0.5",\
#             "PSMCTSTRInterStep0.01Ec100.0K0.5A0.5","PSMCTSTRInterStep0.01Ec10.0K0.5A0.5","PSMCTSTRInterStep0.01Ec1.0K0.5A0.5",\
#             "PSMCTSTRCInterStep0.01Ec100.0K0.5A0.5","PSMCTSTRCInterStep0.01Ec10.0K0.5A0.5","PSMCTSTRCInterStep0.01Ec1.0K0.5A0.5",]
# plot_name = "PSMCTS_alpha05"
# policies = ["PSMCTSInterStep0.01Ec100.0K0.5A0.85","PSMCTSInterStep0.01Ec10.0K0.5A0.85","PSMCTSInterStep0.01Ec1.0K0.5A0.85",\
#             "PSMCTSTRCInterStep0.01Ec100.0K0.5A0.85","PSMCTSTRCInterStep0.01Ec10.0K0.5A0.85","PSMCTSTRCInterStep0.01Ec1.0K0.5A0.85",]
# plot_name = "PSMCTS_ec"
# policies = ["PSMCTSMQInterStep1.0Ec100.0K0.5A0.5","PSMCTSMQInterStep1.0Ec10.0K0.5A0.5","PSMCTSMQInterStep1.0Ec1.0K0.5A0.5",\
#             "PSMCTSTRCMQInterStep1.0Ec100.0K0.5A0.5","PSMCTSTRCMQInterStep1.0Ec10.0K0.5A0.5","PSMCTSTRCMQInterStep1.0Ec1.0K0.5A0.5"]
# plot_name = "PSMCTS_MQ"
# policies = ["GATRDInterStep1.0Fmax","GATRInterStep100.0Fmax","PSMCTSMQInterStep1.0Ec1.0K0.5A0.5",]
# plot_name = "PSMCTS_GA"
# policies = ["PSMCTSInterStep0.01Ec100.0K0.5A0.5","PSMCTSInterStep0.01Ec10.0K0.5A0.5","PSMCTSInterStep0.01Ec1.0K0.5A0.5",\
#             "PSMCTSInterStep1.0Ec100.0K0.5A0.5","PSMCTSInterStep1.0Ec10.0K0.5A0.5","PSMCTSInterStep1.0Ec1.0K0.5A0.5"]
# plot_name = "PSMCTS_step"
# policies = ["PSMCTSMQInterStep1.0Ec1.0K0.5A0.5","PSMCTSMQInterStep10.0Ec1.0K0.5A0.5","PSMCTSMQInterStep50.0Ec1.0K0.5A0.5",\
#             "PSMCTSTRMQInterStep10.0Ec1.0K0.5A0.5","PSMCTSTRMQInterStep50.0Ec1.0K0.5A0.5",\
#             "PSMCTSTRCMQInterStep1.0Ec1.0K0.5A0.5","PSMCTSTRCMQInterStep10.0Ec1.0K0.5A0.5","PSMCTSTRCMQInterStep50.0Ec1.0K0.5A0.5"]
# plot_name = "PSMCTSMQ_step"
# policies = ["PSMCTSInterStep1.0Ec1.0K0.5A0.5","PSMCTSInterStep10.0Ec1.0K0.5A0.5",\
#             "PSMCTSTRInterStep1.0Ec1.0K0.5A0.5","PSMCTSTRInterStep10.0Ec1.0K0.5A0.5",\
#             "PSMCTSTRCInterStep1.0Ec1.0K0.5A0.5","PSMCTSTRCInterStep10.0Ec1.0K0.5A0.5"]
# plot_name = "PSMCTS_Qmax"
# policies = ["PSMCTSInterStep10.0Ec1.0K0.5A0.5","PSMCTSTRCInterStep1.0Ec1.0K0.5A0.5","GATRDInterStep1.0Fmax"]
# plot_name = "Deter_best"
# policies = ["MCTS_AS","MCTS_BV","MCTS_RS","RLInter","RLLSTMInter",\
#             "GATRDInterStep1.0Fmax","GATRInterStep10.0Fmax",\
#             "PSMCTSInterStep1.0Ec1.414K0.5A0.5Qmax","PSMCTSTRInterStep1.0Ec1.414K0.5A0.5Qmax","PSMCTSTRCInterStep1.0Ec1.414K0.5A0.5Qmax"]
# plot_name = "total"
policies = ["PSMCTSTRCInterStep1.0Ec1.0K0.5A0.5","MCTS_BV","RLInter","GATRDInterStep1.0Fmax"]
plot_name = "cs332report1"
legends = ["PSMCTS","MCTSBV","TRPO","GA"]
# policies = ["PSMCTSTRCInterStep1.0Ec1.0K0.5A0.5",\
#             "PSMCTSInterStep1.0Ec1.0K0.5A0.5","PSMCTSTRInterStep1.0Ec1.0K0.5A0.5"]
# plot_name = "cs332report2"
# legends = ["PSMCTS","PSMCTS*","PSMCTS**"]
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# colors = []
# for i in range(len(policies)):
#     colors.append('C'+str(i))

for exp in exps:
    plts = []
    # legends = []
    fig = plt.figure(figsize=(10, 10))

    for (policy_index,policy) in enumerate(policies):
        print(policy)
        Rewards = []
        min_array_length = np.inf
        for trial in range(n_trial):
            file_path = prepath+exp+'/Data/AST/Lexington/'+policy+'/'+str(trial)+'/process.csv'
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
                                    avg_top += max(0.0,float(row[entry_dict["reward "+str(k)]]))
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
        # plot, = plt.plot(steps,np.mean(Rewards,0),color=colors[policy_index])
        plot, = plt.plot(steps,np.mean(Rewards,0))
        plts.append(plot)
        # legends.append(policy)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend(plts,legends,loc='lower right')
    plt.xlabel('Step Number')
    plt.ylabel('Average Top '+str(top_k) +' Reward')        
    fig.savefig(prepath+exp+'/Data/Plot/avgtop10/'+plot_name+'_avgtop10.pdf')
    plt.close(fig)