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
min_reward = -0.05#-np.inf

prepath = "../AcrobotStoch/Data/AST/Lexington"
exp = "AcrobotStoch"
plot_path = "../AcrobotStoch/Data/Plot/BestMean/"
policies = ["TRPO",\
        "GATRDStep1.0Fmean","GATRDStep0.1Fmean","GATRDStep0.01Fmean",\
        "PSMCTSStep1.0Ec1.414K0.5A0.5Qmax","PSMCTSStep0.1Ec1.414K0.5A0.5Qmax","PSMCTSStep0.01Ec1.414K0.5A0.5Qmax",\
        "PSMCTSTRCStep1.0Ec1.414K0.5A0.5Qmax","PSMCTSTRCStep0.1Ec1.414K0.5A0.5Qmax","PSMCTSTRCStep0.01Ec1.414K0.5A0.5Qmax"]
plot_name = exp

# prepath = "../CartpoleNdRewardt/Data/AST/Lexington"
# exp = "CartpoleNdRewardt"
# plot_path = "../CartpoleNdRewardt/Data/Plot/Avgtop10/"
# policies = ["GATRDInterStep1.0Fmax","PSMCTSTRCInterStep1.0Ec1.414K0.5A0.5Qmax",\
#             "PSMCTSTRCInterStep1.0Ec1.414K0.5A0.3Qmax","PSMCTSTRCInterStep1.0Ec1.414K0.5A0.1Qmax",\
#             "PSMCTSTRCInterStep1.0Ec1.414K0.5A0.8Qmax","PSMCTSTRCInterStep1.0Ec0.1K0.5A0.5Qmax",\
#             "PSMCTSTRCInterStep1.0Ec0.5K0.5A0.5Qmax","PSMCTSTRInterStep1.0Ec1.414K0.5A0.5Qmax",\
#             "PSMCTSTRCInterStep1.0Ec0.1K0.5A0.3Qmax"]
# plot_name = "PSMCTSTRC"
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# colors = []
# for i in range(len(policies)):
#     colors.append('C'+str(i))


plts = []
legends = []
fig = plt.figure(figsize=(10, 10))

for (policy_index,policy) in enumerate(policies):
    print(policy)
    Means = []
    min_array_length = np.inf
    for trial in range(n_trial):
        file_path = prepath+'/'+policy+'/'+str(trial)+'/process.csv'
        if os.path.exists(file_path):
            print(trial)
            steps = []
            means = []
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
                            mean = np.clip(float(row[entry_dict["BestMean"]]),min_reward,max_reward)
                            means.append(mean)
            print(means[-1]) 
            if len(means) < min_array_length:
                min_array_length = len(means)
            Means.append(means)
    steps = steps[:min_array_length]
    Means = [means[:min_array_length] for means in Means]
    # plot, = plt.plot(steps,np.mean(Rewards,0),color=colors[policy_index])
    plot, = plt.plot(steps,np.mean(Means,0))
    plts.append(plot)
    legends.append(policy)

plt.legend(plts,legends)
plt.xlabel('Step Number')
plt.ylabel('Best Mean')        
fig.savefig(plot_path+plot_name+'_bestmean.pdf')
plt.close(fig)