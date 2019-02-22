import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 15})
from matplotlib import pyplot as plt
import numpy as np

n_trial = 20
top_k = 1
batch_size = 4000
max_step = np.inf
max_reward = np.inf
min_reward = -np.inf

exp_name = 'BipedalWalker'#''Acrobot'#'MountainCar'#CartPole'#'LunarLander'#
exp_param = 'L100TL30new'#'L100Th19999new'#'L100P00015new'#'L100Th1203It02'#'L100R1U0'#
extra_name = ''#'hyper'
prepath = "../"+exp_name+"/Data/AST/Lexington/"+exp_param
plot_path = "../"+exp_name+"/Data/Plot/avgtop"+str(top_k)+"/"
# policies = ["TRPOStep0.1","TRPOStep1.0",\
#         "GATRDP100T20K3Step1.0Fmean","GATRDP100T20K3Step0.1Fmean","GATRDP100T20K3Step0.01Fmean",\
#         "PSMCTSTRCK0.5A0.5Ec1.414Step1.0FmeanQmax","PSMCTSTRCK0.5A0.5Ec1.414Step0.1FmeanQmax","PSMCTSTRCK0.5A0.5Ec1.414Step0.01FmeanQmax"]
# policies = ["TRPOStep0.1",\
#         "GATRDP100T20K3Step1.0Fmean",\
#         "PSMCTSTRCK0.5A0.5Ec1.414Step1.0FmeanQmax"]
# policies = [
#         "TRPOStep0.1","TRPOStep1.0",\
#         "GATRDP50T20K3Step1.0Fmean","GATRDP100T20K3Step1.0Fmean","GATRDP200T20K3Step1.0Fmean",\
#         "GATRDP500T20K3Step1.0Fmean","GATRDP1000T20K3Step1.0Fmean",\
#         "PSMCTSTRCK0.3A0.5Ec1.414Step1.0FmeanQmax","PSMCTSTRCK0.5A0.5Ec1.414Step1.0FmeanQmax","PSMCTSTRCK1.0A0.5Ec1.414Step1.0FmeanQmax",\
#         "PSMCTSTRCK0.3A0.3Ec1.414Step1.0FmeanQmax","PSMCTSTRCK1.0A1.0Ec1.414Step1.0FmeanQmax"
#         ]
policies = [
        "TRPOStep0.1","TRPOStep1.0",\
        "GATRDP100T20K3Step1.0Fmean","GATRDP500T20K3Step1.0Fmean","GATRDP1000T20K3Step1.0Fmean",\
        "PSMCTSTRCK0.5A0.5Ec1.414Step1.0FmeanQmax",\
        "PSMCTSTRCK0.3A0.3Ec1.414Step1.0FmeanQmax","PSMCTSTRCK0.8A0.8Ec1.414Step1.0FmeanQmax"\
        ]
plot_name = exp_name+'_'+exp_param+'avgtop'+str(top_k)+'trial'+str(n_trial)+extra_name


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
    plot, = plt.plot(steps,np.mean(Rewards,0))
    # plot,_,_ = plt.errorbar(steps,np.mean(Rewards,0),yerr=np.std(Rewards,0)/np.sqrt(n_trial),errorevery=10)
    plts.append(plot)
    legends.append(policy)

plt.legend(plts,legends)
plt.xlabel('Step Number')
plt.ylabel('Average Top '+str(top_k) +' Reward')        
fig.savefig(plot_path+plot_name)
plt.close(fig)