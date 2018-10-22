import csv
from matplotlib import pyplot as plt
import numpy as np

n_trial = 2
top_k = 10
batch_size = 4000
max_step = 5e6

data = "Oct21"
exps = ["cartpole"]
plolicies = ["MCTS_RS","MCTS_AS","MCTS_BV","RLInter","RLNonInter","GAInter","GANonInter"]

for exp in exps:
    plts = []
    legends = []
    fig = plt.figure()
    for policy in plolicies:
        print(policy)
        for trial in range(n_trial):
            print(trial)
            steps = []
            rewards = []
            with open('Data/AST/'+data+'/'+policy+'/'+str(trial)+'/process.csv') as csv_file:
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
                            rewards.append(max(0.0,float(row[entry_dict["reward 0"]])))
                plot, = plt.plot(steps,rewards)
                plts.append(plot)
                legends.append(policy+' '+str(trial))

    plt.legend(plts,legends)
    plt.xlabel('Step number')
    plt.ylabel('Best reward')        
    fig.savefig('Data/Plot/'+exp+data+'_top1progress'+'.pdf')
    plt.close(fig)