import csv
from matplotlib import pyplot as plt
import numpy as np

n_trial = 5
top_k = 10

exps = ["cartpole"]
plolicies = ["MCTS","MCTS_AS","MCTS_BV","RLInter","RLNonInter","GAInter","GANonInter"]

for exp in exps:
    fig = plt.figure()
    plts = []
    for policy in plolicies:
        print(policy)
        with open('Data/AST/'+exp+'_'+policy+'.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            rewards = np.zeros((n_trial,top_k))
            for (i,row) in enumerate(csv_reader):
                print(i)
                if i != 0:
                    for j in range(len(row)):
                        if j!= 0:
                            rewards[i-1,j-1] = float(row[j])
            plts.append(plt.scatter(range(top_k),np.max(rewards[:,:],axis=0)))
    plt.legend(plts,plolicies)
    plt.xlabel('top i')
    plt.ylabel('reward')        
    fig.savefig('Data/Plot/'+exp+'.pdf')
    plt.close(fig)