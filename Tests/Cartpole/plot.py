import csv
from matplotlib import pyplot as plt
import numpy as np

n_trial = 5
top_k = 10

exps = ["cartpole"]
plolicies = ["MCTS","MCTS_AS","MCTS_BV","RLInter","RLNonInter","GAInter","GANonInter"]

for exp in exps:
    plts_r = [] #reward
    success_rates = []
    for policy in plolicies:
        print(policy)
        success = 0.0
        with open('Data/AST/CSV/'+exp+'_'+policy+'.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            rewards = np.zeros((n_trial,top_k))
            for (i,row) in enumerate(csv_reader):
                print(i)
                if i != 0:
                    for j in range(len(row)):
                        if j!= 0:
                            rewards[i-1,j-1] = float(row[j])
                    if rewards[i-1,0] > -5e3:
                        success += 1.0
            success_rates.append(success/n_trial)
            plts_r.append(plt.scatter(range(top_k),np.max(rewards[:,:],axis=0)))

    fig = plt.figure()
    plt.legend(plts_r,plolicies)
    plt.xlabel('top i')
    plt.ylabel('reward')        
    fig.savefig('Data/Plot/'+exp+'.pdf')
    plt.close(fig)

    fig = plt.figure()
    ax = plt.gca()
    plt.scatter(range(len(plolicies)),success_rates)
    ax.set_xticklabels(plolicies)
    plt.ylabel('success rate')        
    fig.savefig('Data/Plot/'+exp+'_sr.pdf')
    plt.close(fig)