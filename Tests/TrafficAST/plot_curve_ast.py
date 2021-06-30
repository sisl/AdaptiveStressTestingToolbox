import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

max_itr = np.inf

fields = [
            'reward 0',
            ]
field_names = [
            'Best Return',
            ]

itr_name = 'Step Num'
itr_field = 'StepNum'

min_loss = [0.]*100
max_loss = [np.inf]*100

prepath = "./Data/AST"
plot_path = "./Data/AST"

policies = [
         "MCTSASK0.5A0.5Ec10.0",
         "TRPObs2000lr0.1",
         "GATRDP100T20K3lr1.0Fmean",
         "PSMCTSTRCK0.5A0.5Ec10.0CA4lr1.0FmeanQmax",
         ]
pre_name = ''
post_name = ''

# policy_names = policies
policy_names = [
                "MCTS",
                "TRPO",
                "GATRD",
                "PSMCTSTRC",
            ]
extra_name = 'seed2'
seeds = [2]

colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

for fid,field in enumerate(fields):
    print(field)
    fig = plt.figure(fid)
    legends = []
    plts = []
    for (policy_index,policy) in enumerate(policies):
        policy_path = pre_name+policy+post_name
        Itrs = []
        Losses = []
        min_itr = np.inf
        for trial in seeds:
            folder_path = prepath+'/'+policy_path+'/'+'seed'+str(trial)
            print(folder_path)
            if os.path.exists(folder_path):
                print(policy+'_'+str(trial))
                itrs = []
                losses = []

                file_path = folder_path+'/progress.csv'
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
                            # print(entry_dict)
                        else:
                            itr = int(float(row[entry_dict[itr_field]]))
                            if itr > max_itr:
                                break

                            if field in entry_dict.keys():
                                loss = np.clip(float(row[entry_dict[field]]),
                                                    min_loss[fid],max_loss[fid])
                            else:
                                loss = 0.

                            itrs.append(itr)
                            losses.append(loss)

                if len(losses) < min_itr:
                    min_itr = len(losses)
                Losses.append(losses)
        Losses = [losses[:min_itr] for losses in Losses]
        itrs = itrs[:min_itr]
        Losses = np.array(Losses)
        print(Losses.shape)
        y = np.mean(Losses,0)
        print(y[-1])
        yerr = np.std(Losses,0)
        plot, = plt.plot(itrs,y,colors[policy_index])
        plt.fill_between(itrs,y+yerr,y-yerr,linewidth=0,
                            facecolor=colors[policy_index],alpha=0.3)
        plts.append(plot)
        legends.append(policy_names[policy_index])

    plt.legend(plts,legends,loc='best')
    plt.xlabel(itr_name)
    plt.ylabel(field_names[fid]) 
    fig.savefig(plot_path+'/'+extra_name+field_names[fid]+'.pdf')
    plt.close(fig)