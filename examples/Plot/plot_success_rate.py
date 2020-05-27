import csv
import os.path

from matplotlib import pyplot as plt

# bug! the xlabel are not aligned
top_k = 10
batch_size = 4000
max_step = 5e6

date = "Lexington"
exps = ["cartpole"]
plolicies = ["MCTS_RS", "MCTS_AS", "MCTS_BV", "RLInter", "RLNonInter", "GAInter", "GANonInter"]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

for exp in exps:
    success_rates = []
    legends = []
    for (policy_index, policy) in enumerate(plolicies):
        print(policy)
        trial_num = 0.0
        success = 0.0
        file_path = 'Data/AST/' + date + '/' + policy + '/' + 'total_result.csv'
        if os.path.exists(file_path):
            with open(file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                file_empty = True
                for (i, row) in enumerate(csv_reader):
                    file_empty = False
                    print(i)
                    if i != 0:
                        trial_num += 1.0
                        if float(row[1]) > 0.0:
                            success += 1.0
                if file_empty is False:
                    success_rates.append(success / trial_num)
                    legends.append(policy)
    fig = plt.figure(figsize=(20, 20))
    ax = plt.gca()
    plt.scatter(range(len(legends)), success_rates)
    ax.set_xticklabels(legends)
    plt.ylabel('success rate')
    fig.savefig('Data/Plot/' + 'success_rate.pdf')
    plt.close(fig)
