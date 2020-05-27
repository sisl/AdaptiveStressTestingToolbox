import csv

from matplotlib import pyplot as plt

pre_path = './Data/AST/Lexington/GATRISInter/'
plot_name = "GATRISInter"
n_trial = 5

fig = plt.figure()
for i in range(n_trial):
    csv_path = pre_path + str(i) + '/process.csv'
    plt.subplot(2, 3, i + 1)
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        SumOtherWeights = []
        iterations = []
        for (i, row) in enumerate(csv_reader):
            print(i)
            if i == 0:
                entry_dict = {}
                for index in range(len(row)):
                    entry_dict[row[index]] = index
            else:
                SumOtherWeights.append(float(row[entry_dict["SumOtherWeights"]]))
                iterations.append(int(row[entry_dict["Iteration"]]))

    plt.scatter(iterations, SumOtherWeights, s=0.1, marker=',')
    plt.xlabel('Iteration')
    plt.ylabel('SumOtherWeights')
fig.savefig('Data/Plot/sumOtherWeights/' + plot_name + '_SumOtherWeights.pdf')
plt.close(fig)
