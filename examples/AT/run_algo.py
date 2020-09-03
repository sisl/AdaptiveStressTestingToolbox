# This file kicks off separate runs AT given different algorithms and seeds.
# To run this from the REPL: exec(open("run_algo.py").read())

import os
from shutil import copyfile

AT = "AT2" # "AT1" or "AT2"
os.environ["AT"] = AT

for i in range(10):
    for algo in ["mcts", "drl", "random"]:
        os.environ["ALGO"] = algo
        os.environ["SEED"] = str(i)
        print("Running ALGO: ", algo, " with SEED:", i)
        exec(open("example_batch_runner_at.py").read())
        copyfile("data/" + algo + "/progress.csv", "data/jmlr_at_case_study/" + algo + "_progress_" + AT + "_" + str(i) + ".csv")