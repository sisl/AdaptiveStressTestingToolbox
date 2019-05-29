# Import the example classes
from mylab.rewards.heuristic_reward import HeuristicReward
from mylab.spaces.benchmark_spaces import BenchmarkSpaces

# Import RSS stuff
from mylab.simulators.benchmark_simulator import BenchmarkSimulator
from mylab.rewards.benchmark_action_model import samp_traj, traj_md, traj_dist, BenchmarkActionModel

# Import the AST classes
from mylab.envs.ast_env import ASTEnv
from mylab.algos.mcts import MCTS
import mylab.mcts.BoundedPriorityQueues as BPQ

# Useful imports
import os.path as osp
import argparse
from example_save_trials import *
import tensorflow as tf

# Logger Params
parser = argparse.ArgumentParser()

parser.add_argument('--iters', type=int, default=101)
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--clip_range', type=float, default=0.3)

# Env Params
parser.add_argument('--action_only', type=bool, default=True)
parser.add_argument('--fixed_init_state', type=bool, default=False)

# MCTS Params
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--k', type=float, default=0.5)
parser.add_argument('--ec', type=float, default=1000.0)

args = parser.parse_args()

# Generate trajectories of interest and plot them
ndim = 2
T = 10
nT = 10
E = [samp_traj(dim=ndim, T=T) for i in range(nT)]
probs = [traj_md(Ei) for Ei in E]


import matplotlib.pyplot as plt
import matplotlib.colors
cmap = plt.cm.viridis
norm = matplotlib.colors.Normalize(vmin=np.min(probs), vmax=np.max(probs))

fig, ax = plt.subplots()
for i in range(nT):
	ax.plot(E[i][:,0], E[i][:,1], color=cmap(norm(probs[i])))

plt.title("Trajectories in $E$ Colored by Probability")
plt.xlabel("$a_1$")
plt.ylabel("$a_2$")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm)
cbar.set_label("$\log(P)$")

plt.savefig("target_trajectories.png")

print(np.vstack(E).shape)
amin = np.min(np.vstack(E), axis=0)
amax = np.max(np.vstack(E), axis=0)
T = E[0].shape[0]
print("amin: ", amin, " amax: ", amax)

tol = 1
sim = BenchmarkSimulator(E, traj_dist, tol, max_path_length = T)

reward_function = HeuristicReward(BenchmarkActionModel(), np.array([-10000, -1]))
spaces = BenchmarkSpaces(amin, amax, T)

# Create the environment
top_k = 10
top_paths = BPQ.BoundedPriorityQueue(top_k)


s_0 = np.zeros((0, ndim))
env = ASTEnv(fixed_init_state=True,
			 s_0=s_0,
			 simulator=sim,
			 reward_function=reward_function,
			 spaces=spaces
			 )

algo = MCTS(
	    env=env,
		stress_test_num=2,
		max_path_length=T,
		ec=args.ec,
		n_itr = 10000,
		k=args.k,
		alpha=args.alpha,
		clear_nodes=False,
		log_interval=1000,
	    top_paths=top_paths,
	    plot_tree=False,
	    plot_path='./tree'
	    )


result = algo.train()

# Write out the episode results
# header = 'trial, step, ' + 'v_x_car, v_y_car, x_car, y_car, '
# for i in range(0, sim.c_num_peds):
# 	header += 'v_x_ped_' + str(i) + ','
# 	header += 'v_y_ped_' + str(i) + ','
# 	header += 'x_ped_' + str(i) + ','
# 	header += 'y_ped_' + str(i) + ','
#
# for i in range(0, sim.c_num_peds):
# 	header += 'a_x_' + str(i) + ','
# 	header += 'a_y_' + str(i) + ','
# 	header += 'noise_v_x_' + str(i) + ','
# 	header += 'noise_v_y_' + str(i) + ','
# 	header += 'noise_x_' + str(i) + ','
# 	header += 'noise_y_' + str(i) + ','
#
# header += 'reward'

# print("saving top trials to disk")
# print('header size: ', len(header.split(",")))
# trial = 0
# X = np.zeros((0,17))
i=0
for (reward_predict, action_seq) in result:
	traj = np.vstack([a.get() for a in action_seq])
	ax.plot(traj[:, 0], traj[:, 1], label="Best Trajectory", color = "red")
	i = i+1
	break

ax.legend()
plt.savefig("found_trajectories.png")


	# trial += 1
	# sim.reset(s_0)
	# rewards = []
	# for a in action_seq:
	# 	action = a.get()
	# 	sim.step(action)
	# 	reward = reward_function.give_reward(action, info=sim.get_reward_info())
	# 	rewards.append(reward)
	# 	if sim._is_terminal or sim.is_goal():
	# 		break


	# print("trial ", trial, " reward total: ", np.sum(rewards))
	# assert len(rewards) == len(sim._info)
	# for i in range(len(sim._info)):
	# 	sim._info[i][0] = trial
	# 	sim._info[i][16] = rewards[i]
	#
	# X = np.vstack((X, np.array(sim._info)))

# print("X shape: ", X.shape)
# filename = args.log_dir + '/mcts_crashes.csv'
# print("writing file to ", filename)
# np.savetxt(fname=filename,
# 		   X=X,
# 		   delimiter=',',
# 		   header=header)
# n = np.zeros((50,6))
# for i in range(50):
# 	n[i,:] = spaces.action_space.sample()
#
# print(np.mean(n, axis=0))

# sim = ExampleSimulator(simulatorSettings)
# reward_function = ExampleReward(rewardSettings)
# spaces = ExampleSpaces(limits)
# env = ASTEnv(simulator=sim,
# 			 reward_function=reward_function,
# 			 spaces=spaces
# 			 )
# algo = RLAlgorithm(policy=Policy,
# 				   optimizer=Serializable
# 	    )

