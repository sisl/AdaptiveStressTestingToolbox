# Import the example classes
from mylab.rewards.heuristic_reward import HeuristicReward
from mylab.spaces.benchmark_spaces import BenchmarkSpaces

# Import RSS stuff
from mylab.simulators.benchmark_simulator import BenchmarkSimulator
from mylab.rewards.benchmark_action_model import samp_traj, traj_md, traj_dist, BenchmarkActionModel

# Import the AST classes
from mylab.envs.ast_env import ASTEnv
from mylab.samplers.ast_vectorized_sampler import ASTVectorizedSampler
from garage.tf.algos.trpo import TRPO
from garage.tf.envs.base import TfEnv
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.envs.normalized_env import normalize

# Useful imports
import os.path as osp
import argparse
from example_save_trials import *
import tensorflow as tf
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='benchmar_exp')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

# Generate trajectories of interest and plot them
ndim = 2
T = 5
nT = 3
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

tol = 0.1
sim = BenchmarkSimulator(E, traj_dist, tol, max_path_length = T)

reward_function = HeuristicReward(BenchmarkActionModel(), np.array([-10000, -1]))
spaces = BenchmarkSpaces(amin, amax, T)

# Create the environment
s_0 = np.zeros((ndim))
env = TfEnv(normalize(ASTEnv(action_only=True,
                             fixed_init_state=True,
                             s_0=s_0,
                             simulator=sim,
                             reward_function=reward_function,
                             spaces=spaces
                             )))

# Instantiate the garage objects
policy = GaussianLSTMPolicy(name='lstm_policy',
                            env_spec=env.spec,
                            hidden_dim=64,
                            use_peepholes=True)
baseline = LinearFeatureBaseline(env_spec=env.spec)
sampler_cls = ASTVectorizedSampler
optimizer = ConjugateGradientOptimizer
optimizer_args = {'hvp_approach':FiniteDifferenceHvp(base_eps=1e-5)}
algo = TRPO(
    env=env,
    policy=policy,
    baseline=LinearFeatureBaseline(env_spec=env.spec),
    batch_size=5000,
    step_size=0.1,
    n_itr=101,
    store_paths=True,
    optimizer=optimizer,
    optimizer_args=optimizer_args,
    max_path_length=T,
    sampler_cls=sampler_cls,
    sampler_args={"sim": sim,
                  "reward_function": reward_function})


with tf.Session() as sess:
    # Run the experiment
    algo.train(sess=sess)

    # data = joblib.load(args.log_dir + '/itr_' + str(i) + '.pkl')
    # paths = data['paths']
    # for n, a_path in enumerate(paths):
    #     cache = a_path['env_infos']['info']['cache']
    #     cache[:, 0] = n
    #     trials = np.concatenate((trials, cache), axis=0)
    #     if cache[-1, -1] == 0.0:
    #         crashes = np.concatenate((crashes, cache), axis=0)

# i=0
# for (reward_predict, action_seq) in result:
# 	traj = np.vstack([a.get() for a in action_seq])
# 	ax.plot(traj[:, 0], traj[:, 1], label="T"+str(i), color = "red")
# 	i = i+1
# 	break
#
# ax.legend()
# plt.savefig("found_trajectories.png")
