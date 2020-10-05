import pickle
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

from ast_toolbox.algos.go_explore import *


def convert_drl_itr_data_to_expert_trajectory(last_iter_data):
    best_rollout_idx = np.argmax(np.array([np.sum(rollout['rewards']) for rollout in last_iter_data['paths']]))
    best_rollout = last_iter_data['paths'][best_rollout_idx]
    expert_trajectory = []
    collision_step = 1 + np.amax(np.nonzero(best_rollout['rewards']))
    if collision_step == best_rollout['rewards'].shape[0]:
        print('NO COLLISION FOUND IN ANY TRAJECTORY - NOT SAVING EXPERT TRAJECTORY')
    else:
        for step_num in range(collision_step + 1):
            expert_trajectory_step = {}
            expert_trajectory_step['action'] = best_rollout['env_infos']['actions'][step_num, :]
            expert_trajectory_step['observation'] = best_rollout['observations'][step_num, :]
            expert_trajectory_step['reward'] = best_rollout['rewards'][step_num]
            expert_trajectory_step['state'] = best_rollout['env_infos']['state'][step_num, :]

            expert_trajectory.append(expert_trajectory_step)

    return expert_trajectory


def convert_mcts_itr_data_to_expert_trajectory(best_actions, sim, s_0, reward_function):
    expert_trajectories = []
    reward_sums = []
    for actions in best_actions:
        sim.reset(s_0=s_0)
        expert_trajectory = []
        reward_sum = 0
        for action in actions:
            # Run a simulation step
            observation = sim.step(action)
            state = sim.clone_state()
            reward = reward_function.give_reward(
                action=action,
                info=sim.get_reward_info())

            # Save out the step info in the correct format
            expert_trajectory_step = {}
            expert_trajectory_step['action'] = action
            expert_trajectory_step['observation'] = observation
            expert_trajectory_step['reward'] = reward
            expert_trajectory_step['state'] = state

            expert_trajectory.append(expert_trajectory_step)

            reward_sum += reward

        expert_trajectories.append(expert_trajectory)
        reward_sums.append(reward_sum)

    # Sort the lists by total reward an return the best on that ends in collision
    sum_reward_sorted_expert_trajectories = [
        list(x) for x in zip(*sorted(zip(reward_sums, expert_trajectories), key=itemgetter(0)))][1]
    for expert_trajectory in sum_reward_sorted_expert_trajectories:
        import pdb
        pdb.set_trace()
        if expert_trajectory[-1]['reward'] == 0:
            return expert_trajectory

    # No expert trajectory ended in collision, return empty list
    return []


def load_convert_and_save_drl_expert_trajectory(last_iter_filename, expert_trajectory_filename):
    with open(last_iter_filename, 'rb') as f:
        last_iter_data = pickle.load(f)

    expert_trajectory = convert_drl_itr_data_to_expert_trajectory(last_iter_data=last_iter_data)

    if len(expert_trajectory) > 0:
        with open(expert_trajectory_filename, 'wb') as f:
            pickle.dump(expert_trajectory, f)


def load_convert_and_save_mcts_expert_trajectory(best_actions_filename,
                                                 expert_trajectory_filename,
                                                 sim,
                                                 s_0,
                                                 reward_function):
    with open(best_actions_filename, 'rb') as f:
        best_actions = pickle.load(f)

    expert_trajectory = convert_mcts_itr_data_to_expert_trajectory(best_actions=best_actions,
                                                                   sim=sim,
                                                                   s_0=s_0,
                                                                   reward_function=reward_function)

    if len(expert_trajectory) > 0:
        with open(expert_trajectory_filename, 'wb') as f:
            pickle.dump(expert_trajectory, f)


def get_meta_filename(filename):
    return filename + '_meta.dat'


def get_pool_filename(filename):
    return filename + '_pool.dat'


def get_cellpool(filename, dbname=None, dbtype=db.DB_HASH, flags=db.DB_CREATE, protocol=pickle.HIGHEST_PROTOCOL):
    # def open_pool(self, dbname=None, dbtype=db.DB_HASH, flags=db.DB_CREATE, protocol=pickle.HIGHEST_PROTOCOL):
    # We can't save our database as a class attribute due to pickling errors.
    # To prevent errors from code repeat, this convenience function opens the database and
    # loads the latest meta data, the returns the database.
    cell_pool_db = db.DB()
    cell_pool_db.open(get_pool_filename(filename), dbname=dbname, dbtype=dbtype, flags=flags)
    cell_pool_shelf = shelve.Shelf(cell_pool_db, protocol=protocol)
    return cell_pool_shelf


def get_metadata(filename):
    metadata = None
    with contextlib.suppress(FileNotFoundError):
        with open(get_meta_filename(filename), "rb") as f:
            metadata = pickle.load(f)

    return metadata


def plot_goal_trajectories(filename, goal_limit=None, sort_by_reward=False):
    plot_trajectories(filename, plot_terminal=False, plot_goal=True, terminal_limit=None,
                      goal_limit=goal_limit, sort_by_reward=sort_by_reward)


def plot_terminal_trajectories(filename, terminal_limit=None, sort_by_reward=False):
    plot_trajectories(filename, plot_terminal=True, plot_goal=False, terminal_limit=terminal_limit,
                      goal_limit=None, sort_by_reward=sort_by_reward)


def plot_trajectories(filename, plot_terminal=True, plot_goal=True, terminal_limit=None, goal_limit=None,
                      sort_by_reward=False):
    cell_pool_shelf = get_cellpool(filename)
    metadata = get_metadata(filename)

    if plot_terminal:
        terminal_dict = metadata['terminal_dict']

        if sort_by_reward:
            terminal_dict = {k: v for k, v in sorted(terminal_dict.items(), key=lambda item: item[1])}

        terminal_key_list = terminal_dict.keys()

        if terminal_limit is None:
            terminal_limit = len(terminal_key_list)

        for key_idx, key in enumerate(terminal_key_list):

            if key_idx >= terminal_limit:
                break

            sys.stdout.write("\rProcessing Terminal Trajectory {0} / {1}".format(key_idx, terminal_limit))
            sys.stdout.flush()

            cell = cell_pool_shelf[key]
            ped_trajectory = cell.state[11:13].copy().reshape((1, 2))

            while (cell.parent is not None):
                cell = cell_pool_shelf[cell.parent]
                ped_trajectory = np.concatenate((cell.state[11:13].copy().reshape((1, 2)), ped_trajectory))
                plt.plot(ped_trajectory[:, 0], ped_trajectory[:, 1], color=(.1, .9, .1))

    if plot_goal:
        goal_dict = metadata['goal_dict']

        if sort_by_reward:
            goal_dict = {k: v for k, v in sorted(goal_dict.items(), key=lambda item: item[1])}

        goal_key_list = goal_dict.keys()

        if goal_limit is None:
            goal_limit = len(goal_key_list)

        for key_idx, key in enumerate(goal_key_list):

            if key_idx >= goal_limit:
                break

            sys.stdout.write("\rProcessing Goal Trajectory {0} / {1}".format(key_idx, goal_limit))
            sys.stdout.flush()

            cell = cell_pool_shelf[key]
            ped_trajectory = cell.state[11:13].copy().reshape((1, 2))

            while (cell.parent is not None):
                cell = cell_pool_shelf[cell.parent]
                ped_trajectory = np.concatenate((cell.state[11:13].copy().reshape((1, 2)), ped_trajectory))
                plt.plot(ped_trajectory[:, 0], ped_trajectory[:, 1], color=(.9, .1, .1))

    plt.show()

    # def plot_trajectories_from_actions(filename, f_actions_to_trajectories):


#     cell_pool_shelf = get_cellpool(filename)
#     metadata = get_metadata(filename)
#
#     root_cell = get_root_cell(pool, rcell)
#     sim.restore_state(root_cell.state)
#     actions = rcell.trajectory[:,1:].astype(np.float32) / 1000.0

# trajectories = f_actions_to_trajectories()


def get_root_cell(pool, cell):
    root_cell = cell
    while (root_cell.parent is not None):
        root_cell = pool[root_cell.parent]

    return root_cell


def render(car=None, ped=None, noise=None, ped_obs=None, gif=False):
    if gif:
        return
    else:
        fig = plt.figure(constrained_layout=True)
        if noise is not None and (car is not None or ped is not None or ped_obs is not None):
            # Plotting Noise and trajectories
            gs = GridSpec(4, 2, figure=fig)
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[2, 1])
            ax5 = fig.add_subplot(gs[3, 1])

            ax1.set_title('Car, Actual Pedestrian, and Observed Pedestrian Trajectories')
        elif noise is not None:
            #     Only plotting noise
            gs = GridSpec(4, 1, figure=fig)
            ax2 = fig.add_subplot(gs[0, 0])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[2, 0])
            ax5 = fig.add_subplot(gs[3, 0])
        else:
            #     Only plotting trajectories
            gs = GridSpec(1, 1, figure=fig)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title('Car, Actual Pedestrian, and Observed Pedestrian Trajectories')
        if noise is not None:
            axs = [ax2, ax3, ax4, ax5]
            ax_titles = ['Noise: Pedestrian X Velocity', 'Noise: Pedestrian Y Velocity',
                         'Noise: Pedestrian X Position', 'Noise: Pedestrian Y Position']
            x = np.arange(noise.shape[0] + 2)
            for idx, ax in enumerate(axs):
                y = np.concatenate(([0], noise[:, idx], [0]))

                ax.step(x, y, color='black', where='pre')
                ax.axhline(0, color='black', lw=2)

                ycopy = y.copy()
                ycopy[y < 0] = 0
                ax.fill_between(x, ycopy, facecolor='green', step='pre')

                ycopy = y.copy()
                ycopy[y >= 0] = 0
                ax.fill_between(x, ycopy, facecolor='red', step='pre')
                ax.set_title(ax_titles[idx])

        if ped is not None:
            ax1.quiver(ped[:, 2], ped[:, 3], ped[:, 0], ped[:, 1], scale=50)

            ped_final_pos = ped[-1, 2:]
            rad = 0.125

            circle = Circle((ped_final_pos[0], ped_final_pos[1]), radius=rad)
            pc = PatchCollection([circle], facecolor='blue', alpha=0.2,
                                 edgecolor='blue')
            ax1.add_collection(pc)

        if ped_obs is not None:
            ax1.quiver(ped_obs[:, 2], ped_obs[:, 3], ped_obs[:, 0], ped_obs[:, 1], scale=50, color='gray')
        if car is not None:
            ax1.quiver(car[:, 2], car[:, 3], car[:, 0], car[:, 1], scale=500)

            car_final_pos = car[-1, 2:]
            x_dist = 2.5
            y_dist = 1.4

            rect = Rectangle((car_final_pos[0] - x_dist / 2, car_final_pos[1] - y_dist / 2), x_dist, y_dist)
            pc = PatchCollection([rect], facecolor='red', alpha=0.2,
                                 edgecolor='red')
            ax1.add_collection(pc)

    plt.show()
    return
