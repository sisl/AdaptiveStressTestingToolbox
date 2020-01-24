import matplotlib.pyplot as plt
import numpy as np
from bsddb3 import db
import pickle
import shelve
import os, sys
import contextlib
import pdb


def get_meta_filename(filename):
    return filename+'_meta.dat'

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
    plot_trajectories(filename, plot_terminal=False, plot_goal=True, terminal_limit=None, goal_limit=goal_limit, sort_by_reward=sort_by_reward)

def plot_terminal_trajectories(filename, terminal_limit=None, sort_by_reward=False):
    plot_trajectories(filename, plot_terminal=True, plot_goal=False, terminal_limit=terminal_limit, goal_limit=None, sort_by_reward=sort_by_reward)

def plot_trajectories(filename, plot_terminal=True, plot_goal=True, terminal_limit=None, goal_limit=None, sort_by_reward=False):
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
            ped_trajectory = cell.state[11:13].copy().reshape((1,2))

            while(cell.parent is not None):
                cell = cell_pool_shelf[cell.parent]
                ped_trajectory = np.concatenate((cell.state[11:13].copy().reshape((1,2)), ped_trajectory))
                plt.plot(ped_trajectory[:,0], ped_trajectory[:,1], color=(.1, .9, .1))

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
    while(root_cell.parent is not None):
        root_cell=pool[root_cell.parent]

    return root_cell


