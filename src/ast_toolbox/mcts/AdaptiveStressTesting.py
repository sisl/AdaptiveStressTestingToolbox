import pickle

import numpy as np
# import garage.misc.logger as logger
from dowel import logger
from dowel import tabular

import ast_toolbox.mcts.MDP as MDP


class ASTParams:
    """Structure that stores internal parameters for AST.

    Parameters
    ----------
    max_steps : int, optional
        The maximum search depth.

    """

    def __init__(self, max_steps, log_interval, log_tabular, log_dir=None, n_itr=100):
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.log_tabular = log_tabular
        self.log_dir = log_dir
        self.n_itr = n_itr


class AdaptiveStressTest:
    """The AST wrapper for MCTS using the actions in env.action_space.

    Parameters
    ----------
    p : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTParams`
        The AST parameters
    env : :py:class:`ast_toolbox.envs.go_explore_ast_env.GoExploreASTEnv`.
        The environment.
    top_paths : :py:class:`ast_toolbox.mcts.BoundedPriorityQueues`, optional
        The bounded priority queue to store top-rewarded trajectories.
    """

    def __init__(self, p, env, top_paths):
        self.params = p
        self.env = env
        self.sim_hash = hash(0)
        self.transition_model = self.transition_model()
        self.step_count = 0
        self._isterminal = False
        self._reward = 0.0
        self.action_seq = []
        self.trajectory_reward = 0.0
        self.top_paths = top_paths
        self.iter = 0

    def reset_step_count(self):
        """Reset the env step count.
        """
        self.step_count = 0

    def initialize(self):
        """Initialize training variables.

        Returns
        ----------
        env_reset :
            The reset result from the env.
        """
        self._isterminal = False
        self._reward = 0.0
        self.action_seq = []
        self.trajectory_reward = 0.0
        return self.env.reset()

    def update(self, action):
        """Update the environment as well as the assosiated parameters.

        Parameters
        ----------
        action : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTAction`
            The AST action.

        Returns
        ----------
        obs : :py:class:`numpy.ndarry`
            The observation from the env step.
        reward : float
            The reward from the env step.
        done : bool
            The terminal indicator from the env step.
        info : dict
            The env info from the env step.
        """
        self.step_count += 1
        obs, reward, done, info = self.env.step(action.get())
        self._isterminal = done
        self._reward = reward
        self.action_seq.append(action)
        self.trajectory_reward += reward
        if done:
            self.top_paths.enqueue(self.action_seq, self.trajectory_reward, make_copy=True)
        self.logging()
        return obs, reward, done, info

    def logging(self):
        """Logging the training information.
        """
        if self.params.log_tabular and self.iter <= self.params.n_itr:
            if self.step_count % self.params.log_interval == 0:
                self.iter += 1
                logger.log(' ')
                tabular.record('StepNum', self.step_count)
                record_num = 0
                if self.params.log_dir is not None:
                    if self.step_count == self.params.log_interval:  # first time logging
                        best_actions = []
                    else:
                        with open(self.params.log_dir + '/best_actions.p', 'rb') as f:
                            best_actions = pickle.load(f)

                    best_actions.append(np.array([x.get() for x in self.top_paths.pq[0][0]]))
                    with open(self.params.log_dir + '/best_actions.p', 'wb') as f:
                        pickle.dump(best_actions, f)

                for (topi, path) in enumerate(self.top_paths):
                    tabular.record('reward ' + str(topi), path[1])
                    record_num += 1

                for topi_left in range(record_num, self.top_paths.N):
                    tabular.record('reward ' + str(topi_left), 0)
                logger.log(tabular)
                logger.dump_all(self.step_count)
                tabular.clear()

    def isterminal(self):
        """Check whether the current path is finished.

        Returns
        ----------
        isterinal : bool
            Whether the current path is finished.
        """
        return self._isterminal

    def get_reward(self):
        """Get the current AST reward.

        Returns
        ----------
        reward : bool
            The AST reward.
        """
        return self._reward

    def random_action(self):
        """Randomly sample an action for the rollout.

        Returns
        ----------
        action : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTAction`
            The sampled action.
        """
        return ASTAction(self.env.action_space.sample())

    def explore_action(self, s, tree):
        """Randomly sample an action for the exploration.

        Parameters
        ----------
        s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
            The current state.
        tree : dict
            The searching tree.

        Returns
        ----------
        action : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTAction`
            The sampled action.
        """
        return ASTAction(self.env.action_space.sample())

    def transition_model(self):
        """Generate the transition model used in MCTS.

        Returns
        ----------
        transition_model : :py:class:`ast_toolbox.mcts.MDP.TransitionModel`
            The transition model.
        """
        def get_initial_state():
            self.t_index = 1
            self.initialize()
            s = ASTState(self.t_index, None, None)
            self.sim_hash = s.hash
            return s

        def get_next_state(s0, a0):
            assert self.sim_hash == s0.hash
            self.t_index += 1
            self.update(a0)
            s1 = ASTState(self.t_index, s0, a0)
            self.sim_hash = s1.hash
            r = self.get_reward()
            return s1, r

        def isterminal(s):
            assert self.sim_hash == s.hash
            return self.isterminal()

        def go_to_state(target_state):
            s = get_initial_state()
            actions = get_action_sequence(target_state)
            # print("go to state with actions: ",actions)
            R = 0.0
            for a in actions:
                s, r = get_next_state(s, a)
                R += r
            assert s == target_state
            return R, actions
        return MDP.TransitionModel(get_initial_state, get_next_state, isterminal, self.params.max_steps, go_to_state)


class ASTState:
    """The AST state.

    Parameters
    ----------
    t_index : int
        The index of the timestep.
    parent : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The parent state.
    action : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTAction`
        The action leading to this state.
    """

    def __init__(self, t_index, parent, action):
        self.t_index = t_index
        self.parent = parent
        self.action = action
        self.hash = hash(self)

    def __hash__(self):
        """The redefined hashing method.

        Returns
        ----------
        hash : int
            The hashing result.
        """
        if self.parent is None:
            return hash((self.t_index, None, hash(self.action)))
        else:
            return hash((self.t_index, self.parent.hash, hash(self.action)))

    def __eq__(self, other):
        """The redefined equal method.

        Returns
        ----------
        is_equal : bool
            Whether the two states are equal.
        """
        return hash(self) == hash(other)


class ASTAction:
    def __init__(self, action):
        """The AST action.

        Parameters
        ----------
        action :
            The true actions used in the env.
        """
        self.action = action

    def __hash__(self):
        """The redefined hashing method.

        Returns
        ----------
        hash : int
            The hashing result.
        """
        return hash(tuple(self.action))

    def __eq__(self, other):
        """The redefined equal method.

        Returns
        ----------
        is_equal : bool
            Whether the two states are equal.
        """
        return np.array_equal(self.action, other.action)

    def get(self):
        """Get the true action.

        Returns
        ----------
        action :
            The true actions used in the env.
        """
        return self.action


def get_action_sequence(s):
    """Get the action sequence that leads to the state.

    Parameters
    ----------
    s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The target state.

    Returns
    ----------
    actions : list[:py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTAction`]
        The action sequences leading to the target state.
    """
    actions = []
    while s.parent is not None:
        actions.append(s.action)
        s = s.parent
    actions = list(reversed(actions))
    return actions
