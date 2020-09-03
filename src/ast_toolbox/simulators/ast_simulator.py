# from garage.envs.base import GarageEnv
# from garage.envs.base import Step
# from garage.spaces import Box


class ASTSimulator(object):
    """
    Class template for a non-interactive simulator.
    """

    def __init__(self,
                 blackbox_sim_state=True,
                 open_loop=True,
                 fixed_initial_state=True,
                 max_path_length=50):
        """
        :function goal_set - function definition that accepts a state, and returns true if state is in set.
        :parameter  s_0 - the initial state of the simulator.
        """
        self.c_max_path_length = max_path_length

        self.blackbox_sim_state = blackbox_sim_state
        self.open_loop = open_loop
        self.fixed_initial_state = fixed_initial_state

        self._is_terminal = False
        self.initial_conditions = None
        self.observation = None

        self._path_length = 0

    def simulate(self, actions, s_0):
        """
        Run/finish the simulation
        Input
        -----
        actions : A sequential list of actions taken by the simulation
        Outputs
        -------
        (terminal_index)
        terminal_index : The index of the action that resulted in a state in the goal set E. If no state is found
                        terminal_index should be returned as -1.

        """
        raise NotImplementedError

    def step(self, action):
        """
        Handle anything that needs to take place at each step, such as a simulation update or write to file
        Input
        -----
        action : action taken on the turn
        Outputs
        -------
        observation : The true state simulation
        s_0 : the intitial condition of the run

        """
        self._path_length += 1
        if self._path_length >= self.c_max_path_length:
            self._is_terminal = True

        if not self.open_loop:
            return self.closed_loop_step(action)

        return self.initial_conditions

    def closed_loop_step(self, action):
        raise NotImplementedError

    def reset(self, s_0):
        """
        Resets the state of the environment, returning an initial observation.
        Inputs
        -------
        s_0: the initial state
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.initial_conditions = s_0
        self._is_terminal = False
        self._path_length = 0

        return self.observation_return()

    def observation_return(self):
        """
        Helper function to return the correct observation based on settings

        :param obs: True simulation state observation
        :param s0: Initial state of the
        :return: Either current simulation state or the initial conditions
        """
        if not self.blackbox_sim_state:
            return self.observation

        return self.initial_conditions

    def get_reward_info(self):
        """
        returns any info needed by the reward function to calculate the current reward
        """
        raise NotImplementedError

    def is_goal(self):
        """
        returns whether the current state is in the goal set
        :return: boolean, true if current state is in goal set.
        """
        raise NotImplementedError

    def is_terminal(self):
        return self._is_terminal

    def log(self):
        """
        perform any logging steps
        """
