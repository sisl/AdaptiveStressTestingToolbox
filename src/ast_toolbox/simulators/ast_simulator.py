"""Class template to wrap a simulator for interaction with AST."""


class ASTSimulator(object):
    """Class template to wrap a simulator for interaction with AST.

    This class already tracks the simulator options to return the correct observation type. In addition,
    `max_path_length` and `self._path_length` are handled by this parent class.

    Parameters
    ----------
    blackbox_sim_state : bool, optional
        True if the true simulation state can not be observed, in which case actions and the initial conditions are
        used as the observation. False if the simulation state can be observed, in which case it will be used.
    open_loop : bool, optional
        True if the simulation is open-loop, meaning that AST must generate all actions ahead of time, instead
        of being able to output an action in sync with the simulator, getting an observation back before
        the next action is generated. False to get interactive control, which requires that `blackbox_sim_state`
        is also False.
    fixed_init_state : bool, optional
        True if the initial state is fixed, False to sample the initial state for each rollout from the observaation
        space.
    max_path_length : int, optional
        Maximum length of a single rollout.
    """

    def __init__(self,
                 blackbox_sim_state=True,
                 open_loop=True,
                 fixed_initial_state=True,
                 max_path_length=50):

        self.c_max_path_length = max_path_length

        self.blackbox_sim_state = blackbox_sim_state
        self.open_loop = open_loop
        self.fixed_initial_state = fixed_initial_state

        self._is_terminal = False
        self.initial_conditions = None
        self.observation = None

        self._path_length = 0

    def simulate(self, actions, s_0):
        """Run a full simulation given the AST solver's actions and initial conditions.

        `simulate` takes in the AST solver's actions and the initial conditions. It should return two values: a
        terminal index and an array of relevant simulation information.

        Parameters
        ----------
        actions : list[array_like]
            A sequential list of actions taken by the AST Solver which deterministically control the simulation.
        s_0 : array_like
            An array specifying the initial conditions to set the simulator to.

        Returns
        -------
        terminal_index : int
            The index of the action that resulted in a state in the goal set E. If no state is found
            terminal_index should be returned as -1.
        array_like
            An array of relevant simulator info, which can then be used for analysis or diagnostics.

        """
        raise NotImplementedError

    def step(self, action):
        """Step the simulation forward in time.

        `step` takes in a the actions that deterministically control a single step forward in the simulation. It
        checks to see if the rollout horizon has been reached, and then calls `closed_loop_step` if the simulation
        is set to `open_loop == False`.

        Parameters
        ----------
        action : array_like
            A 1-D array of actions taken by the AST Solver which deterministically control
            a single step forward in the simulation.

        Returns
        -------
        array_like
            An observation from the timestep, which is either from the simulator if `open_loop` is False and
            `blackbox_sim_state` is True, or else the initial conditions.

        """
        self._path_length += 1
        if self._path_length >= self.c_max_path_length:
            self._is_terminal = True

        if not self.open_loop:
            return self.closed_loop_step(action)

        return self.initial_conditions

    def closed_loop_step(self, action):
        """User implemented function to step the simulation forward in time when closed-loop control is active.

        This function should step the simulator forward a single timestep based on the given action. It will only
        be called when `open_loop` is False. This function should always return `self.observation_return()`.

        Parameters
        ----------
        action : array_like
            A 1-D array of actions taken by the AST Solver which deterministically control
            a single step forward in the simulation.

        Returns
        -------
        array_like
            An observation from the timestep, determined by the settings and the `observation_return` helper function.
        """
        return self.observation_return()

    def reset(self, s_0):
        """Resets the state of the environment, returning an initial observation.

        User implementations should always call the super class implementation.
        This function should always return `self.observation_return()`.

        Parameters
        ----------
        s_0 : array_like
            The initial conditions to reset the simulator to.

        Returns
        -------
        array_like
            An observation from the timestep, determined by the settings and the `observation_return` helper function.
        """
        self.initial_conditions = s_0
        self._is_terminal = False
        self._path_length = 0

        return self.observation_return()

    def observation_return(self):
        """
        Helper function to return the correct observation based on settings.

        Returns
        -------
        array_like
            An observation from the timestep, which is either from the simulator if `open_loop` is False and
            `blackbox_sim_state` is True, or else the initial conditions.
        """
        if not self.blackbox_sim_state:
            return self.observation

        return self.initial_conditions

    def get_reward_info(self):
        """
        Returns any info needed by the reward function to calculate the current reward.
        """
        raise NotImplementedError

    def is_goal(self):
        """
        Returns whether the current state is in the goal set.
        Returns
        -------
        bool
            True if current state is in goal set.
        """
        raise NotImplementedError

    def is_terminal(self):
        """
        Returns whether rollout horizon has been reached.
        Returns
        -------
        bool
            True if rollout horizon has been reached.
        """
        return self._is_terminal

    def log(self):
        """
        perform any logging steps
        """

    def clone_state(self):
        """Clone the simulator state for later resetting.

        This function is used in conjunction with `restore_state` for Go-Explore and Backwards Algorithm
        to do their deterministic resets.

        Returns
        -------
        array_like
            An array of all the simulation state variables.

        """

    def restore_state(self, in_simulator_state):
        """Reset the simulation deterministically to a previously cloned state.

        This function is used in conjunction with `clone_state` for Go-Explore and Backwards Algorithm
        to do their deterministic resets.

        Parameters
        ----------
        in_simulator_state : array_like
            An array of all the simulation state variables.

        """

    def render(self, **kwargs):
        """Either renders a simulation scene or returns data used for external rendering.

        Parameters
        ----------
        kwargs :
            Keyword arguments used in the simulators `render` function.
        """
