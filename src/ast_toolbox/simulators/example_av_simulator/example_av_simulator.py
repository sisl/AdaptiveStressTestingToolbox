"""Example simulator wrapper for a scenario of an AV approaching a crosswalk where some pedestrians are crossing."""
import numpy as np  # Used for math

from ast_toolbox.simulators import ASTSimulator  # import parent Simulator class
from ast_toolbox.simulators.example_av_simulator import ToyAVSimulator  # import the simulator to wrap


class ExampleAVSimulator(ASTSimulator):  # Define the class
    """Example simulator wrapper for a scenario of an AV approaching a crosswalk where some pedestrians are crossing.

    Wraps :py:class:`ast_toolbox.simulators.example_av_simulator.ToyAVSimulator`

    Parameters
    ----------
    num_peds : int
        Number of pedestrians crossing the street.
    simulator_args : dict
        Dictionary of keyword arguments to be passed to the wrapped simulator.
    kwargs :
        Keyword arguments passed to the super class.
    """

    def __init__(self,
                 num_peds=1,
                 simulator_args=None,
                 **kwargs):

        # Constant hyper-params -- set by user
        self.c_num_peds = num_peds
        if simulator_args is None:
            simulator_args = {}

        self._action = np.array([0] * (6 * self.c_num_peds))
        self.simulator = ToyAVSimulator(num_peds=num_peds, **simulator_args)

        # initialize the base Simulator
        super().__init__(**kwargs)

    def get_first_action(self):
        """ An initialization method used in Go-Explore.

        Returns
        -------
        array_like
            A 1-D array of the same dimension as the action space, all zeros.
        """
        return np.array([0] * (6 * self.c_num_peds))

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
        return self.simulator.run_simulation(actions=actions, s_0=s_0, simulation_horizon=self.c_max_path_length)

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
        # grab simulation state, if interactive
        self.observation = np.ndarray.flatten(self.simulator.step_simulation(action))

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

        # Call ASTSimulator's reset function (required!)
        super(ExampleAVSimulator, self).reset(s_0=s_0)
        # Reset the simulation
        self.observation = np.ndarray.flatten(self.simulator.reset(s_0))

        return self.observation_return()

    def get_reward_info(self):
        """
        Returns any info needed by the reward function to calculate the current reward.
        """
        # Get the ground truth state from the toy simulator
        sim_state = self.simulator.get_ground_truth()

        return {"peds": sim_state['peds'],
                "car": sim_state['car'],
                "is_goal": self.is_goal(),
                "is_terminal": self.is_terminal()}

    def is_goal(self):
        """
        Returns whether the current state is in the goal set.
        Returns
        -------
        bool
            True if current state is in goal set.
        """
        # Ask the toy simulator if a collision was detected
        return self.simulator.collision_detected()

    def log(self):
        """
        Perform any logging steps.
        """
        # Get the ground truth state from the toy simulator
        sim_state = self.simulator.get_ground_truth()

        # Create a cache of step specific variables for post-simulation analysis
        cache = np.hstack([0.0,  # Dummy, will be filled in with trial # during post processing in save_trials.py
                           sim_state['step'],
                           np.ndarray.flatten(sim_state['car']),
                           np.ndarray.flatten(sim_state['peds']),
                           np.ndarray.flatten(sim_state['action']),
                           np.ndarray.flatten(sim_state['car_obs']),
                           0.0])

        self._info.append(cache)

    def clone_state(self):
        """Clone the simulator state for later resetting.

        This function is used in conjunction with `restore_state` for Go-Explore and Backwards Algorithm
        to do their deterministic resets.

        Returns
        -------
        array_like
            An array of all the simulation state variables.

        """
        # Get the ground truth state from the toy simulator
        simulator_state = self.simulator.get_ground_truth()

        return np.concatenate((np.array([simulator_state['step']]),
                               np.array([simulator_state['path_length']]),
                               np.array([int(simulator_state['is_terminal'])]),
                               simulator_state['car'],
                               simulator_state['car_accel'],
                               simulator_state['peds'].flatten(),
                               simulator_state['car_obs'].flatten(),
                               simulator_state['action'].flatten(),
                               simulator_state['initial_conditions']), axis=0)

    def restore_state(self, in_simulator_state):
        """Reset the simulation deterministically to a previously cloned state.

        This function is used in conjunction with `clone_state` for Go-Explore and Backwards Algorithm
        to do their deterministic resets.

        Parameters
        ----------
        in_simulator_state : array_like
            An array of all the simulation state variables.

        """
        # Put the simulators state variables in dict form
        simulator_state = {}

        simulator_state['step'] = in_simulator_state[0]
        simulator_state['path_length'] = in_simulator_state[1]
        simulator_state['is_terminal'] = bool(in_simulator_state[2])
        simulator_state['car'] = in_simulator_state[3:7]
        simulator_state['car_accel'] = in_simulator_state[7:9]
        peds_end_index = 9 + self.c_num_peds * 4
        simulator_state['peds'] = in_simulator_state[9:peds_end_index].reshape((self.c_num_peds, 4))
        car_obs_end_index = peds_end_index + self.c_num_peds * 4
        simulator_state['car_obs'] = in_simulator_state[peds_end_index:car_obs_end_index].reshape((self.c_num_peds, 4))
        simulator_state['action'] = in_simulator_state[car_obs_end_index:car_obs_end_index + self._action.shape[0]]
        simulator_state['initial_conditions'] = in_simulator_state[car_obs_end_index + self._action.shape[0]:]

        # Set ground truth of actual simulator
        self.simulator.set_ground_truth(simulator_state)

        # Set wrapper state variables
        self._info = []
        self.initial_conditions = np.array(simulator_state['initial_conditions'])
        self._is_terminal = simulator_state['is_terminal']
        self._path_length = simulator_state['path_length']
