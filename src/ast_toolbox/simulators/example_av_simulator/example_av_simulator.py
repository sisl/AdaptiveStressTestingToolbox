import numpy as np  # Used for math

from ast_toolbox.simulators import ASTSimulator  # import base Simulator class
from ast_toolbox.simulators.example_av_simulator import ToyAVSimulator

# Define the class


class ExampleAVSimulator(ASTSimulator):
    """
    Class template for a non-interactive simulator.
    """
    # Accept parameters for defining the behavior of the system under test[SUT]

    def __init__(self,
                 num_peds=1,
                 simulator_args=None,
                 # dt=0.1,
                 # alpha=0.85,
                 # beta=0.005,
                 # v_des=11.17,
                 # delta=4.0,
                 # t_headway=1.5,
                 # a_max=3.0,
                 # s_min=4.0,
                 # d_cmf=2.0,
                 # d_max=9.0,
                 # min_dist_x=2.5,
                 # min_dist_y=1.4,
                 # car_init_x=-35.0,
                 # car_init_y=0.0,
                 # blackbox_sim_state = True,
                 **kwargs):
        # Constant hyper-params -- set by user
        self.c_num_peds = num_peds
        if simulator_args is None:
            simulator_args = {}

        # self.c_dt = dt
        # self.c_alpha = alpha
        # self.c_beta = beta
        # self.c_v_des = v_des
        # self.c_delta = delta
        # self.c_t_headway = t_headway
        # self.c_a_max = a_max
        # self.c_s_min = s_min
        # self.c_d_cmf = d_cmf
        # self.c_d_max = d_max
        # self.c_min_dist = np.array([min_dist_x, min_dist_y])
        # self.c_car_init_x = car_init_x
        # self.c_car_init_y = car_init_y
        # self.blackbox_sim_state = blackbox_sim_state

        # These are set by reset, not the user
        # self._car = np.zeros((4))
        # self._car_accel = np.zeros((2))
        # self._peds = np.zeros((self.c_num_peds, 4))
        # self._measurements = np.zeros((self.c_num_peds, 4))
        # self._car_obs = np.zeros((self.c_num_peds, 4))
        # self._env_obs = np.zeros((self.c_num_peds, 4))
        # self._done = False
        # self._reward = 0.0
        # self._info = []
        # self._step = 0
        # self._path_length = 0
        # # self._action = None
        self._action = np.array([0] * (6 * self.c_num_peds))
        # self._first_step = True
        # self.directions = np.random.randint(2, size=self.c_num_peds) * 2 - 1
        # self.y = np.random.rand(self.c_num_peds) * 14 - 5
        # self.x = np.random.rand(self.c_num_peds) * 4 - 2
        # self._state = None
        # TODO: Handle toy simulator parameters
        self.simulator = ToyAVSimulator(num_peds=num_peds, **simulator_args)

        # initialize the base Simulator
        super().__init__(**kwargs)

    def get_first_action(self):
        return np.array([0] * (6 * self.c_num_peds))

    def simulate(self, actions, s_0):
        """
        Run/finish the simulation
        Input
        -----
        action : A sequential list of actions taken by the simulation
        Outputs
        -------
        (terminal_index)
        terminal_index : The index of the action that resulted in a state in the goal set E. If no state is found
                        terminal_index should be returned as -1.

        """
        return self.simulator.run_simulation(actions=actions, s_0=s_0, simulation_horizon=self.c_max_path_length)

    def closed_loop_step(self, action):
        """
        Handle anything that needs to take place at each step, such as a simulation update or write to file
        Input
        -----
        action : action taken on the turn
        Outputs
        -------
        (terminal_index)
        terminal_index : The index of the action that resulted in a state in the goal set E. If no state is found
                        terminal_index should be returned as -1.

        """
        # grab simulation state, if interactive
        self.observation = np.ndarray.flatten(self.simulator.step_simulation(action))

        return self.observation_return()

    def reset(self, s_0):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """

        # return the initial simulation state
        super(ExampleAVSimulator, self).reset(s_0=s_0)
        self.observation = np.ndarray.flatten(self.simulator.reset(s_0))
        # self.observation = obs
        return self.observation_return()

    def get_reward_info(self):
        """
        returns any info needed by the reward function to calculate the current reward
        """
        sim_state = self.simulator.get_ground_truth()

        return {"peds": sim_state['peds'],
                "car": sim_state['car'],
                "is_goal": self.is_goal(),
                "is_terminal": self.is_terminal()}

    def is_goal(self):
        """
        returns whether the current state is in the goal set
        :return: boolean, true if current state is in goal set.
        """
        return self.simulator.collision_detected()

    def log(self):
        # Create a cache of step specific variables for post-simulation analysis
        sim_state = self.simulator.get_ground_truth()
        cache = np.hstack([0.0,  # Dummy, will be filled in with trial # during post processing in save_trials.py
                           sim_state['step'],
                           np.ndarray.flatten(sim_state['car']),
                           np.ndarray.flatten(sim_state['peds']),
                           np.ndarray.flatten(sim_state['action']),
                           np.ndarray.flatten(sim_state['car_obs']),
                           0.0])
        self._info.append(cache)

    # def sensors(self, car, peds, noise):
    #
    #     measurements = peds + noise
    #     return measurements
    #
    # def tracker(self, observation_old, measurements):
    #     observation = np.zeros_like(observation_old)
    #
    #     observation[:, 0:2] = observation_old[:, 0:2]
    #     observation[:, 2:4] = observation_old[:, 2:4] + self.c_dt * observation_old[:, 0:2]
    #     residuals = measurements[:, 2:4] - observation[:, 2:4]
    #
    #     observation[:, 2:4] += self.c_alpha * residuals
    #     observation[:, 0:2] += self.c_beta / self.c_dt * residuals
    #
    #     return observation

    # def update_car(self, obs, v_car):
    #
    #     cond = np.repeat(np.resize(np.logical_and(obs[:, 3] > -1.5, obs[:, 3] < 4.5), (self.c_num_peds, 1)), 4, axis=1)
    #     in_road = np.expand_dims(np.extract(cond, obs), axis=0)
    #
    #     if in_road.size != 0:
    #         mins = np.argmin(in_road.reshape((-1, 4)), axis=0)
    #         v_oth = obs[mins[3], 0]
    #         s_headway = obs[mins[3], 2] - self._car[2]
    #         s_headway = max(10 ** -6, abs(s_headway)) * np.sign(s_headway)  # avoid div by zero error later
    #
    #         del_v = v_oth - v_car
    #         s_des = self.c_s_min + v_car * self.c_t_headway - v_car * del_v / (2 * np.sqrt(self.c_a_max * self.c_d_cmf))
    #         if self.c_v_des > 0.0:
    #             v_ratio = v_car / self.c_v_des
    #         else:
    #             v_ratio = 1.0
    #
    #         a = self.c_a_max * (1.0 - v_ratio ** self.c_delta - (s_des / s_headway) ** 2)
    #
    #     else:
    #         del_v = self.c_v_des - v_car
    #         a = del_v
    #
    #     if np.isnan(a):
    #         pdb.set_trace()
    #     # pdb.set_trace()
    #     return np.clip(a, -self.c_d_max, self.c_a_max)

    # def move_car(self, car, accel):
    #     car[2:4] += self.c_dt * car[0:2]
    #     car[0:2] += self.c_dt * accel
    #     return car

    # def update_peds(self):
    #     # Update ped state from actions
    #     action = self._action.reshape((self.c_num_peds, 6))[:, 0:2]
    #
    #     mod_a = np.hstack((action,
    #                        self._peds[:, 0:2] + 0.5 * self.c_dt * action))
    #     if np.any(np.isnan(mod_a)):
    #         pdb.set_trace()
    #
    #     self._peds += self.c_dt * mod_a
    #     # Enforce max abs(velocity) on pedestrians
    #     self._peds[:, 0:2] = np.clip(self._peds[:, 0:2], a_min=[-4.5, -4.5], a_max=[4.5, 4.5])
    #     if np.any(np.isnan(self._peds)):
    #         pdb.set_trace()

    # def observe(self):
    #     self._env_obs = self._peds - self._car

    def clone_state(self):
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
        # Set ground truth of actual simulator
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

        self.simulator.set_ground_truth(simulator_state)

        # Set wrapper state variables
        self._info = []
        self.initial_conditions = simulator_state['initial_conditions']
        self._is_terminal = simulator_state['is_terminal']
        self._path_length = simulator_state['path_length']

    def _get_obs(self):
        if self.blackbox_sim_state:
            return np.array(self.initial_conditions)
            # if self._action is None:
            # return np.array([0] * (6*self.c_num_peds))
            # return self._action
        return self.simulator._env_obs

    def render(self, car, ped, noise, gif=False):
        if gif:
            return
        else:
            return
