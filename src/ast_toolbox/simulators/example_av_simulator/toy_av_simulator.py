import pdb  # Used for debugging

import numpy as np  # Used for math

# Define the class


class ToyAVSimulator():
    """
    Class template for a non-interactive simulator.
    """
    # Accept parameters for defining the behavior of the system under test[SUT]

    def __init__(self,
                 num_peds=1,
                 dt=0.1,
                 alpha=0.85,
                 beta=0.005,
                 v_des=11.17,
                 delta=4.0,
                 t_headway=1.5,
                 a_max=3.0,
                 s_min=4.0,
                 d_cmf=2.0,
                 d_max=9.0,
                 min_dist_x=2.5,
                 min_dist_y=1.4,
                 car_init_x=-35.0,
                 car_init_y=0.0,
                 # blackbox_sim_state = True,
                 **kwargs):
        # Constant hyper-params -- set by user
        self.c_num_peds = num_peds
        self.c_dt = dt
        self.c_alpha = alpha
        self.c_beta = beta
        self.c_v_des = v_des
        self.c_delta = delta
        self.c_t_headway = t_headway
        self.c_a_max = a_max
        self.c_s_min = s_min
        self.c_d_cmf = d_cmf
        self.c_d_max = d_max
        self.c_min_dist = np.array([min_dist_x, min_dist_y])
        self.c_car_init_x = car_init_x
        self.c_car_init_y = car_init_y
        # self.blackbox_sim_state = blackbox_sim_state

        # These are set by reset, not the user
        self._car = np.zeros((4))
        self._car_accel = np.zeros((2))
        self._peds = np.zeros((self.c_num_peds, 4))
        self._measurements = np.zeros((self.c_num_peds, 4))
        self._car_obs = np.zeros((self.c_num_peds, 4))
        self._env_obs = np.zeros((self.c_num_peds, 4))
        self._done = False
        self._reward = 0.0
        self._info = []
        self._step = 0
        self._path_length = 0
        # self._action = None
        self._action = np.array([0] * (6 * self.c_num_peds))
        self._first_step = True
        self.directions = np.random.randint(2, size=self.c_num_peds) * 2 - 1
        self.y = np.random.rand(self.c_num_peds) * 14 - 5
        self.x = np.random.rand(self.c_num_peds) * 4 - 2
        self._state = None

    def run_simulation(self, actions, s_0, simulation_horizon):
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
        # initialize the simulation
        path_length = 0
        self.reset(s_0)
        self._info = []

        simulation_horizon = np.minimum(simulation_horizon, len(actions))

        # Take simulation steps unbtil horizon is reached
        while path_length < simulation_horizon:
            # get the action from the list
            self._action = actions[path_length]

            # Step the simulation forward in time
            self.step_simulation(self._action)

            # check if a crash has occurred. If so return the timestep, otherwise continue
            if self.collision_detected():
                return path_length, np.array(self._info)
            path_length = path_length + 1

        # horizon reached without crash, return -1
        self._is_terminal = True
        return -1, np.array(self._info)

    def step_simulation(self, action):
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
        # return None

        # get the action from the list
        self._action = action

        # move the peds
        self.update_peds()

        # move the car
        self._car = self.move_car(self._car, self._car_accel)

        # take new measurements and noise them
        noise = self._action.reshape((self.c_num_peds, 6))[:, 2:6]
        self._measurements = self.sensors(self._car, self._peds, noise)

        # filter out the noise with an alpha-beta tracker
        self._car_obs = self.tracker(self._car_obs, self._measurements)

        # select the SUT action for the next timestep
        self._car_accel[0] = self.update_car(self._car_obs, self._car[0])

        # grab simulation state, if interactive
        self.observe()
        self.observation = np.ndarray.flatten(self._env_obs)
        # record step variables
        self.log()

        # else:
        #     obs = None

        # self._path_length += 1
        # if self._path_length >= self.c_max_path_length:
        #     self._is_terminal = True

        # if self.blackbox_sim_state:
        #     obs = action

        # pdb.set_trace()
        # print(obs)
        return self.observation

    def reset(self, s_0):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """

        # initialize variables
        self._info = []
        self._step = 0
        self._path_length = 0
        self._is_terminal = False
        self.initial_conditions = s_0
        self._action = np.array([0] * (6 * self.c_num_peds))
        self._first_step = True

        # Get v_des if it is sampled from a range
        v_des = self.initial_conditions[3 * self.c_num_peds]

        # initialize SUT location
        car_init_x = self.initial_conditions[3 * self.c_num_peds + 1]
        self._car = np.array([v_des, 0.0, car_init_x, self.c_car_init_y])

        # zero out the first SUT acceleration
        self._car_accel = np.zeros((2))

        # initialize pedestrian locations and velocities
        pos = self.initial_conditions[0:2 * self.c_num_peds]
        self.x = pos[0:self.c_num_peds * 2:2]
        self.y = pos[1:self.c_num_peds * 2:2]
        v_start = self.initial_conditions[2 * self.c_num_peds:3 * self.c_num_peds]
        self._peds[0:self.c_num_peds, 0] = np.zeros((self.c_num_peds))
        self._peds[0:self.c_num_peds, 1] = v_start
        self._peds[0:self.c_num_peds, 2] = self.x
        self._peds[0:self.c_num_peds, 3] = self.y

        # Calculate the relative position measurements
        self._measurements = self._peds
        self._env_obs = self._measurements
        self._car_obs = self._measurements

        # return the initial simulation state
        self.observation = np.ndarray.flatten(self._measurements)
        # self.observation = obs
        return self.observation

    def collision_detected(self):
        """
        returns whether the current state is in the goal set
        :return: boolean, true if current state is in goal set.
        """
        # calculate the relative distances between the pedestrians and the car
        dist = self._peds[:, 2:4] - self._car[2:4]

        # return True if any relative distance is within the SUT's hitbox and the car is still moving
        if (np.any(np.all(np.less_equal(abs(dist), self.c_min_dist), axis=1)) and
                self._car[0] > 0.5):
            return True

        return False

    def log(self):
        # Create a cache of step specific variables for post-simulation analysis
        cache = np.hstack([0.0,  # Dummy, will be filled in with trial # during post processing in save_trials.py
                           self._step,
                           np.ndarray.flatten(self._car),
                           np.ndarray.flatten(self._peds),
                           np.ndarray.flatten(self._action),
                           np.ndarray.flatten(self._car_obs),
                           0.0])
        self._info.append(cache)
        self._step += 1

    def sensors(self, car, peds, noise):

        measurements = peds + noise
        return measurements

    def tracker(self, observation_old, measurements):
        observation = np.zeros_like(observation_old)

        observation[:, 0:2] = observation_old[:, 0:2]
        observation[:, 2:4] = observation_old[:, 2:4] + self.c_dt * observation_old[:, 0:2]
        residuals = measurements[:, 2:4] - observation[:, 2:4]

        observation[:, 2:4] += self.c_alpha * residuals
        observation[:, 0:2] += self.c_beta / self.c_dt * residuals

        return observation

    def update_car(self, obs, v_car):

        cond = np.repeat(np.resize(np.logical_and(obs[:, 3] > -1.5, obs[:, 3] < 4.5), (self.c_num_peds, 1)), 4, axis=1)
        in_road = np.expand_dims(np.extract(cond, obs), axis=0)

        if in_road.size != 0:
            mins = np.argmin(in_road.reshape((-1, 4)), axis=0)
            v_oth = obs[mins[3], 0]
            s_headway = obs[mins[3], 2] - self._car[2]
            s_headway = max(10 ** -6, abs(s_headway)) * np.sign(s_headway)  # avoid div by zero error later

            del_v = v_oth - v_car
            s_des = self.c_s_min + v_car * self.c_t_headway - v_car * del_v / (2 * np.sqrt(self.c_a_max * self.c_d_cmf))
            if self.c_v_des > 0.0:
                v_ratio = v_car / self.c_v_des
            else:
                v_ratio = 1.0

            a = self.c_a_max * (1.0 - v_ratio ** self.c_delta - (s_des / s_headway) ** 2)

        else:
            del_v = self.c_v_des - v_car
            a = del_v

        if np.isnan(a):
            pdb.set_trace()
        # pdb.set_trace()
        return np.clip(a, -self.c_d_max, self.c_a_max)

    def move_car(self, car, accel):
        car[2:4] += self.c_dt * car[0:2]
        car[0:2] += self.c_dt * accel
        return car

    def update_peds(self):
        # Update ped state from actions
        action = self._action.reshape((self.c_num_peds, 6))[:, 0:2]

        mod_a = np.hstack((action,
                           self._peds[:, 0:2] + 0.5 * self.c_dt * action))
        if np.any(np.isnan(mod_a)):
            pdb.set_trace()

        self._peds += self.c_dt * mod_a
        # Enforce max abs(velocity) on pedestrians
        self._peds[:, 0:2] = np.clip(self._peds[:, 0:2], a_min=[-4.5, -4.5], a_max=[4.5, 4.5])
        if np.any(np.isnan(self._peds)):
            pdb.set_trace()

    def observe(self):
        self._env_obs = self._peds - self._car

    def get_ground_truth(self):
        return {'step': self._step,
                'path_length': self._path_length,
                'is_terminal': self._is_terminal,
                'car': self._car,
                'car_accel': self._car_accel,
                'peds': self._peds,
                'car_obs': self._car_obs,
                'action': self._action,
                'initial_conditions': self.initial_conditions,
                }
        # simulator_state = np.concatenate((np.array([self._step]),
        #                                   np.array([self._path_length]),
        #                                   np.array([int(self._is_terminal)]),
        #                                   self._car,
        #                                   self._car_accel,
        #                                   self._peds.flatten(),
        #                                   self._car_obs.flatten(),
        #                                   self._action.flatten(),
        #                                   self.initial_conditions), axis=0)
        #
        # return simulator_state

    def set_ground_truth(self, in_simulator_state):
        in_simulator_state.copy()

        self._step = in_simulator_state['step']
        self._path_length = in_simulator_state['path_length']
        self._is_terminal = in_simulator_state['is_terminal']
        self._car = in_simulator_state['car']
        self._car_accel = in_simulator_state['car_accel']
        self._peds = in_simulator_state['peds']
        self._car_obs = in_simulator_state['car_obs']
        self._action = in_simulator_state['action']
        self.initial_conditions = in_simulator_state['initial_conditions']

        # self._step = simulator_state[0]
        # self._path_length = simulator_state[1]
        # self._is_terminal = bool(simulator_state[2])
        # self._car = simulator_state[3:7]
        # self._car_accel = simulator_state[7:9]
        # peds_end_index = 9 + self.c_num_peds * 4
        # self._peds = simulator_state[9:peds_end_index].reshape((self.c_num_peds, 4))
        # car_obs_end_index = peds_end_index + self.c_num_peds * 4
        # self._car_obs = simulator_state[peds_end_index:car_obs_end_index].reshape((self.c_num_peds, 4))
        # self._action = simulator_state[car_obs_end_index:car_obs_end_index + self._action.shape[0]]
        # self.initial_conditions = simulator_state[car_obs_end_index + self._action.shape[0]:]
        # self._info = []

    def render(self, car, ped, noise, gif=False):
        if gif:
            return
        else:
            return
