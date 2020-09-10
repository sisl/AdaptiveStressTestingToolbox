"""A toy simulator of a scenario of an AV approaching a crosswalk where some pedestrians are crossing."""
import pdb  # Used for debugging

import numpy as np  # Used for math

# Define the class


class ToyAVSimulator():
    """A toy simulator of a scenario of an AV approaching a crosswalk where some pedestrians are crossing.

    The vehicle runs a modified version of the Intelligent Driver Model [1]_. The vehicle treats the closest
    pedestrian in the road as a car to follow. If no pedestrians are in the road, it attempts to maintain the
    desired speed. Noisy observations of the pedestrian are smoothed through an alpha-beta filter [2]_.

    A collision results if any pedestrian's x-distance and y-distance to the ego vehicle are less than the respective
    `min_dist_x` and `min_dist_y`.

    The origin is centered in the middle of the east/west lane and the north/south crosswalk.
    The positive x proceeds east down the lane, the positive y proceeds north across the crosswalk.

    Parameters
    ----------
    num_peds : int
        The number of pedestrians crossing the street.
    dt : float
        The length (in seconds) of each timestep.
    alpha : float
        The alpha parameter in the tracker's alpha-beta filter [2]_.
    beta : float
        The beta parameter in the tracker's alpha-beta filter [2]_.
    v_des : float
        The desired velocity, in meters per second,  for the ego vehicle to maintain
    delta : float
        The delta parameter in the IDM algorithm [1]_.
    t_headway : float
        The headway parameter in the IDM algorithm [1]_.
    a_max : float
        The maximum acceleration parameter in the IDM algorithm [1]_.
    s_min : float
        The minimum follow distance parameter in the IDM algorithm [1]_.
    d_cmf : float
        The maximum comfortable deceleration parameter in the IDM algorithm [1]_.
    d_max : float
        The maximum deceleration parameter in the IDM algorithm [1]_.
    min_dist_x : float
        The minimum x-distance between the ego vehicle and a pedestrian.
    min_dist_y : float
        The minimum y-distance between the ego vehicle and a pedestrian.
    car_init_x : float
        The initial x-position of the ego vehicle.
    car_init_y : float
        The initial y-position of the ego vehicle.

    References
    ----------
    .. [1] Treiber, Martin, Ansgar Hennecke, and Dirk Helbing.
        "Congested traffic states in empirical observations and microscopic simulations."
        Physical review E 62.2 (2000): 1805.
        `<https://journals.aps.org/pre/abstract/10.1103/PhysRevE.62.1805>`_
    .. [2] Rogers, Steven R. "Alpha-beta filter with correlated measurement noise."
        IEEE Transactions on Aerospace and Electronic Systems 4 (1987): 592-594.
        `<https://ieeexplore.ieee.org/abstract/document/4104388>`_
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
                 ):

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
        """Run a full simulation given the AST solver's actions and initial conditions.

        Parameters
        ----------
        actions : list[array_like]
            A sequential list of actions taken by the AST Solver which deterministically control the simulation.
        s_0 : array_like
            An array specifying the initial conditions to set the simulator to.
        simulation_horizon : int
            The maximum number of steps a simulation rollout is allowed to run.

        Returns
        -------
        terminal_index : int
            The index of the action that resulted in a state in the goal set E. If no state is found
            terminal_index should be returned as -1.
        array_like
            An array of relevant simulator info, which can then be used for analysis or diagnostics.

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
        Handle anything that needs to take place at each step, such as a simulation update or write to file.

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
        # return None

        # get the action from the list
        self._action = action

        # move the peds
        self.update_peds()

        # move the car
        self._car = self.move_car(self._car, self._car_accel)

        # take new measurements and noise them
        noise = self._action.reshape((self.c_num_peds, 6))[:, 2:6]
        self._measurements = self.sensors(self._peds, noise)

        # filter out the noise with an alpha-beta tracker
        self._car_obs = self.tracker(self._car_obs, self._measurements)

        # select the SUT action for the next timestep
        self._car_accel[0] = self.update_car(self._car_obs, self._car[0])

        # grab simulation state, if interactive
        self.observe()
        self.observation = np.ndarray.flatten(self._env_obs)
        # record step variables
        self.log()

        return self.observation

    def reset(self, s_0):
        """Resets the state of the environment, returning an initial observation.

        Parameters
        ----------
        s_0 : array_like
            The initial conditions to reset the simulator to.

        Returns
        -------
        array_like
            An observation from the timestep, determined by the settings and the `observation_return` helper function.
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
        Returns whether the current state is in the goal set.

        Checks to see if any pedestrian's position violates both the `min_dist_x` and `min_dist_y` constraints.

        Returns
        -------
        bool
            True if current state is in goal set.
        """
        # calculate the relative distances between the pedestrians and the car
        dist = self._peds[:, 2:4] - self._car[2:4]

        # return True if any relative distance is within the SUT's hitbox and the car is still moving
        if (np.any(np.all(np.less_equal(abs(dist), self.c_min_dist), axis=1)) and
                self._car[0] > 0.5):
            return True

        return False

    def log(self):
        """
        Perform any logging steps.

        """
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

    def sensors(self, peds, noise):
        """Get a noisy observation of the pedestrians' locations and velocities.

        Parameters
        ----------
        peds : array_like
            Positions and velocities of the pedestrians.
        noise : array_like
            Noise to add to the positions and velocities of the pedestrians.

        Returns
        -------
        array_like
            Noisy observation of the pedestrians' locations and velocities.

        """

        measurements = peds + noise
        return measurements

    def tracker(self, estimate_old, measurements):
        """An alpha-beta filter to smooth noisy observations into an estimate of pedestrian state.

        Parameters
        ----------
        estimate_old : array_like
            The smoothed state estimate from the previous timestep.
        measurements : array_like
            The noisy observation of pedestrian state from the current timestep.

        Returns
        -------
        array_like
            The smoothed state estimate of pedestrian state from the current timestep.
        """
        observation = np.zeros_like(estimate_old)

        observation[:, 0:2] = estimate_old[:, 0:2]
        observation[:, 2:4] = estimate_old[:, 2:4] + self.c_dt * estimate_old[:, 0:2]
        residuals = measurements[:, 2:4] - observation[:, 2:4]

        observation[:, 2:4] += self.c_alpha * residuals
        observation[:, 0:2] += self.c_beta / self.c_dt * residuals

        return observation

    def update_car(self, obs, v_car):
        """Calculate the ego vehicle's acceleration.

        Parameters
        ----------
        obs : array_like
            Smoothed estimate of pedestrian state from the `tracker`.
        v_car : float
            Current velocity of the ego vehicle.

        Returns
        -------
        float
            The acceleration of the ego vehicle.

        """
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
        """Update the ego vehicle's state.

        Parameters
        ----------
        car : array_like
            The ego vehicle's state: [x-velocity, y-velocity, x-position, y-position].
        accel : float
            The ago vehicle's acceleration.

        Returns
        -------
        array_like
            An updated version of the ego vehicle's state.

        """
        car[2:4] += self.c_dt * car[0:2]
        car[0:2] += self.c_dt * accel
        return car

    def update_peds(self):
        """Update the pedestrian's state.

        """
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
        """Get the ground truth state of the pedestrian relative to the ego vehicle.

        """
        self._env_obs = self._peds - self._car

    def get_ground_truth(self):
        """Clones the ground truth simulator state.

        Returns
        -------
        dict
            A dictionary of simulator state variables.
        """
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

    def set_ground_truth(self, in_simulator_state):
        """Sets the simulator state variables.

        Parameters
        ----------
        in_simulator_state : dict
            A dictionary of simulator state variables.
        """
        in_simulator_state.copy()

        self._step = in_simulator_state['step']
        self._path_length = in_simulator_state['path_length']
        self._is_terminal = in_simulator_state['is_terminal']
        self._car = in_simulator_state['car']
        self._car_accel = in_simulator_state['car_accel']
        self._peds = in_simulator_state['peds']
        self._car_obs = in_simulator_state['car_obs']
        self._action = in_simulator_state['action']
        self.initial_conditions = np.array(in_simulator_state['initial_conditions'])

        self.observe()
        self.observation = self._env_obs
