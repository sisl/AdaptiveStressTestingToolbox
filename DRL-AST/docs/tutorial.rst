Tutorial
******************
.. _introduction:

1 Introduction
===============

This tutorial is intended for readers to learn how to use this package with their own simulator.
Matery of the underlying theory would be helpful, but is not needed for installation. Please install 
package before proceeding.

.. _about-ast:

1.1 About AST
-----------------
Adaptive Stress Testing is a way of finding flaws in an autonomous agent. For any non-trivial problem, 
searching the space of a stochastic simulation is intractable, and grid searches do not perform well.
By modeling the search as a Markov Decision Process, we can use reinforcement learning to find the
most probable failure. AST treats the simulator as a black box, and only needs access in a few specific
ways. To interface a simulator to the AST packages, a few things will be needed:

* A **Simulator** is a wraper that exposes the simulation software to this package. See the Simulator section for details on Interactive vs. Non-Interactive Simulators
* A **Reward** function dictates the optimization goals of the algorithm. 
* A **Runner** collects all of the run options and starts the method.
* **Space** objects give information on the size and limits of a space. This will be used to
define the **Observation Space** and the **Action Space**

.. _about-this-tutorial:

1.2 About this tutorial
------------------------

In this tutorial, we will create a simple ring road network, which in the
absence of autonomous vehicles, experience a phenomena known as "stop-and-go
waves". An autonomous vehicle in then included into the network and trained
to attenuate these waves. The remainder of the tutorial is organized as follows:

-  In Sections 2, 3, and 4, we create the primary classes needed to run
   a ring road experiment.
-  In Section 5, we run an experiment in the absence of autonomous
   vehicles, and witness the performance of the network.
-  In Section 6, we run the experiment with the inclusion of autonomous
   vehicles, and discuss the changes that take place once the
   reinforcement learning algorithm has converged.


.. _creating-a-simulator:

2 Creating a Simulator
======================

This sections explains how to create a wrapper that exposes your simulator to the AST package. The 
wrapper allows the AST solver to specify actions to control the stochasticity in the simulation. 
Examples of stochastic simulation elements would be an actor, like a pedestrian or a car, or noise
elements, like on the beams of a LIDAR sensor. The simulator must be able to reset on command, and 
detect if a goal state had been reached. The simulator state can be used, but is not neccessary. 
Interactive simulations are optional as well.

.. _interactive-simulations:

2.1 Interactive Simulations
---------------------------

An Interactive Simulation is one in which control can be injected at each step during the actual simulation run. 
For example, if a simulation is run by creating a specification file, and no other control is possible, that 
simulation would not be interactive. A simulation must be interactive for simulation state to be accesable 
to the AST solver. Passing the simulation state to the solver may reduce the number of episodes needed to
converge to a solution. However, pausing the simulation at each step may introduce overhead which slows
the execution. Neither variant is inherently better, so use whatever is appropriate for your project.

.. _inheriting-the-base-simulator:

2.2 Inheriting the Base Simulator
---------------------------------

Start by creating a file named ``example_av_simulator.py`` in the ``simulators`` folder. Create a class titled
``ExampleAVSimulator``, which inherits from ``Simulator``.

::

	#import base Simulator class
	from mylab.simulators.simulator import Simulator

	#Used for math and debugging
	import numpy as np
	import pdb

	#Define the class
	class ExampleAVSimulator(Simulator):

The base generator accepts one input:

* **max_path_length**: The horizon of the simulation, in number of timesteps

A child of the Simulator class is required to define the following five functions: ``simulate``, ``step``, ``reset``, ``get_reward_info``, and ``is_goal``. An optional ``log`` function may also be implemented. 

.. _initializing-the-example-simulator:

2.3 Initializing the Example Simulator
--------------------------------------

Our example simulator will control a modified version of the Intelligent Driver Model (IDM) as our SUT, while adding sensor noise and filtering it out with an alpha-beta tracker. Initial simulation conditions are needed here as well. Because of all thise, the Simulator accepts a number of inputs:

* **num\_peds**: The number of pedestrians in the scenario
* **dt**: The length of the time step, in seconds
* **alpha**: A hyperparameter controlling the alpha-beta tracker that filters noise from the sensors
* **beta**: A hyperparameter controlling the alpha-beta tracker that filters noise from the sensors
* **v\_des**: The desired speed of the SUT
* **t\_headway**: An IDM hyperparameter that controls the target seperation between the SUT and the agent it is following, measured in seconds
* **a\_max**: An IDM hyperparameter that controls the maximum acceleration of the SUT
* **s\_min**: An IDM hyperparameter that controls the minimum distance between the SUT and the agent it is following
* **d\_cmf**: An IDM hyperparameter that controls the maximum comfortable decceleration of the SUT (a soft maximum that is only violated to avoid crashes)
* **d\_max**: An IDM hyperparameter that controls the maximum decceleration of the SUT
* **min\_dist\_x**: Defines the length of the hitbox in the x direction
* **min\_dist\_y**: Defines the length of the hitbox in the y direction
* **car\_init\_x**: Specifies the initial x-position of the SUT
* **car\_init\_y**: Specifies the initial y-position of the SUT
* **action\_only**: A boolean value specifying whether the simulation state is unobserved, so only the previous action will be used as input to the policy. Only set to False if you have an interactive simulatior with an observable state, and you would like to pass that state as part of the input to the policy (see `section 2.1`_)
* **kwargs**: Any keyword arguement not listed here. In particular, ``max_path_length`` should be pased to the base Simulator as one of the **kwargs.

.. _section 2.1: interactive-simulations_

In addition, there are a number of member variables that need to be initialized. The code is below:
::
    def __init__(self,
                 ego = None,
                 num_peds = 1,
                 dt = 0.1,
                 alpha = 0.85,
                 beta = 0.005,
                 v_des = 11.17,
                 delta = 4.0,
                 t_headway = 1.5,
                 a_max = 3.0,
                 s_min = 4.0,
                 d_cmf = 2.0,
                 d_max = 9.0,
                 min_dist_x = 2.5,
                 min_dist_y = 1.4,
                 car_init_x = 35.0,
                 car_init_y = 0.0,
                 action_only = True,
                 **kwargs):
        #Constant hyper-params -- set by user
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
        self.action_only = action_only

        #These are set by reset, not the user
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
        self._action = None
        self._first_step = True
        self.directions = np.random.randint(2, size=self.c_num_peds) * 2 - 1
        self.y = np.random.rand(self.c_num_peds) * 14 - 5
        self.x = np.random.rand(self.c_num_peds) * 4 - 2
        self.low_start_bounds = [-1.0, -4.25, -1.0, 5.0, 0.0, -6.0, 0.0, 5.0]
        self.high_start_bounds = [0.0, -3.75, 0.0, 9.0, 1.0, -2.0, 1.0, 9.0]
        self.v_start = [1.0, -1.0, 1.0, -1.0]
        self._state = None

        #initialize the base Simulator
        super().__init__(**kwargs)

.. _the-simulate-function:

2.4 The ``simulate`` function:
------------------------------

The simulate function runs a simulation using previously generated actions from the policy to control the stochasticity. The simulate function accepts a list of actions and an intitial state. It should run the simulation, then return the timestep that the goal state was achieved, or a -1 if the horizon was reached first. In addition, this function should return any simulation info needed for post-analysis. To do this, first add the following code to the file to handle the simulation aspect:
:: 
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
        if np.any(np.isnan(self._peds)):
            pdb.set_trace()

    def observe(self):
        self._env_obs = self._peds - self._car

These functions handle the backend simulation of the toy problem and the SUT. Now we implement the ``simulate`` function, checking to be sure that the horizon wasn't reached:
::
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
        # initialize the simulation
        path_length = 0
        self.reset(s_0)
        self._info  = []

        # Take simulation steps unbtil horizon is reached
        while path_length < self.c_max_path_length:
            #get the action from the list
            self._action = actions[path_length]

            # move the peds
            self.update_peds()

            # move the car
            self._car = self.move_car(self._car, self._car_accel)

            # take new measurements and noise them
            noise = self._action.reshape((self.c_num_peds,6))[:, 2:6]
            self._measurements = self.sensors(self._car, self._peds, noise)

            # filter out the noise with an alpha-beta tracker
            self._car_obs = self.tracker(self._car_obs, self._measurements)

            # select the SUT action for the next timestep
            self._car_accel[0] = self.update_car(self._car_obs, self._car[0])

            # grab simulation state, if interactive
            self.observe()

            # record step variables
            self.log()

            # check if a crash has occurred. If so return the timestep, otherwise continue
            if self.is_goal():
                return path_length, np.array(self._info)
            path_length = path_length + 1

        # horizon reached without crash, return -1
        self._is_terminal = True
        return -1, np.array(self._info)

.. _the-step-function:

2.5 The ``step`` function:
--------------------------

If a simulation is interactive, the ``step`` function should interact with it at each timestep. The functions takes as input the current action. If the action is interactive and the simulation state is being used, return the state. Otherwise, return ``None``. If the simulation is non-interactive, other per-step actions can still be put here if neccessary - this function is called at each timestep either way. However, there is nothing to do at each step in this case, so the function will just return ``None``.
::
    def step(self, action):
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
        return None

.. _the-reset-function:

2.6 The ``reset`` function:
---------------------------

The reset function should return the simulation to a state where it can accept the next sequence of actions. In some cases this may mean explcitily reseting the simulation parameters, like SUT location or simulation time. It could also mean opening and initializing a new instance of the simulator (in which case the ``simulate`` function should close the current instance). Your implementation of the ``reset`` function may be something else entirely, this is highly dependent on how your simulator functions. The method takes the initial state as an input, and returns the state of the simulator after the reset actions are taken. If the simulation state is not accessable, just return the initial condition parameters that were passed in.
::
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
        self._is_terminal = False
        self.init_conditions = s_0
        self._first_step = True

        # Get v_des if it is sampled from a range
        v_des = self.init_conditions[3*self.c_num_peds]

        # initialize SUT location
        car_init_x = self.init_conditions[3*self.c_num_peds + 1]
        self._car = np.array([v_des, 0.0, car_init_x, self.c_car_init_y])

        # zero out the first SUT acceleration
        self._car_accel = np.zeros((2))

        # initialize pedestrian locations and velocities
        pos = self.init_conditions[0:2*self.c_num_peds]
        self.x = pos[0:self.c_num_peds*2:2]
        self.y = pos[1:self.c_num_peds*2:2]
        v_start = self.init_conditions[2*self.c_num_peds:3*self.c_num_peds]
        self._peds[0:self.c_num_peds, 0] = np.zeros((self.c_num_peds))
        self._peds[0:self.c_num_peds, 1] = v_start
        self._peds[0:self.c_num_peds, 2] = self.x
        self._peds[0:self.c_num_peds, 3] = self.y

        # Calculate the relative position measurements
        self._measurements = self._peds - self._car
        self._env_obs = self._measurements
        self._car_obs = self._measurements

        # return the initial simulation state
        if self.action_only:
            return self.init_conditions
        else:
            self._car = np.array([self.c_v_des, 0.0, self.c_car_init_x, self.c_car_init_y])
            self._car_accel = np.zeros((2))
            self._peds[:, 0:4] = np.array([0.0, 1.0, -0.5, -4.0])
            self._measurements = self._peds - self._car
            self._env_obs = self._measurements
            self._car_obs = self._measurements
            return np.ndarray.flatten(self._measurements)

.. _the-get-reward-info-function:

2.7 The ``get_reward_info`` function:
-------------------------------------

It is likely that your reward function (see XXX) will need some information from the simulator. The reward function will be passed whatever information is returned from this function. For the example, the reward function augments the "no crash" case with the distance between the SUT and the nearest pedestrian. To do this, both the car and pedestrian locations are returned. In addition, boolean values indicating whether a crash has been found or if the horizon has been reached are returned.
::
    def get_reward_info(self):
        """
        returns any info needed by the reward function to calculate the current reward
        """

        return {"peds": self._peds,
                "car": self._car,
                "is_goal": self.is_goal(),
                "is_terminal": self._is_terminal}

.. _the-is-goal-function:

2.8 The ``is_goal`` function:
-----------------------------

This function returns a boolean value indicating if the current state is in the goal set. In the example, this is True if the pedestrian is hit by the car. Therefore this function checks for any pedestrians in the hitbox of the SUT.
::
    def is_goal(self):
        """
        returns whether the current state is in the goal set
        :return: boolean, true if current state is in goal set.
        """
        # calculate the relative distances between the pedestrians and the car
        dist = self._peds[:, 2:4] - self._car[2:4]

        # return True if any relative distance is within the SUT's hitbox
        if (np.any(np.all(np.less_equal(abs(dist), self.c_min_dist), axis=1))):
            return True

        return False

.. _the-log-function-optional:

2.9 The ``log`` function (Optional):
------------------------------------

The log function is a way to store variables from the simulator for later access. In the example, some simulation state information is appended to a list at every timestep.
::
    def log(self):
        # Create a cache of step specific variables for post-simulation analysis
        cache = np.hstack([0.0,  # Dummy, will be filled in with trial # during post processing in save_trials.py
                           self._step,
                           np.ndarray.flatten(self._car),
                           np.ndarray.flatten(self._peds),
                           np.ndarray.flatten(self._action),
                           0.0])
        self._info.append(cache)
        self._step += 1

.. rst-class:: html-toggle

.. _creating-a-reward-function:

3 Creating a Reward Function
============================

This section explains how to create a function that dictates the reward at each timestep of a simulation. AST formulates the problem of searching the space of possible variations of a stochastic simulation as an MDP so that modern-day reinforcement learning (RL) techniques can be used. When optimizing a policy using RL, the reward function is of the utmost importance, as it determines how the agent will learn. Changing the reward function to achieve the desired policy is known as reward shaping. 

.. _reward-shaping:

3.1 Reward Shaping
------------------


**SPOILER ALERT**: This section uses a famous summercamp game as an example. If you are planning on attending a children's summercamp in the near future I highly reccomend you skip this section, lest you ruin the counselors attempts at having fun at your expense. You have been warned.

As an example of reinforcement learning, and the importance of the reward function, consider the famous childrens game "The Hat Game." Common at summer camps, the game usually starts with a counselor holding a hat in his hands, telling the kids he is about to teach them a new game. He will say "Ok, ready everyone....? I can play the hat game," proceed to do a bunch of random things with the hat, and then say "how about you?" He will then pass the hat to a camper, who repeats almost exactly everything the counselor does, but is told "no, you didn't play the hat game." Another counselor will take the hat, say the words, do something completly different with it, and the game is on. The trick is actually the word "OK" - so long as you say that magic word, you have played the hat game, even if you have no hat.

How does this relate to reward shaping? In this case, the children are the policy. They are taking stochastic actions, trying to learn how to play the hat game. The key to the game being fun is that the children are pretrained to pay attentian to meaningless words, and to mimic the hat motions. However, after enough trials (and it can take a long time), most of them will pick up the pattern and attention will shift to "OK." In the vanilla game, there are two rewards. "Yes, you played the hat game" can be considered positive, and "No, you didn't play the hat game" can be considered negative, or just zero. By changing this reward, we could make the game difficulty radically different. Imagine if 10 kids tried the game, and all they got was a binary response on if at least one of them played the game. This would be much harder to pick up on! This is an example of a sparse reward function, or one that only rarely gives rewards, such as at the end of a trajectory. On the other hand, what if the children recieved feedback after every single word or motion on if they had played the hat game during that trial yet. The game would be much easier! These are examples of how different reward functions can make achieving the same policy easier or harder. 

.. _inheriting-the-base-reward-function:

3.2 Inheriting the Base Reward Function
---------------------------------------

Start by creating a file named ``example_av_reward.py`` in the ``rewards`` folder. Create a class title ``ExampleAVReward`` which inherits from ``ASTReward``:
::
	# import base class
	from mylab.rewards.ast_reward import ASTReward

	# useful packages for math and debugging
	import numpy as np
	import pdb

	# Define the class, inherit from the base
	class ExampleAVReward(ASTReward):

The base class does not take an inputs, and there is only one required function - ``give_reward``.

.. _initializing-the-example-reward-function:

3.3 Initializing the Example Reward Function
--------------------------------------------

The reward function will be calculating some rewards based on the probability of certain actions. We have assumed the means action is the 0 vector, but we still need to take the following inputs:

* **num\_peds**: The number of pedestrians in the scenario
* **cov\_x**: The covariance of the gaussian distribution used to model the x-acceleration of a pedestrian
* **cov\_y**: The covariance of the gaussian distribution used to model the y-acceleration of a pedestrian
* **cov\_sensor\_noise**: The covariance of the gaussian distribution used to model the noise on a sensor measurement in both the x and y directions (assumed equal)

The code is below:
::
    def __init__(self,
                 num_peds=1,
                 cov_x=0.1,
                 cov_y=0.01,
                 cov_sensor_noise=0.1):

        self.c_num_peds = num_peds
        self.c_cov_x = cov_x
        self.c_cov_y = cov_y
        self.c_cov_sensor_noise = cov_sensor_noise
        super().__init__()

.. _the-give-reward-function:

3.4 The ``give_reward`` function
--------------------------------

Our example reward function is broken down into three cases, as specified in the paper. The three cases are as follows:

1. There is a crash at the current timestep
2. The horizon of the simulation is reached, with no crash
3. The current step did not find a crash or reach the horizon

The respective reward for each case is as follows:

1. R = 0
2. R = -1E5 - 1E4 * {The distance between the car and the closest pedestrian}
3. R = -log(1 + {likelihood of the actions take})

For case 2, we use the distance between the car and the closest pedestrian as a heurisitc to increase convergence speed. In the early trials, this teaches pedestrians to end closer to the car, which makes it easier to find crash trajectories (see `section 3.1`_). For case 3, using the negative log-likelihood allows us to sum the rewards to find a value that is proportional to the probability of the trajectory. As a stand in for the probability of an action, we use the Mahalanobis distance, a multi-dimensional generalization of distance from the mean. Add the following helper function to your file:
::
    def mahalanobis_d(self, action):
        # Mean action is 0
        mean = np.zeros((6 * self.c_num_peds, 1))
        # Assemble the diagonal covariance matrix
        cov = np.zeros((self.c_num_peds, 6))
        cov[:, 0:6] = np.array([self.c_cov_x, self.c_cov_y,
                                self.c_cov_sensor_noise, self.c_cov_sensor_noise,
                                self.c_cov_sensor_noise, self.c_cov_sensor_noise])
        big_cov = np.diagflat(cov)

        # subtract the mean from our actions
        dif = np.copy(action)
        dif[::2] -= mean[0, 0]
        dif[1::2] -= mean[1, 0]
        
        # calculate the Mahalanobis distance
        dist = np.dot(np.dot(dif.T, np.linalg.inv(big_cov)), dif)

        return np.sqrt(dist)

Now we are ready to calculate the reward. The ``give_reward`` function takes in an action, as well as the info bundle that was returned from the ``get_reward_info`` function in the ``ExampleAVSimulator`` (see `section 2.7`_). The code is as follows:
::
    def give_reward(self, action, **kwargs):
        # get the info from the simulator
        info = kwargs['info']
        peds = info["peds"]
        car = info["car"]
        is_goal = info["is_goal"]
        is_terminal = info["is_terminal"]
        dist = peds[:, 2:4] - car[2:4]

        # update reward and done bool

        if (is_goal): # We found a crash
            reward = 0
        elif (is_terminal):
            reward = -10000 - 1000 * np.min(np.linalg.norm(dist, axis=1)) # We reached
            # the horizon with no crash
        else:
            reward = -np.log(1 + self.mahalanobis_d(action)) # No crash or horizon yet

        return reward

.. _section 3.1: reward-shaping_

.. _section 2.7: the-get-reward-info-function_

.. _creating-the-spaces:

4 Creating the Spaces
=====================

This section shows how to create the action space and observation space for rllab to use. The spaces define the limits of what is possible for inputs to and outputs from the policy. The observation space can be used as input if the simulation state is accesible, and can be used to generate intial conditions if they are being sampled from a range. The action space is the output, and controls the size of the output array from the policy. 

.. _inheriting-the-base-spaces:

4.1 Inheriting the Base Spaces
------------------------------

Create a file named ``example_av_spaces.py`` in the ``spaces`` folder. Create a class titled ``ExampleAVSpaces`` which inherits from ``ASTSpaces``:
::
	from mylab.spaces.ast_spaces import ASTSpaces
	from rllab.spaces import Box
	import numpy as np

	class ExampleAVSpaces(ASTSpaces):

The base spaces don't take any input, but there are two functions to define: ``action_space`` and ``observation_space``. Both of these functions should return an object that inherits from the ''Space'' class, imported from ``rllab.spaces.base``. There are a few options, and you can implement your own, but the ``Box`` class is used here. A ``Box`` is defined by two arrays, ``low`` and ``high``, of equal length, which specifiy the minium and maximum value of each position in the array. The space is then allows any continuos number between the low and high values.

.. _initializing-the-spaces:

4.2 Initializing the Spaces
---------------------------

In order to define our spaces, there are a number of inputs:

* **num\_peds**: The number of pedestrians in the scenario
* **max\_path\_length**: The horizon of the trajectory rollout, in number of timesteps
* **v_des**: The desired velocity of the SUT
* **x\_accel\_low**: The minimum acceleration in the x-direction of the pedestrian
* **y\_accel\_low**: The minimum acceleration in the y-direction of the pedestrian
* **x\_accel\_high**: The maximum acceleration in the x-direction of the pedestrian
* **y\_accel\_high**: The maximum acceleration in the y-direction of the pedestrian
* **x\_boundary\_low**: The minimum x-position of the pedestrian
* **y\_boundary\_low**: The minimum y-position of the pedestrian
* **x\_boundary\_high**: The maximum x-position of the pedestrian
* **y\_boundary\_high**: The maximum y-position of the pedestrian
* **x\_v\_low**:: The minimum initial x-velocity of the pedestrian
* **y\_v\_low**:: The minimum initial y-velocity of the pedestrian
* **x\_v\_high**:: The maximum initial x-velocity of the pedestrian
* **y\_v\_high**:: The maximum initial y-velocity of the pedestrian
* **car\_init\_x**: The initial x-position of the SUT
* **car\_init\_y**: The initial y-position of the SUT

The initialization code is below:
::
    def __init__(self,
                 num_peds=1,
                 max_path_length = 50,
                 v_des=11.17,
                 x_accel_low=-1.0,
                 y_accel_low=-1.0,
                 x_accel_high=1.0,
                 y_accel_high=1.0,
                 x_boundary_low=-10.0,
                 y_boundary_low=-10.0,
                 x_boundary_high=10.0,
                 y_boundary_high=10.0,
                 x_v_low=-10.0,
                 y_v_low=-10.0,
                 x_v_high=10.0,
                 y_v_high=10.0,
                 car_init_x=35.0,
                 car_init_y=0.0,):

        # Constant hyper-params -- set by user
        self.c_num_peds = num_peds
        self.c_max_path_length = max_path_length
        self.c_v_des = v_des
        self.c_x_accel_low = x_accel_low
        self.c_y_accel_low = y_accel_low
        self.c_x_accel_high = x_accel_high
        self.c_y_accel_high = y_accel_high
        self.c_x_boundary_low = x_boundary_low
        self.c_y_boundary_low = y_boundary_low
        self.c_x_boundary_high = x_boundary_high
        self.c_y_boundary_high = y_boundary_high
        self.c_x_v_low = x_v_low
        self.c_y_v_low = y_v_low
        self.c_x_v_high = x_v_high
        self.c_y_v_high = y_v_high
        self.c_car_init_x = car_init_x
        self.c_car_init_y = car_init_y

        super().__init__()

.. _the-action-space:

4.3 The Action Space
--------------------

The ``action_space`` function takes no inputs and returns a child of the ``Space`` class. The length of the action space array determines the output dimension of the policy. Note the ``@Property`` decorator in the code below:
::
    @property
    def action_space(self):
        """
        Returns a Space object
        """
        low = np.array([self.c_x_accel_low, self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])
        high = np.array([self.c_x_accel_high, self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])

        for i in range(1, self.c_num_peds):
            low = np.hstack((low, np.array([self.c_x_accel_low, self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])))
            high = np.hstack((high, np.array([self.c_x_accel_high, self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])))

        return Box(low=low, high=high)

.. _the-observation-space:

4.4 The Observation Space
-------------------------

The ``observation_space`` function takes no inputs and returns a child of the ``Space`` class. If the simulation state is accesible, the ranges of possible values should be defined using this function, which determines the expected input shape to the policy. If initial conditions are sampled, the will be sampled from the observation space. Therefore, the observation space should define the maximum and minimum value of every simulation state that will be passed as input to the policy, as well as a value for every initial condition needed to specify a scenario variation. Note the ``@Property`` decorator in the code below:
::
    @property
    def observation_space(self):
        """
        Returns a Space object
        """

        low = np.array([self.c_x_v_low, self.c_y_v_low, self.c_x_boundary_low, self.c_y_boundary_low])
        high = np.array([self.c_x_v_high, self.c_y_v_high, self.c_x_boundary_high, self.c_y_boundary_high])

        for i in range(1, self.c_num_peds):
            low = np.hstack(
                (low, np.array([self.c_x_v_low, self.c_y_v_low, self.c_x_boundary_low, self.c_y_boundary_low])))
            high = np.hstack(
                (high, np.array([self.c_x_v_high, self.c_y_v_high, self.c_x_boundary_high, self.c_y_boundary_high])))

        if self.action_only:
            low = self.low_start_bounds[:self.c_num_peds * 2]
            low = low + np.ndarray.tolist(0.0 * np.array(self.v_start))[:self.c_num_peds]
            low = low + [0.75 * self.c_v_des]
            low = low + [0.75 * self.c_car_init_x]
            high = self.high_start_bounds[:self.c_num_peds * 2]
            high = high + np.ndarray.tolist(2.0 * np.array(self.v_start))[:self.c_num_peds]
            high = high + [1.25 * self.c_v_des]
            high = high + [1.25 * self.c_car_init_x]

        # pdb.set_trace()
        return Box(low=np.array(low), high=np.array(high))

.. _creating-a-runner:

5 Creating a Runner
===================

This section explains how to create a file to run the experiment we have been creating. This will use all of the example files we have created, and interface them with the a package for handling RL. The backend framework handling the policy definition and optimization is a package called RLLAB. The project is open-source, so if you would like to understand more about what RLLAB is doing please see the documentation here. 

.. _setting-up-the-runners:

5.1 Setting Up the Runners
--------------------------

Create a file called ``example_runner.py`` in your working directory. Add the following code to handle all of the necessary imports:
::
	# Import the example classes
	from mylab.simulators.example_av_simulator import ExampleAVSimulator
	from mylab.rewards.example_av_reward import ExampleAVReward
	from mylab.spaces.example_av_spaces import ExampleAVSpaces

	# Import the AST classes
	from mylab.envs.ast_env import ASTEnv
	from mylab.ast_vectorized_sampler import ASTVectorizedSampler

	# Import the necessary RLLAB classes
	from sandbox.rocky.tf.algos.trpo import TRPO
	from sandbox.rocky.tf.envs.base import TfEnv
	from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
	from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
	from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
	from rllab.envs.normalized_env import normalize
	import rllab.misc.logger as logger

	# Useful imports
	import os.path as osp
	import argparse
	from save_trials import *
	import tensorflow as tf

.. _creating-a-logger:

5.2 Creating a Logger
---------------------

It is useful to get some feedback on how the policy training is going. To do that, an rllab ``logger`` is needed. To handle the parameters needed to specifiy the logger, an ``ArgumentParser`` is used, from the ``argparse`` package. This package allows command line arguments to be passed when executing a file, allowing easier automation of experiments. The ``argparse`` flags specified are listed here:

* **--exp\_name**: Name of the experiment
* **--tabular\_log\_file**: Name of the log file used to dump the tabular experiment logs
* **--text\_log\_file**: Name of the log file used to dump the text based experiment logs
* **--params\_log\_file**: Name of the log file used to write out the input parameters
* **--snapshot\_mode**: How the snapshot recording frequency will be specified
* **--snapshot_gap**: How many episodes to skip between writing out an episode snapshot
* **--log_tabular_only**: A boolean specifiying if only the tabular experiment logs should be written
* **--log-dir**: What directory the logger should write output to

The code for defning these flags, as well as using them to create the logger, is below:
::
	# Logger Params
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, default='crosswalk_exp')
	parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
	parser.add_argument('--text_log_file', type=str, default='tex.txt')
	parser.add_argument('--params_log_file', type=str, default='args.txt')
	parser.add_argument('--snapshot_mode', type=str, default="gap")
	parser.add_argument('--snapshot_gap', type=int, default=10)
	parser.add_argument('--log_tabular_only', type=bool, default=False)
	parser.add_argument('--log_dir', type=str, default='.')
	args = parser.parse_args()

	# Create the logger
	log_dir = args.log_dir

	tabular_log_file = osp.join(log_dir, args.tabular_log_file)
	text_log_file = osp.join(log_dir, args.text_log_file)
	params_log_file = osp.join(log_dir, args.params_log_file)

	logger.log_parameters_lite(params_log_file, args)
	logger.add_text_output(text_log_file)
	logger.add_tabular_output(tabular_log_file)
	prev_snapshot_dir = logger.get_snapshot_dir()
	prev_mode = logger.get_snapshot_mode()
	logger.set_snapshot_dir(log_dir)
	logger.set_snapshot_mode(args.snapshot_mode)
	logger.set_snapshot_gap(args.snapshot_gap)
	logger.set_log_tabular_only(args.log_tabular_only)
	logger.push_prefix("[%s] " % args.exp_name)

.. _specifying-the-experiment:

5.3 Specifying the Experiment
-----------------------------

All of the classes imported earlier will now be used to specify the experiment. The example classes were defined such that every keyword arguement had a default value. These can be changed by passing in a different value, but were left undefined here. The rllab components also have keyword arguements, many of which are specified here. These can be changed as well, but the rllab documentation should be consulted first. Add the following code to your runner file:
::
	# Instantiate the example classes
	sim = ExampleAVSimulator()
	reward_function = ExampleAVReward()
	spaces = ExampleAVSpaces()

	# Create the environment
	env = TfEnv(normalize(ASTEnv(action_only=True,
		                     sample_init_state=False,
		                     s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
		                     simulator=sim,
		                     reward_function=reward_function,
		                     spaces=spaces
		                     )))

	# Instantiate the RLLAB objects
	policy = GaussianLSTMPolicy(name='lstm_policy',
		                    env_spec=env.spec,
		                    hidden_dim=256,
		                    use_peepholes=True)
	baseline = LinearFeatureBaseline(env_spec=env.spec)
	optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
	sampler_cls = ASTVectorizedSampler
	algo = TRPO(
	    env=env,
	    policy=policy,
	    baseline=LinearFeatureBaseline(env_spec=env.spec),
	    batch_size=4000,
	    step_size=0.1,
	    n_itr=101,
	    store_paths=True,
	    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
	    max_path_length=50,
	    sampler_cls=sampler_cls,
	    sampler_args={"sim": sim,
		          "reward_function": reward_function})

.. _running-the-experiment:

5.4 Running the Experiment
--------------------------

When executing the experiment, only two things need to be done. Create a new tensorflow session, and then pass that to the algorithm training function. However, recording values from the experiment is trickier, since that tensorflow session is needed to unpickle the data. There are ways to save the session for later data retrieval, which can be found in the tensorflow documentation. Here, the data will be processed while the session is still active using the save_trials function. Create a ``save_trials.py`` file and add the following code:
::
	import rllab
	import joblib
	import numpy as np
	import sandbox
	import pdb
	import tensorflow as tf


	def save_trials(iters, path, header, sess, save_every_n = 100):
	    #sess.run(tf.global_variables_initializer())
	    for i in range(0, iters):
		if (np.mod(i, save_every_n) != 0):
		    continue
		with tf.variable_scope('Loader' + str(i)):
		    data = joblib.load(path + '/itr_' + str(i) + '.pkl')
		    # pdb.set_trace()
		    paths = data['paths']

		    trials = np.array([]).reshape(0, paths[0]['env_infos']['info']['cache'].shape[1])
		    crashes = np.array([]).reshape(0, paths[0]['env_infos']['info']['cache'].shape[1])
		    for n, a_path in enumerate(paths):
		        cache = a_path['env_infos']['info']['cache']
		        # pdb.set_trace()
		        cache[:, 0] = n
		        trials = np.concatenate((trials, cache), axis=0)
		        if cache[-1,-1] == 0.0:
		            crashes = np.concatenate((crashes, cache), axis=0)

		    np.savetxt(fname=path + '/trials_' + str(i) + '.csv',
		               X=trials,
		               delimiter=',',
		               header=header)

		    np.savetxt(fname=path + '/crashes_' + str(i) + '.csv',
		               X=crashes,
		               delimiter=',',
		               header=header)

Then add the following code to the runner file:
::
	with tf.Session() as sess:
	    # Run the experiment
	    algo.train(sess=sess)

	    # Write out the episode results
	    header = 'trial, step, ' + 'v_x_car, v_y_car, x_car, y_car, '
	    for i in range(0,args.num_peds):
		header += 'v_x_ped_' + str(i) + ','
		header += 'v_y_ped_' + str(i) + ','
		header += 'x_ped_' + str(i) + ','
		header += 'y_ped_' + str(i) + ','

	    for i in range(0,args.num_peds):
		header += 'a_x_'  + str(i) + ','
		header += 'a_y_' + str(i) + ','
		header += 'noise_v_x_' + str(i) + ','
		header += 'noise_v_y_' + str(i) + ','
		header += 'noise_x_' + str(i) + ','
		header += 'noise_y_' + str(i) + ','

	    header += 'reward'
	    save_trials(args.iters, args.log_dir, header, sess, save_every_n=args.snapshot_gap)

6 Running the Example
=====================

This section explains how to run the program, and what the results should look like. Double check that all of the files created earlier in the tutorial are correct (a correct version of each is already included in the repository). Also check that the conda environment is activated, and that rllab has been added to your ``PYTHONPATH``, as explained in the installation guide.

6.1 Running from the Command Line
---------------------------------

Since everything has been configured already in the runner file, running the example is easy. Use the code below in the command line to execute the example program:
::
	mkdir data
	mkdir data/example_results
	python runners/example_runner.py --log_dir <Path-To-DRL-AST>/DRL-AST/data/example_results

Here we are creating a new directory for the logging results, and passing that to the example runner. The program should run for 101 iterations, unless you have changed it. This may take some time! Afterwards, the ``example_results`` directory should contain the following files:

* ``args.txt``: A file containing a JSON dump of the arguments passed to rllab, for posterity
* ``tab.txt``: A text file containing csv-formatted optimization results from each iteration of training
* ``tex.txt``: A text file containing a copy of the text that is output to the terminal during training
* ``itr_<#>.pkl``: A pickled dictionary containing all of the available policy, optimization, and environment data for an iteration. These are created periodically according to the **--snapshot_gap** parameter for the logger (see Section 5.2)
* ``trial_<#>.csv`` and ``crashes_<#>.csv``: These are csv files contianing simulation state information sufficient to recreate the training trajectories from a specific iteration. These are created from the ``itr_<#>.pkl`` files, so the also are created periodically according to the **--snapshot_gap** parameter for the logger. These are generated by the ``save_trials.py`` file, as explained in the next section

6.2 Post-Processing Analysis
----------------------------

Whille rllab creates some logging output through its internal logger, that may not be sufficient for your needs. An alternative approach is to keep whatever information you need in the ``Simulator`` ``self._info`` that is returned to the environment. These are bundled with some other policy and optimizer data into a dictionary, that is then pickled to file. These objects can be loaded later for further analysis. Shown below is an example function that will pull some data from the files:
::
	import joblib
	import numpy as np
	import tensorflow as tf


	def example_save_trials(iters, path, header, sess, save_every_n = 100):
	    for i in range(0, iters):
		if (np.mod(i, save_every_n) != 0):
		    continue
		with tf.variable_scope('Loader' + str(i)):
		    data = joblib.load(path + '/itr_' + str(i) + '.pkl')
		    # pdb.set_trace()
		    paths = data['paths']

		    trials = np.array([]).reshape(0, paths[0]['env_infos']['info']['cache'].shape[1])
		    crashes = np.array([]).reshape(0, paths[0]['env_infos']['info']['cache'].shape[1])
		    for n, a_path in enumerate(paths):
		        cache = a_path['env_infos']['info']['cache']
		        # pdb.set_trace()
		        cache[:, 0] = n
		        trials = np.concatenate((trials, cache), axis=0)
		        if cache[-1,-1] == 0.0:
		            crashes = np.concatenate((crashes, cache), axis=0)

		    np.savetxt(fname=path + '/trials_' + str(i) + '.csv',
		               X=trials,
		               delimiter=',',
		               header=header)

		    np.savetxt(fname=path + '/crashes_' + str(i) + '.csv',
		               X=crashes,
		               delimiter=',',
		               header=header)

Here we are grabbing simulation state information, like the postion and velocities of the car and pedestrian, as well as the pedestrian accelerations, noise on the sensors, and the reward at each step. Every trial is saved to the ``trial_<#>.csv`` file, while only trajectories that end in a collsion are saved to ``crashes_<#>.csv``. These files are useful for visulazing trajectories or analyzing why a collision is occuring. 

6.3 Example Output
------------------
As you run the program, rllab will output optimization updates to the terminal. When the method runs iteration 100, you should see something that looks like this:
::
	| -----------------------  ----------------
	| PolicyExecTime                0.138965
	| EnvExecTime                   0.471907
	| ProcessExecTime               0.0285957
	| Iteration                   100
	| AverageDiscountedReturn    -897.273
	| AverageReturn             -1437.22
	| ExplainedVariance             0.136119
	| NumTrajs                     80
	| Entropy                       8.22841
	| Perplexity                 3745.86
	| StdReturn                  4448.98
	| MaxReturn                  -102.079
	| MinReturn                -24631
	| LossBefore                   -5.66416e-05
	| LossAfter                    -0.0234421
	| MeanKLBefore                  0.0725254
	| MeanKL                        0.0915881
	| dLoss                         0.0233855
	| Time                        857.771
	| ItrTime                       8.16877
	| -----------------------  ----------------

If everything works right, the max return in the last several iterations should be around -100. If you got particularly lucky, the average return may be close to that as well. For your own projects, these number may be very different, depending on your reward function. 

