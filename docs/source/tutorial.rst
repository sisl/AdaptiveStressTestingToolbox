.. _tutorial-tutorial:

Tutorial
******************
.. _tutorial-introduction:

*This tutorial is up-to-date for version `2020.09.01.1`*

1 Introduction
===============

This tutorial is intended for readers to learn how to use this package with their own simulator. Familiarity with the underlying theory is recommended, but is not strictly necessary for use. Please install the package before proceeding.

.. _tutorial-about-ast:

1.1 About AST
-----------------
Adaptive Stress Testing is a way of finding flaws in an autonomous agent. For any non-trivial problem, searching the space of a stochastic simulation is intractable, and grid searches do not perform well. By modeling the search as a Markov decision process (MDP), we can use reinforcement learning to find the most probable failure. AST treats the simulator as a black box, and only needs access in a few specific ways. To interface a simulator to the AST packages, a few things will be needed:

* A **Simulator** wrapper that exposes the simulation software to this package. See :ref:`tutorial-simulation_options` for details on closed-loop vs. open-loop Simulators
* A **Reward** function dictates the optimization goals of the algorithm.
* The **Spaces** objects give information on the size and limits of a space. This will be used to define the **Observation Space** and the **Action Space**
* A **Runner** collects all of the run options and starts the experiment.

.. _tutorial-about-this-tutorial:

1.2 About this tutorial
------------------------

In this tutorial, we will test a basic autonomous vehicle's ability to safely navigate a crosswalk. We will find the most-likely pedestrian trajectory that leads to a collision. The remainder of the tutorial is organized as follows:

-  In Section 2, we will interface with a simulator (:ref:`tutorial-creating-a-simulator`).
-  In Section 3, we will create a reward function (:ref:`tutorial-creating-a-reward-function`).
-  In Section 4, we will define the action and state spaces (:ref:`creating-the-spaces`).
-  In Section 5, we will create a runner file (:ref:`tutorial-creating-a-runner`).
-  In Section 6, we run the experiment (:ref:`tutorial-running-the-example`).

.. _tutorial-creating-a-simulator:

2 Creating a Simulator
======================

This sections explains how to create a wrapper that exposes your simulator to the AST package. The wrapper allows the AST solver to specify actions to control the stochasticity in the simulation. Examples of stochastic simulation elements could include an actor, like a pedestrian or a car, or noise elements, like on the beams of a LIDAR sensor. The simulator must be able to reset on command and detect if a goal state had been reached. The simulator state can be used, but is not necessary. Before we begin, let's define 3 different settings that tell the ASTEnv what sort of simulator it is interacting with.

We will be wrapping an example autonomous vehicle simulator that runs a toy problem of an autonomous vehicle approaching a crosswalk with pedestrians crossing. The simulator code can be found at `ast_toolbox.simulators.example_av_simulator.toy_av_simulator.py <https://github.com/sisl/AdaptiveStressTestingToolbox/blob/master/src/ast_toolbox/simulators/example_av_simulator/toy_av_simulator.py>`_.

.. _tutorial-simulation_options:

2.1 Simulation Options
---------------------------
Three options must be specified to inform ASTEnv what type of simulator it is interacting with. They are listed as follows, with the default in bold, and the actual variable name in parentheses:

* **Open-loop** vs. Closed-loop control (open_loop): A *closed-loop* simulation is one in which control can be injected at each step during the actual simulation run, vs an *open-loop* simulation where all actions must be specified ahead of time. Essentially, in a closed-loop system we are "closing the loop" by including the toolbox in the calculation of each timestep. For example, if a simulation is run by creating a specification file, and no other control is possible, that simulation would be open-loop. There is no inherent advantage to either mode, and open-loop will be far more common. Closed-loop mode will generally only be used by white-box systems, where closed-loop control is required.
* **Black box simulation state** vs. White box simulation state (blackbox_sim_state): When running in *black box* simulation mode, the solver does not have access to the true state of the simulator, instead choosing actions based on the initial condition and the history of actions taken so far. If your simulator can provide access to the simulation state, it can be faster and more efficient to run in *white box* simulation mode, in which the simulation state is used as the input to the reinforcement learning algorithm at each time step. White box simulation mode requires closed-loop control.
* **Fixed initial state** vs. Generalized initial state (fixed_init_state): A simulation with a *fixed initial state* starts every rollout from the exact same simulation state, while a simulation with a *Generalized initial state* samples from a space of initial conditions. For example, if you had a 1-D state space, starting at x=0 would be a fixed initial state, while sampling x from [-2,2] at the start of each simulation would be a generalized initial state. For more information on the specifics see `Efficient Autonomy Validation in Simulation with Adaptive Stress Testing <https://arxiv.org/abs/1907.06795>`_.

.. _tutorial-inheriting-the-base-simulator:

2.2 Inheriting the Base Simulator
---------------------------------

Start by creating a file named ``example_av_simulator.py`` in the ``simulators`` folder. Create a class titled ``ExampleAVSimulator``, which inherits from ``Simulator``.

.. code-block:: python


   import numpy as np  # Used for math

   from ast_toolbox.simulators import ASTSimulator  # import parent Simulator class
   from ast_toolbox.simulators.example_av_simulator import ToyAVSimulator  # import the simulator to wrap


   class ExampleAVSimulator(ASTSimulator):  # Define the class

The base generator accepts four values, three of which are boolean values for the settings defined in :ref:`tutorial-simulation_options`:

* **max_path_length**: The horizon of the simulation, in number of timesteps
* **open_loop**: True for open-loop simulation, False for closed-loop simulation
* **blackbox_sim_state**: True for black box simulation state, False for white box simulation state
* **fixed_init_state**: True for fixed initial simulation state, False for generalized initial simulation state

A child of the ``ASTSimulator`` class is required to define the following three functions:
   - ``simulate``.
   - ``get_reward_info``.
   - ``is_goal``.
The following functions may be optionally overridden as well:
   - ``closed_loop_step``.
   - ``reset``.
   - ``clone_state``.
   - ``restore_state``.
   - ``render``.
Finally, it is not recommended that you touch these functions:
   - ``step``.
   - ``observation_return``.
   - ``is_terminal``.

For use with the Go-Explore algorithm, the ``clone_state`` and ``restore_state`` functions must be defined.

.. _tutorial-initializing-the-example-simulator:

2.3 Initializing the Example Simulator
--------------------------------------

Our example simulator takes 3 values:

* **num\_peds**: The number of pedestrians in the scenario.
* **simulator_args**: A dict of named arguments to be passed to the toy simulator.
* **kwargs**: Any keyword arguement not listed here. In particular, the base class arguments covered in :ref:`tutorial-inheriting-the-base-simulator` should be passed to the base Simulator as one of the kwargs.

The toy simulator will control a modified version of the Intelligent Driver Model (IDM) as our system under test (SUT), while adding sensor noise and filtering it out with an alpha-beta tracker. Initial simulation conditions are needed here as well. Because of all this, the Simulator accepts a number of inputs:

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

In addition, there are a number of member variables that need to be initialized. The code is below:

.. code-block:: python

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

        # initialize the parent ASTSimulator
        super().__init__(**kwargs)

.. _tutorial-the-simulate-function:

2.4 The ``simulate`` function:
------------------------------

The simulate function runs a simulation using previously generated actions from the policy to control the stochasticity. The simulate function accepts a list of actions and an initial state. It should run the simulation, then return the timestep in which the goal state was achieved, or a -1 if the horizon was reached first. In addition, this function should return any simulation info needed for post-analysis.

For the example, out toy simulator conveniently has a single function to call that already follows the same conventions. Note that in most cases, the simulate function may require significantly more API calls to the simulator, as well as changing the inputs and outputs to forms the simulator will accept and back again. Now we implement the ``simulate`` function, checking to be sure that the horizon wasn't reached:

.. code-block:: python

    def simulate(self, actions, s_0):

        return self.simulator.run_simulation(actions=actions, s_0=s_0, simulation_horizon=self.c_max_path_length)

.. _tutorial-the-closed-loop-step-function-optional:

2.5 The ``closed_loop_step`` function (Optional):
-------------------------------------------------

If a simulation is closed-loop, the ``closed_loop_step`` function should step the simulation forward at each timestep. The functions takes as input the current action. We return the output of ``observation_return`` function defined by the ``ASTSimulator``, which ensures we return the correct values depending on the simulator settings. It is highly recommended to use this function. If the simulation is open-loop, other per-step actions can still be put here if it is desirable - this function is called at each timestep either way. Since we are running the simulator open-loop in this tutorial, we could just have this function return None. However, we have implemented the function as an example of how the simulator could be run closed-loop.

Again, our toy simulator already has a closed-loop mode that follows the same convention so we can just call the ``step_simulation`` function.

.. code-block:: python

    def closed_loop_step(self, action):

        # grab simulation state, if interactive
        self.observation = np.ndarray.flatten(self.simulator.step_simulation(action))

        return self.observation_return()

.. _tutorial-the-reset-function-optional:

2.6 The ``reset`` function (Optional):
--------------------------------------

The reset function should return the simulation to a state where it can accept the next sequence of actions. In some cases this may mean explicitly resetting the simulation parameters, like SUT location or simulation time. It could also mean opening and initializing a new instance of the simulator (in which case the ``simulate`` function should close the current instance). Your implementation of the ``reset`` function may be something else entirely, it is highly dependent on how your simulator functions. The method takes the initial state as an input, and returns the state of the simulator after the reset actions are taken. If reset is defined, ``observation_return`` should again be used to return the correct observation type. In addition, the super class's reset must still be called.

Our toy simulator already has a reset function, so we just call the super class's reset, call the toy simulator's reset, and then return ``observation_return``.

.. code-block:: python

    def reset(self, s_0):

        # Call ASTSimulator's reset function (required!)
        super(ExampleAVSimulator, self).reset(s_0=s_0)
        # Reset the simulation
        self.observation = np.ndarray.flatten(self.simulator.reset(s_0))

.. _tutorial-the-get-reward-info-function:

2.7 The ``get_reward_info`` function:
-------------------------------------

It is likely that your reward function (see :ref:`tutorial-creating-a-reward-function`) will need some information from the simulator. The reward function will be passed whatever information is returned from this function.

For the example, the example reward function uses a heuristic reward to help guide the policy toward failures -- when a trajectory ends without a crash, an extra penalty is applied that scales with the distance between the SUT and the nearest pedestrian in the last timestep. To do this, both the car and pedestrian locations are returned. In addition, boolean values indicating whether a crash has been found or if the horizon has been reached are returned. To access these values, we grab the ground truth state from the toy simulator.

.. code-block:: python

        # Get the ground truth state from the toy simulator
        sim_state = self.simulator.get_ground_truth()

        return {"peds": sim_state['peds'],
                "car": sim_state['car'],
                "is_goal": self.is_goal(),
                "is_terminal": self.is_terminal()}

.. _tutorial-the-is-goal-function:

2.8 The ``is_goal`` function:
-----------------------------

This function returns a boolean value indicating if the current state is in the goal set.

In the example, this is True if the pedestrian is hit by the car. The toy simulator has a ``collision_detected`` function that we can call to check for a collision.

.. code-block:: python

    def is_goal(self):

        # Ask the toy simulator if a collision was detected
        return self.simulator.collision_detected()

.. _tutorial-the-log-function-optional:

2.9 The ``log`` function (Optional):
------------------------------------

The log function is a way to store variables from the simulator for later access.

In the example, some simulation state information is appended to a list at every timestep after getting the ground truth from the toy simulator.

.. code-block:: python

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

.. _tutorial-the-clone-state-and-restore-state-functions-optional:

2.10 The ``clone_state`` and ``restore_state`` functions (Optional):
--------------------------------------------------------------------

Some parts of the Toolbox (for example, Go-Explore and the Backward Algorithm) rely on deterministic resets of the simulator to find failures efficiently. The ``clone_state`` and ``restore_state`` functions provide this functionality.

The ``clone_state`` function should return a 1-D numpy array with enough information to deterministically reset the simulation to an exact state.

In our example, the toy simulator's ``get_ground_truth`` returns a dictionary of state variables, so we arrange them into a numpy array:

.. code-block:: python

    def clone_state(self):

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

The ``restore_state`` function should accept a 1-D array and use it to deterministically reset it to a specific state. How you do the reset is up to you, whether it is through a reset style scenario instantiation, through running the simulator from the start back to the exact same point, or another method altogether.

The toy simulator has a ``set_ground_truth`` function that sets it to a specific state, so we will use that. We take the 1-D array and translate it back into a dictionary of state variables that the toy simulator wants. We also set the state variables of the ``ExampleAVSimulator``:

.. code-block:: python

    def restore_state(self, in_simulator_state):

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

.. _tutorial-creating-a-reward-function:

3 Creating a Reward Function
============================

This section explains how to create a function that dictates the reward at each timestep of a simulation. AST formulates the problem of searching the space of possible rollouts of a stochastic simulation as an MDP so that modern-day reinforcement learning (RL) techniques can be used. When optimizing a policy using RL, the reward function is of the utmost importance, as it determines what behavior the agent will learn. Changing the reward function to achieve the desired policy is known as reward shaping.

.. _tutorial-reward-shaping:

3.1 Reward Shaping
------------------


**SPOILER ALERT**: This section uses a famous summer-camp game as an example. If you are planning on attending a children's summer-camp in the near future I highly recommend you skip this section, lest you ruin the counselors' attempts at having fun at your expense. You have been warned.

As an example of reinforcement learning, and the importance of the reward function, consider the famous children's game "The Hat Game." Common at summer-camps, the game usually starts with a counselor holding a hat in his hands, telling the kids he is about to teach them a new game. He will say "Ok, ready everyone....? I can play the hat game," proceed to do a bunch of random things with the hat, such as flipping it over or tossing it in the air, and then say "how about you?" He will then pass the hat to a camper, who repeats almost exactly everything the counselor does, but is told "no, you didn't play the hat game." Another counselor will take the hat, say the words, do something completely different with it, and the game is on. The trick is actually the word "OK" - so long as you say that magic word, you have played the hat game, even if you have no hat.

How does this relate to reward shaping? In this case, the children are the policy. They are taking stochastic actions, trying to learn how to play the hat game. The key to the game being fun is that the children are predisposed to pay attention to the hat motions, but not the words beforehand. However, after enough trials (and it can take a long time), most of them will pick up the pattern and attention will shift to "OK." In the vanilla game, there are two rewards. "Yes, you played the hat game" can be considered positive, and "No, you didn't play the hat game" can be considered negative, or just zero. By changing this reward, we could make the game difficulty radically different. Imagine if 10 kids tried the game, and all they got was a binary response on if at least one of them played the game. This would be much harder to pick up on! This is an example of a sparse reward function, or one that only rarely gives rewards, such as at the end of a trajectory. On the other hand, what if the children received feedback after every single word or motion on if they had played the hat game during that trial yet. The game would be much easier! These are examples of how different reward functions can make achieving the same policy easier or harder.

How does this relate yo our tutorial? Similar to the kids, our policy will be trying to learn the correct behavior from rewards. While some policies may be better at this task than others, all of them will struggle if the reward function is too sparse. We can make the task much easier, and therefore get better and faster results, if we can introduce heuristic rewards that guide our policy to failures.
.. _tutorial-inheriting-the-base-reward-function:

3.2 Inheriting the Base Reward Function
---------------------------------------

Start by creating a file named ``example_av_reward.py`` in the ``rewards`` folder. Create a class title ``ExampleAVReward`` which inherits from ``ASTReward``:

.. code-block:: python

   import numpy as np # useful packages for math

   from ast_toolbox.rewards import ASTReward # import base class


   # Define the class, inherit from the base
   class ExampleAVReward(ASTReward):

The base class does not take any inputs, and there is only one required function - ``give_reward``.

.. _tutorial-initializing-the-example-reward-function:

3.3 Initializing the Example Reward Function
--------------------------------------------

The reward function will be calculating some rewards based on the probability of certain actions. We have assumed the means action is the 0 vector, but we still need to take the following inputs:

* **num\_peds**: The number of pedestrians in the scenario
* **cov\_x**: The covariance of the gaussian distribution used to model the x-acceleration of a pedestrian
* **cov\_y**: The covariance of the gaussian distribution used to model the y-acceleration of a pedestrian
* **cov\_sensor\_noise**: The covariance of the gaussian distribution used to model the noise on a sensor measurement in both the x and y directions (assumed equal)
* **use\_heuristic**: Whether our reward function should use the heuristic reward we provide. As mentioned above, using this reward, when possible, will improve results and decrease training time.

The code is below:

.. code-block:: python

    def __init__(self,
                 num_peds=1,
                 cov_x=0.1,
                 cov_y=0.01,
                 cov_sensor_noise=0.1,
                 use_heuristic=True):

        self.c_num_peds = num_peds
        self.c_cov_x = cov_x
        self.c_cov_y = cov_y
        self.c_cov_sensor_noise = cov_sensor_noise
        self.use_heuristic = use_heuristic
        super().__init__()

.. _tutorial-the-give-reward-function:

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

For case 2, we use the distance between the car and the closest pedestrian as a heuristic to increase convergence speed. In the early trials, this teaches pedestrians to end closer to the car, which makes it easier to find crash trajectories (see :ref:`tutorial-reward-shaping`). For case 3, using the negative log-likelihood allows us to sum the rewards to find a value that is proportional to the probability of the trajectory. As a stand in for the probability of an action, we use the Mahalanobis distance, a multi-dimensional generalization of distance from the mean. Add the following helper function to your file:

.. code-block:: python

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

Now we are ready to calculate the reward. The ``give_reward`` function takes in an action, as well as the info bundle that was returned from the ``get_reward_info`` function in the ``ExampleAVSimulator`` (see :ref:`tutorial-the-get-reward-info-function`). The code is as follows:

.. code-block:: python

    def give_reward(self, action, **kwargs):
        # get the info from the simulator
        info = kwargs['info']
        peds = info["peds"]
        car = info["car"]
        is_goal = info["is_goal"]
        is_terminal = info["is_terminal"]
        dist = peds[:, 2:4] - car[2:4]

        # update reward and done bool

        if (is_goal):  # We found a crash
            reward = 0
        elif (is_terminal):
            # reward = 0
            # Heuristic reward based on distance between car and ped at end
            if self.use_heuristic:
                heuristic_reward = np.min(np.linalg.norm(dist, axis=1))
            else:
                # No Herusitic
                heuristic_reward = 0
            reward = -100000 - 10000 * heuristic_reward  # We reached
            # the horizon with no crash
        else:
            reward = -self.mahalanobis_d(action)  # No crash or horizon yet

        return reward

.. _creating-the-spaces:

4 Creating the Spaces
=====================

This section shows how to create the action space and observation space for ``garage`` to use. The spaces define the limits of what is possible for inputs to and outputs from the policy. The observation space can be used as input if the simulation state is accessible, and can be used to generate initial conditions if they are being sampled from a range. The action space defines the output space of the policy, and controls the size of the output array from the policy.

.. _tutorial-inheriting-the-base-spaces:

4.1 Inheriting the Base Spaces
------------------------------

Create a file named ``example_av_spaces.py`` in the ``spaces`` folder. Create a class titled ``ExampleAVSpaces`` which inherits from ``ASTSpaces``:

.. code-block:: python

   import numpy as np
   from gym.spaces.box import Box

   from ast_toolbox.spaces import ASTSpaces


   class ExampleAVSpaces(ASTSpaces):

The base spaces don't take any input, but there are two functions to define: ``action_space`` and ``observation_space``. Both of these functions should return an object that inherits from the ''Space'' class, imported from ``gym.spaces``. There are a few options, and you can implement your own, but the ``Box`` class is used here. A ``Box`` is defined by two arrays, ``low`` and ``high``, of equal length, which specify the minimum and maximum value of each position in the array. The space then allows any continuous number between the low and high values.

.. _tutorial-initializing-the-spaces:

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
* **open\_loop**: Whether or not the simulation is being run in open-loop mode (See :ref:`tutorial-simulation_options`)

The initialization code is below:

.. code-block:: python

    def __init__(self,
                 num_peds=1,
                 max_path_length=50,
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
                 car_init_x=-35.0,
                 car_init_y=0.0,
                 open_loop=True,
                 ):

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
        self.open_loop = open_loop
        self.low_start_bounds = [-1.0, -6.0, -1.0, 5.0, 0.0, -6.0, 0.0, 5.0]
        self.high_start_bounds = [1.0, -1.0, 0.0, 9.0, 1.0, -2.0, 1.0, 9.0]
        self.v_start = [1.0, -1.0, 1.0, -1.0]
        super().__init__()

.. _tutorial-the-action-space:

4.3 The Action Space
--------------------

The ``action_space`` function takes no inputs and returns a child of the ``Space`` class. The length of the action space array determines the output dimension of the policy. Note the ``@Property`` decorator in the code below:

.. code-block:: python

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        low = np.array([self.c_x_accel_low, self.c_y_accel_low, -3.0, -3.0, -3.0, -3.0])
        high = np.array([self.c_x_accel_high, self.c_y_accel_high, 3.0, 3.0, 3.0, 3.0])

        for i in range(1, self.c_num_peds):
            low = np.hstack((low, np.array([self.c_x_accel_low, self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])))
            high = np.hstack((high, np.array([self.c_x_accel_high, self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])))

        return Box(low=low, high=high, dtype=np.float32)

.. _tutorial-the-observation-space:

4.4 The Observation Space
-------------------------

The ``observation_space`` function takes no inputs and returns a child of the ``Space`` class. If the simulation state is accessible, the ranges of possible values should be defined using this function, which determines the expected input shape to the policy. If initial conditions are sampled, they will be sampled from the observation space. Therefore, the observation space should define the maximum and minimum value of every simulation state that will be passed as input to the policy, as well as a value for every initial condition needed to specify a scenario variation. Note the ``@Property`` decorator in the code below:

.. code-block:: python

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

        if self.open_loop:
            low = self.low_start_bounds[:self.c_num_peds * 2]
            low = low + np.ndarray.tolist(0.0 * np.array(self.v_start))[:self.c_num_peds]
            low = low + [0.75 * self.c_v_des]

            high = self.high_start_bounds[:self.c_num_peds * 2]
            high = high + np.ndarray.tolist(2.0 * np.array(self.v_start))[:self.c_num_peds]
            high = high + [1.25 * self.c_v_des]

            if self.c_car_init_x > 0:
                low = low + [0.75 * self.c_car_init_x]
                high = high + [1.25 * self.c_car_init_x]
            else:
                low = low + [1.25 * self.c_car_init_x]
                high = high + [0.75 * self.c_car_init_x]

        return Box(low=np.array(low), high=np.array(high), dtype=np.float32)

.. _tutorial-creating-a-runner:

5 Creating a Runner
===================

This section explains how to create a file to run the experiment we have been creating. This will use all of the example files we have created, and interface them with the a package for handling RL. The backend framework handling the policy definition and optimization is a package called RLLAB. The project is open-source, so if you would like to understand more about what RLLAB is doing please see the documentation here.

.. _tutorial-setting-up-the-runners:

5.1 Setting Up the Runners
--------------------------

Create a file called ``example_runner.py`` in your working directory. Add the following code to handle all of the necessary imports:

.. code-block:: python

   # Import the example classes
   import os

   import fire
   # Useful imports
   import tensorflow as tf
   from garage.envs.normalized_env import normalize
   from garage.experiment import run_experiment
   from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
   # Import the necessary garage classes
   from garage.tf.algos.ppo import PPO
   from garage.tf.envs.base import TfEnv
   from garage.tf.experiment import LocalTFRunner
   from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
   from garage.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
   # from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
   from garage.tf.policies import GaussianLSTMPolicy

   # Import the AST classes
   from ast_toolbox.envs import ASTEnv
   from ast_toolbox.rewards import ExampleAVReward
   from ast_toolbox.samplers import ASTVectorizedSampler
   from ast_toolbox.simulators import ExampleAVSimulator
   from ast_toolbox.spaces import ExampleAVSpaces
   from ast_toolbox.utils.go_explore_utils import load_convert_and_save_expert_trajectory

.. _tutorial-specifying-the-experiment:

5.2 Specifying the Experiment
-----------------------------

All of the classes imported earlier will now be used to specify the experiment. We will create a ``runner`` function that takes in dictionaries of keyword arguments for the different objects. The function will define a ``run_task`` function that executes an experiment, and then will pass this function's handle to the ``run_experiment`` function. See the ``garage`` docs for more info.

.. code-block:: python

   def runner(
       env_args=None,
       run_experiment_args=None,
       sim_args=None,
       reward_args=None,
       spaces_args=None,
       policy_args=None,
       baseline_args=None,
       algo_args=None,
       runner_args=None,
       sampler_args=None,
       save_expert_trajectory=False,
   ):

       if env_args is None:
           env_args = {}

       if run_experiment_args is None:
           run_experiment_args = {}

       if sim_args is None:
           sim_args = {}

       if reward_args is None:
           reward_args = {}

       if spaces_args is None:
           spaces_args = {}

       if policy_args is None:
           policy_args = {}

       if baseline_args is None:
           baseline_args = {}

       if algo_args is None:
           algo_args = {}

       if runner_args is None:
           runner_args = {'n_epochs': 1}

       if sampler_args is None:
           sampler_args = {}

       if 'n_parallel' in run_experiment_args:
           n_parallel = run_experiment_args['n_parallel']
       else:
           n_parallel = 1
           run_experiment_args['n_parallel'] = n_parallel

       if 'max_path_length' in sim_args:
           max_path_length = sim_args['max_path_length']
       else:
           max_path_length = 50
           sim_args['max_path_length'] = max_path_length

       if 'batch_size' in runner_args:
           batch_size = runner_args['batch_size']
       else:
           batch_size = max_path_length * n_parallel
           runner_args['batch_size'] = batch_size

       def run_task(snapshot_config, *_):

           config = tf.ConfigProto()
           config.gpu_options.allow_growth = True
           with tf.Session(config=config) as sess:
               with tf.variable_scope('AST', reuse=tf.AUTO_REUSE):

                   with LocalTFRunner(
                           snapshot_config=snapshot_config, max_cpus=4, sess=sess) as local_runner:
                       # Instantiate the example classes
                       sim = ExampleAVSimulator(**sim_args)
                       reward_function = ExampleAVReward(**reward_args)
                       spaces = ExampleAVSpaces(**spaces_args)

                       # Create the environment
                       if 'id' in env_args:
                           env_args.pop('id')
                       env = TfEnv(normalize(ASTEnv(simulator=sim,
                                                    reward_function=reward_function,
                                                    spaces=spaces,
                                                    **env_args
                                                    )))

                       # Instantiate the garage objects
                       policy = GaussianLSTMPolicy(env_spec=env.spec, **policy_args)

                       baseline = LinearFeatureBaseline(env_spec=env.spec, **baseline_args)

                       optimizer = ConjugateGradientOptimizer
                       optimizer_args = {'hvp_approach': FiniteDifferenceHvp(base_eps=1e-5)}

                       algo = PPO(env_spec=env.spec,
                                  policy=policy,
                                  baseline=baseline,
                                  optimizer=optimizer,
                                  optimizer_args=optimizer_args,
                                  **algo_args)

                       sampler_cls = ASTVectorizedSampler
                       sampler_args['sim'] = sim
                       sampler_args['reward_function'] = reward_function

                       local_runner.setup(
                           algo=algo,
                           env=env,
                           sampler_cls=sampler_cls,
                           sampler_args=sampler_args)

                       # Run the experiment
                       local_runner.train(**runner_args)
                       print('done!')

       run_experiment(
           run_task,
           **run_experiment_args,
       )

.. _tutorial-running-the-experiment:

5.3 Running the Experiment
--------------------------

Now create a file named ``example_batch_runner.py``. While ``example_runner.py`` gave us a runner template, the batch runner will be where we specify the actual arguments that define our experiment set-up. By dividing the files in this way, it makes it much easier to set-up and run many different experiment specifications at once.

.. code-block:: python

   import pickle

   from examples.AV.example_runner_drl_av import runner as drl_runner

   if __name__ == '__main__':
       # Overall settings
       max_path_length = 50
       s_0 = [0.0, -4.0, 1.0, 11.17, -35.0]
       base_log_dir = './data'
       # experiment settings
       run_experiment_args = {'snapshot_mode': 'last',
                              'snapshot_gap': 1,
                              'log_dir': None,
                              'exp_name': None,
                              'seed': 0,
                              'n_parallel': 8,
                              'tabular_log_file': 'progress.csv'
                              }

       # runner settings
       runner_args = {'n_epochs': 101,
                      'batch_size': 5000,
                      'plot': False
                      }

       # env settings
       env_args = {'id': 'ast_toolbox:GoExploreAST-v1',
                   'blackbox_sim_state': True,
                   'open_loop': False,
                   'fixed_init_state': True,
                   's_0': s_0,
                   }

       # simulation settings
       sim_args = {'blackbox_sim_state': True,
                   'open_loop': False,
                   'fixed_initial_state': True,
                   'max_path_length': max_path_length
                   }

       # reward settings
       reward_args = {'use_heuristic': True}

       # spaces settings
       spaces_args = {}

       # DRL Settings

       drl_policy_args = {'name': 'lstm_policy',
                          'hidden_dim': 64,
                          }

       drl_baseline_args = {}

       drl_algo_args = {'max_path_length': max_path_length,
                        'discount': 0.99,
                        'lr_clip_range': 1.0,
                        'max_kl_step': 1.0,
                        # 'log_dir':None,
                        }


       # DRL settings
       exp_log_dir = base_log_dir
       run_experiment_args['log_dir'] = exp_log_dir + '/drl'
       run_experiment_args['exp_name'] = 'drl'

       drl_runner(
           env_args=env_args,
           run_experiment_args=run_experiment_args,
           sim_args=sim_args,
           reward_args=reward_args,
           spaces_args=spaces_args,
           policy_args=drl_policy_args,
           baseline_args=drl_baseline_args,
           algo_args=drl_algo_args,
           runner_args=runner_args,
       )

.. _tutorial-running-the-example:

6 Running the Example
=====================

This section explains how to run the program, and what the results should look like. Double check that all of the files created earlier in the tutorial are correct (a correct version of each is already included in the repository). Also check that the conda environment is activated, and that garage has been added to your ``PYTHONPATH``, as explained in the installation guide.

6.1 Running from the Command Line
---------------------------------

Since everything has been configured already in the runner file, running the example is easy. Use the code below in the command line to execute the example program from the top-level directory:

.. code-block:: python

	mkdir data
	python example_batch_runner.py

Here we are creating a new directory for the output, and then running the batch runner we created above (see :ref:`tutorial-running-the-experiment`). The program should run for 101 iterations, unless you have changed it. This may take some time!

6.2 Example Output
------------------

As you run the program, rllab will output optimization updates to the terminal. When the method runs iteration 100, you should see something that looks like this::

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

If everything works right, the max return in the last several iterations should be around -100. If you got particularly lucky, the average return may be close to that as well. For your own projects, these numbers may be very different, depending on your reward function.

