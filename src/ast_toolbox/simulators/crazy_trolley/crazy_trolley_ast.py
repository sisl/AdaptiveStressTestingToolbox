"""Class template to wrap a simulator for interaction with AST."""
import numpy as np
import tensorflow as tf
import warnings

from collections import deque
from skimage import color, img_as_ubyte
from skimage.transform import resize

from garage.experiment import Snapshotter

from ast_toolbox.simulators import ASTSimulator
from ast_toolbox.simulators.crazy_trolley.crazy_trolley import CrazyTrolleyRenderedGame


class CrazyTrolleyASTSimulator(ASTSimulator):
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

    def __init__(self, height=16, width=32, from_pixels=True, rgb=True, random_level=False,
                 skip=5, noop=30, stack_frames=5, max_and_skip=True, grayscale=True, resize = None,
                 **kwargs):

        super().__init__(**kwargs)

        snapshotter = Snapshotter()
        self.session = tf.compat.v1.Session()
        with self.session.as_default():  # optional, only for TensorFlow
            data = snapshotter.load('./examples/crazy_trolley/crazy_trolley_suts/experiment_2021_05_24_11_26_48_0001')
            self.agent = data['algo'].policy

            self.skip = skip
            self.noop = noop
            self.max_and_skip = max_and_skip
            self.stack_frames = stack_frames
            self.player_observation = deque(maxlen=self.stack_frames)
            self.grayscale = grayscale
            self.resize = resize
            if self.resize is None:
                self.resize = (height, width)
            self._obs_buffer = np.zeros((2,) + self.resize,
                                        dtype=np.uint8)

            self.renderer = CrazyTrolleyRenderedGame(ax=None, height=height, width=width, rgb=rgb)
            self.game = self.renderer.game

            self.score = 0

            self.rendered = False
            self.random_level = random_level


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
        self.reset(s_0)
        while self._path_length <= self.c_max_path_length:
            self._path_length += 1
            action = actions[self._path_length]
            self.closed_loop_step(action)

            if self.game.game_over():
                return self._path_length, np.array(self._info)

        return -1, np.array(self._info)

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
        with self.session.as_default() :
            print('Closed Loop Step')
            while not (self.game.game_over or self.game.end_of_frame):
                # self.renderer.update_frame()
                # player_observation = self.process_frame(self.renderer.display_frame)
                player_action = self.agent.get_action(self._stack_frames())

                self.game.player_action(player_action)

                self.game.tick()

                self.skip_frames(self.skip)

            if not self.game.game_over:
                while(self.game.end_of_frame):
                    # Force game updates until the next frame generates
                    self.game.tick()



            self.observation = action

            return self.observation_return()

    def skip_frames(self, skip):
        for i in range(skip):
            # Skip a certain number of frames to prevent insane action speed
            self.game.tick()
            if self.max_and_skip:
                if i == self.skip - 2:
                    self.renderer.update_frame()
                    obs = self.process_frame(self.renderer.display_frame)
                    self._obs_buffer[0] = self.process_frame(obs)
                elif i == self.skip - 1:
                    self.renderer.update_frame()
                    obs = self.process_frame(self.renderer.display_frame)
                    self._obs_buffer[1] = obs

        if self.max_and_skip:
            self.player_observation.append(self._obs_buffer.max(axis=0))
        else:
            self.renderer.update_frame()
            self.player_observation.append(self.process_frame(self.renderer.display_frame))

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
        self.score = 0

        self.renderer.new_game()

        if self.random_level:
            self.start_level = np.random.randint(low=0, high=50)
            self.game._level = self.start_level
            self.game.new_frame()

        # Fill up frame stack
        self.renderer.update_frame()
        self.player_observation.clear()
        for _ in range(self.stack_frames):
            self.player_observation.append(self.process_frame(self.renderer.display_frame))


        # Handle noop actions
        self.skip_frames(np.random.randint(low=self.skip, high=self.noop))
        # for _ in range(self.noop):
        #     self.game.tick()

        # self.renderer.update_frame()
        # self.observation = self.renderer.display_frame
        self.observation = np.array([0])

        self.rendered = False

        return super(CrazyTrolleyASTSimulator, self).reset(s_0=s_0)

    def get_reward_info(self):
        """
        Returns any info needed by the reward function to calculate the current reward.
        """
        return {"frame_probability": self.game.frame_probability,
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
        return self.game.game_over

    def is_terminal(self):
        """
        Returns whether rollout horizon has been reached.
        Returns
        -------
        bool
            True if rollout horizon has been reached.
        """
        return self.game.game_over

    def log(self):
        """
        perform any logging steps
        """
        pass

    def clone_state(self):
        """Clone the simulator state for later resetting.

        This function is used in conjunction with `restore_state` for Go-Explore and Backwards Algorithm
        to do their deterministic resets.

        Returns
        -------
        array_like
            An array of all the simulation state variables.

        """
        return np.array([0])

    def restore_state(self, in_simulator_state):
        """Reset the simulation deterministically to a previously cloned state.

        This function is used in conjunction with `clone_state` for Go-Explore and Backwards Algorithm
        to do their deterministic resets.

        Parameters
        ----------
        in_simulator_state : array_like
            An array of all the simulation state variables.

        """
        pass

    def render(self, **kwargs):
        """Either renders a simulation scene or returns data used for external rendering.

        Parameters
        ----------
        kwargs :
            Keyword arguments used in the simulators `render` function.
        """
        pass

    def _stack_frames(self):
        return np.stack(self.player_observation, axis=2)

    def process_frame(self, frame):
        # handle frame processing like grayscale, stacking, or resizing
        with warnings.catch_warnings():
            """
            Suppressing warnings for
            1. possible precision loss when converting from float64 to uint8
            2. anti-aliasing will be enabled by default in skimage 0.15
            """
            warnings.simplefilter('ignore')
            # Save dtype to cast after processing
            dtype = frame.dtype
            # Set to greyscale
            if self.grayscale:
                frame = img_as_ubyte(color.rgb2gray((frame)))
            # Resize frame
            frame = resize(frame, (self.resize))  # now it's float

            return frame.astype(dtype)

