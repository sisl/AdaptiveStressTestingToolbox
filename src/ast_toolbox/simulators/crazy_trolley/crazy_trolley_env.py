import gym
from gym import spaces

import dill

from matplotlib import pyplot as plt
import numpy as np

from garage.envs.base import Step

from ast_toolbox.simulators.crazy_trolley.crazy_trolley import CrazyTrolleyHeadlessGame, CrazyTrolleyRenderedGame
import pdb

class CrazyTrolleyEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, height=16, width=32, from_pixels=True, rgb=True):
        super(CrazyTrolleyEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input:
        if from_pixels:
            shape = (height, width, 3)
        else:
            shape = (height, width)
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        self.renderer = CrazyTrolleyRenderedGame(ax=None, height=height, width=width, rgb=rgb)
        self.game = self.renderer.game

        self.score = 0

    def step(self, action):
        self.game.player_action(action)
        self.game.tick()

        self.renderer.update_frame()

        observation = self.renderer.display_frame

        reward = self.game.score - self.score
        self.score = self.game.score

        done = self.game.game_over

        info = np.array([])

        return Step(observation=observation,
                    reward=reward,
                    done=done)
        # return observation, reward, done, info

    def reset(self):
        self.score = 0

        self.renderer.new_game()
        observation = self.renderer.display_frame

        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        # display_frame = self.renderer.get_display_frame(self.game.frame)
        self.renderer.update_frame()
        plt.imshow(self.renderer.display_frame, interpolation='none')
        plt.show()

    def close (self):
        pass

    def get_action_meanings(self):
        return self.game.action_meaning_dict

    def __getstate__(self):
        # pdb.set_trace()
        pickle_dict = self.__dict__.copy()
        pickle_dict['renderer'] = dill.dumps(self.__dict__['renderer'] )
        pickle_dict.pop('game')
        return pickle_dict

    def __setstate__(self, pickle_dict):
        self.__dict__ = pickle_dict
        self.__dict__['renderer'] = dill.loads(pickle_dict['renderer'])
        self.__dict__['game'] = self.__dict__['renderer'].game

if __name__ == '__main__':
    env = CrazyTrolleyEnv()
    for i in range(1000):
        if i % 10 == 0:
            env.step(action=np.random.randint(low=0, high=5))
            env.render()
        else:
            env.step(action=0)
