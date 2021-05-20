import time

import gym
from gym import spaces

import dill

from matplotlib import pyplot as plt
import numpy as np

from garage.envs.base import Step

from ast_toolbox.simulators.crazy_trolley.crazy_trolley import CrazyTrolleyHeadlessGame, CrazyTrolleyRenderedGame
import pdb

def show_plot(plt):
    plt.show()

class CrazyTrolleyEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, height=16, width=32, from_pixels=True, rgb=True, random_level=False):
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

        # self.fig, self.ax = plt.subplots()
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        # self.canvas = self.ax.figure.canvas

        self.renderer = CrazyTrolleyRenderedGame(ax=None, height=height, width=width, rgb=rgb)
        self.game = self.renderer.game

        self.score = 0

        self.rendered = False
        self.random_level = random_level

    def step(self, action, skip=4):
        self.game.player_action(action)

        self.game.tick()
        for _ in range(skip):
            # Skip a certain number of frames to prevent insane action speed
            self.game.tick()

        self.renderer.update_frame()

        observation = self.renderer.display_frame

        reward = self.game.score - self.score
        # Small penalty for taking an action to reduce spurious actions
        if action > 0:
            reward -= 1
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

        if self.random_level:
            self.game._level = np.random.randint(low=0, high=50)
            self.game.new_frame()
            self.renderer.update_frame()
        observation = self.renderer.display_frame

        self.rendered = False

        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        # display_frame = self.renderer.get_display_frame(self.game.frame)
        if not self.rendered:
            self.rendered = True

            # plt.ion()
            # self.ax = plt.gca()

            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)

            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)
            for (_, spine) in self.ax.spines.items():
                # spine.set_color('0.0')
                # spine.set_linewidth(1.0)
                spine.set_color(None)

            self.fig.canvas.draw()

            self.disp = self.ax.imshow(self.renderer.display_frame_with_header, interpolation='none')
            self.title = self.ax.text(0.5, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                                      transform=self.ax.transAxes, ha="center")

            # plt.draw()
            plt.show(block=False)
            # pdb.set_trace()

        self.renderer.update_frame()

        # self.disp = self.ax.imshow(self.renderer.display_frame_with_header, interpolation='none')
        # self.title = self.ax.text(0.5, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
        #                           transform=self.ax.transAxes, ha="center")

        self.disp.set_data(self.renderer.display_frame_with_header)
        # self.ax.draw_artist(self.disp)
        #
        self.title.set_text('Level: {level:03d}          Score: {score:07d}          Lives: {lives}'.format(
            level=self.game.level,
            score=self.game.score,
            lives=self.game.lives))

        self.fig.canvas.draw()

        # plt.draw()

        # pdb.set_trace()
        # self.ax.draw_artist(self.title)


    def close (self):
        pdb.set_trace()
        if self.rendered:
            plt.ioff()
            plt.close()


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
    env.reset()
    for i in range(1000):
        if i % 10 == 0:
            env.step(action=np.random.randint(low=0, high=5))
            env.render()
            # pdb.set_trace()
            # timestep = 1.0
            # time.sleep(timestep)
        else:
            env.step(action=0)
