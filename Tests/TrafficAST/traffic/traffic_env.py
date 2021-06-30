import random
import numpy as np
import scipy.spatial.distance as ssd

import gym
from gym import spaces
from gym.utils import seeding

from traffic.road import Road, RoadSegment
from traffic.car import Car
from traffic.drivers.driver import XYSeperateDriver
from traffic.drivers.oned_drivers import IDMDriver, PDDriver
from traffic.constants import *

class TrafficEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self,
            road,
            cars,
            drivers,
            dt=0.1,
            ):

        self.dt = dt

        self._viewer = None

        self._road = road

        self._cars = cars

        self._drivers = drivers

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        [car.seed(seed) for car in self._cars]
        [driver.seed(seed) for driver in self._drivers]
        return [seed]

    @property
    def observation_space(self):
        return self._cars[0].observation_space(self._cars, self._road, include_x=False)

    @property
    def action_space(self):
        return self._cars[0].action_space()

    def reset(self):
        self._reset()
        self.close()
        return self.observe()

    def _reset(self):
        pass

    def step(self, action):

        self.update(action)

        obs = self.observe()

        reward = self.get_reward()

        done = self.is_terminal()

        info = self.get_info()

        return obs, reward, done, info

    def update(self, action):
        for driver in self._drivers:
            driver.observe(self._cars, self._road)
        self._actions = [driver.get_action() for driver in self._drivers]
        [action.update(car, self.dt) for (car, action) in zip(self._cars, self._actions)]

    def observe(self):
        return self._cars[0].observe(self._cars, self._road, include_x=False)

    def get_reward(self):
        return 0.0

    def get_info(self):
        return {}

    def is_terminal(self):
        return False

    def setup_viewer(self):
        from traffic import rendering
        self.viewer = rendering.Viewer(800, 800)
        self.viewer.set_bounds(-20.0, 20.0, -20.0, 20.0)

    def setup_extra_render(self, extra_input):
        pass

    def update_extra_render(self, extra_input):
        pass

    def get_camera_center(self):
        return self._cars[0].position

    def render(self, mode='human', screen_size=800, extra_input=None):
        if (not hasattr(self, 'viewer')) or (self.viewer is None):
            self.setup_viewer()

            self._road.setup_render(self.viewer)

            for driver in self._drivers:
                driver.setup_render(self.viewer)

            for car in self._cars:
                car.setup_render(self.viewer)

            self.setup_extra_render(extra_input)

        camera_center = self.get_camera_center()
        self._road.update_render(camera_center)

        for driver in self._drivers:
            driver.update_render(camera_center)

        for cid, car in enumerate(self._cars):
            car.update_render(camera_center)

        self.update_extra_render(extra_input)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    import time
    road=Road([RoadSegment([(-20.,-1.5),(100,-1.5),(100,7.5),(-20,7.5)])])
    dt = 0.1
    n_cars=2
    car_class=Car
    driver_class=XYSeperateDriver
    x_driver_class=IDMDriver
    y_driver_class=PDDriver
    driver_sigma=0.0
    car_length=5.0
    car_width=2.0
    car_max_accel=10.0
    car_max_speed=40.0
    car_expose_level=4
    cars = [car_class(idx=cid, length=car_length, width=car_width, color=random.choice(BLUE_COLORS),
                      max_accel=car_max_accel, max_speed=car_max_speed,
                      expose_level=car_expose_level) for cid in range(n_cars)
            ]
    cars[0].set_position(np.array([0.0, 0.0]))
    cars[0].set_velocity(np.array([0.0, 0.0]))
    cars[1].set_position(np.array([20.0, 0.0]))
    cars[1].set_velocity(np.array([0.0, 0.0]))

    drivers = [driver_class(idx=did, car=car, dt=dt,
                x_driver=x_driver_class(idx=did, car=car, sigma=driver_sigma, axis=0), 
                y_driver=y_driver_class(idx=did, car=car, sigma=driver_sigma, axis=1)) 
                for (did, car) in enumerate(cars)
                ]
    drivers[0].x_driver.set_v_des(10.0)
    drivers[0].y_driver.set_p_des(3.0)
    drivers[1].x_driver.set_v_des(0.0)
    drivers[1].y_driver.set_p_des(0.0)

    env = TrafficEnv(road, cars, drivers, dt)
    obs = env.reset()
    img = env.render()
    done = False
    while True:  #not done:
        obs, reward, done, info = env.step(None)
        print('obs: ', obs)
        print('reward: ', reward)
        print('info: ', info)
        env.render()
        time.sleep(0.1)
    env.close()
