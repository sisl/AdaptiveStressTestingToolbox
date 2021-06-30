import numpy as np
import scipy.spatial.distance as ssd

import gym
from gym import spaces
from gym.utils import seeding


from traffic.actions.xy_accel_action import XYAccelAction

class Driver:
	def __init__(self, idx, car, dt):
		self._idx = idx
		self.car = car
		self.dt = dt

		self.seed()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def observe(self, cars, road):
		pass

	def get_action(self):
		pass

	def reset(self):
		pass

	def setup_render(self, viewer):
		pass

	def update_render(self, camera_center):
		pass

	def remove_render(self, viewer):
		pass

class OneDDriver(Driver):
    def __init__(self, axis, direction=1, **kwargs):
        self.set_axis(axis)
        self.set_direction(direction)
        super(OneDDriver, self).__init__(**kwargs)

    def set_axis(self, axis):
        if axis == 0:
            self.axis0 = 0
            self.axis1 = 1
        else:
            self.axis0 = 1
            self.axis1 = 0

    def set_direction(self,direction):
        self.direction = direction

class XYSeperateDriver(Driver):
	def __init__(self, x_driver, y_driver, **kwargs):
		self.x_driver = x_driver
		self.y_driver = y_driver
		super(XYSeperateDriver, self).__init__(**kwargs)
		assert self.x_driver.car is self.car
		assert self.y_driver.car is self.car

	def observe(self, cars, road):
		self.x_driver.observe(cars, road)
		self.y_driver.observe(cars, road)

	def get_action(self):
		a_x = self.x_driver.get_action()
		a_y = self.y_driver.get_action()
		return XYAccelAction(a_x, a_y)

	def reset(self):
		self.x_driver.reset()
		self.y_driver.reset()









