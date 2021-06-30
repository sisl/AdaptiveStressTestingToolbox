import numpy as np
import scipy.spatial.distance as ssd

import gym
from gym import spaces

from traffic.drivers.driver import OneDDriver

class IDMDriver(OneDDriver):
    """
    Intelligent Driver Model
    """

    def __init__(self, sigma=0.0, v_des=10.0, k_spd=1.0, delta=4.0, T=1.5, s_des=1.5, s_min=0., a_max=3.0, d_cmf=2.0,
                 d_max=9.0, v_min=0., min_overlap=0., **kwargs):
        # sigma action noise [m/s^2]
        # v_des desired speed [m/s]
        # k_spd proportional constant for speed tracking when in freeflow [s⁻¹]
        # delta acceleration exponent [-]
        # T desired time headway [s]
        # s_des desire gap [m]
        # s_min minimum acceptable gap [m]
        # a_max maximum acceleration ability [m/s^2]
        # d_cmf comfortable deceleration [m/s^2] (positive)
        # d_max maximum deceleration [m/s^2] (positive)
        self.sigma = sigma
        self.v_des = v_des
        self.k_spd = k_spd
        self.delta = delta
        self.T = T
        self.s_des = s_des
        self.s_min = s_min
        self.a_max = a_max
        self.d_cmf = d_cmf
        self.d_max = d_max
        self.v_min = v_min
        self.min_overlap = min_overlap


        self.front_distance = None
        self.front_speed = None
        self.a = None

        super(IDMDriver, self).__init__(**kwargs)

    def set_v_des(self, v_des):
        self.v_des = v_des

    def observe(self, cars, road):
        min_dist = np.inf
        speed = None
        for car in cars:
            if car is self.car:
                continue
            if (car.position[self.axis0]-self.car.position[self.axis0]) * self.direction > self.s_min:
                # another car is in front of me
                if self.car.get_distance(car,self.axis1) < self.min_overlap:
                    # print('overlapping')
                    # overlapping
                    dist = self.car.get_distance(car,self.axis0) 
                    # if dist > 0.0 and dist < min_dist:
                    if dist < min_dist:
                        min_dist = dist
                        speed = car.velocity[self.axis0] * self.direction

        self.front_distance = min_dist
        self.front_speed = speed

    def get_action(self):
        v_des = self.v_des
        v_ego = self.car.velocity[self.axis0] * self.direction
        # if v_ego < 0.0:
            # v_ego = 0.0
        if self.front_distance < 0.1:
            self.front_distance = 0.1
        if self.front_speed is not None:
            dv = self.front_speed - v_ego
            s_des = self.s_des + v_ego * self.T - v_ego * dv / (
                2 * np.sqrt(self.a_max * self.d_cmf))
            if v_des > 0.0:
                v_ratio = v_ego / v_des
            else:
                v_ratio = 1.0
            self.a = self.a_max * (1.0 - v_ratio**self.delta - (s_des / self.front_distance)**2)
        else:
            dv = v_des - v_ego
            self.a = dv * self.k_spd
        self.a = np.clip(self.a,-self.d_max,self.a_max)
        a = self.a + self.sigma * self.np_random.normal()
        # print('1: ',self.front_distance, self.front_speed,self.a)
        if v_ego + a*self.dt < self.v_min:
            a = (self.v_min-v_ego)/self.dt
            # print("2: ",v_ego, self.v_min, a)

        return a * self.direction


class PDDriver(OneDDriver):
    """
    PD controller driving model
    """

    def __init__(self, sigma=0.0, p_des=0.0, a_max=1.0, k_p=2.0, k_d=2.0, **kwargs):
        self.sigma = sigma
        self.p_des = p_des
        self.a_max = a_max
        self.k_p = k_p
        self.k_d = k_d
        super(PDDriver, self).__init__(**kwargs)

    def set_p_des(self, p_des):
        self.p_des = p_des

    def observe(self, cars, road):
        self.t = self.p_des - self.car.position[self.axis0]
        self.v_t = self.car.velocity[self.axis0]

    def get_action(self):
        self.a = np.clip(self.k_p * self.t - self.k_d * self.v_t, -self.a_max, self.a_max)
        return self.a + self.sigma * self.np_random.normal()

class PDriver(OneDDriver):
    """
    P controller driving model
    """

    def __init__(self, sigma=0.0, v_des=0.0, a_max=1.0, k_p=2.0, **kwargs):
        self.sigma = sigma
        self.v_des = v_des
        self.a_max = a_max
        self.k_p = k_p
        super(PDriver, self).__init__(**kwargs)

    def set_v_des(self, v_des):
        self.v_des = v_des

    def observe(self, cars, road):
        self.t = self.v_des - self.car.velocity[self.axis0]

    def get_action(self):
        self.a = np.clip(self.k_p * self.t, -self.a_max, self.a_max)
        return self.a + self.sigma * self.np_random.normal()

class ConstantDriver(OneDDriver):
    """
    constant controller driving model
    """

    def __init__(self, a=0., sigma=0.0, **kwargs):
        self.sigma = sigma
        self.a = a
        super(ConstantDriver, self).__init__(**kwargs)

    def set_a(self, a):
        self.a = a

    def observe(self, cars, road):
        return

    def get_action(self):
        return self.a + self.sigma * self.np_random.normal()
