import random
import itertools
import numpy as np
from gym import spaces
from scipy.stats import norm

from traffic.traffic_env import TrafficEnv
from traffic.road import Road, RoadSegment
from traffic.car import Car
from traffic.drivers.driver import Driver, XYSeperateDriver
from traffic.drivers.oned_drivers import IDMDriver, PDDriver, PDriver
from traffic.actions.xy_accel_action import XYAccelAction
from traffic.constants import *

LANE_WIDTH = 4.
LANE_LENGTH = 80.

def which_lane(car):
    if car.position[1] <= LANE_WIDTH:
        return 0
    else:
        return 1

class EnvDriver(XYSeperateDriver):
    def __init__(self, 
                target_lane,
                min_front_gap,
                x_des,
                x_sigma, y_sigma,
                **kwargs):
        self.target_lane = target_lane
        self.min_front_gap = min_front_gap
        self.x_des = x_des
        x_driver =  PDDriver(sigma=x_sigma, p_des=0., a_max=1.0, axis=0, k_p=2.0, k_d=5.0, **kwargs)
        y_driver =  PDDriver(sigma=y_sigma, p_des=0., a_max=1.0, axis=1, k_p=2.0, k_d=5.0, **kwargs)
        super(EnvDriver, self).__init__(x_driver,y_driver,**kwargs)
        self.ast_action = 0.

    def observe(self, cars, road):
        ego_lane_id = which_lane(self.car)
        x, y = self.car.position
        min_front_distance = np.inf

        for car in cars:
            if car is self.car:
                continue
            else:
                lane_id = which_lane(car)
                if (lane_id == ego_lane_id) and (car.position[0] > x):
                    front_distance = car.position[0] - x
                    if front_distance < min_front_distance:
                        min_front_distance = front_distance

        self.x_driver.p_des = np.minimum(self.x_des+x, 
                                        self.x_des+x+min_front_distance-self.min_front_gap)
        self.y_driver.p_des = ego_lane_id*LANE_WIDTH+0.5*LANE_WIDTH

        self.x_driver.observe(cars, road)
        self.y_driver.observe(cars, road)

    def apply_ast_action(self, ast_action):
        self.ast_action = ast_action

    def get_action(self):
        a_x = self.x_driver.get_action() + self.ast_action
        a_y = self.y_driver.get_action() 
        return XYAccelAction(a_x, a_y)

    def setup_render(self, viewer):
        self.car._color = GREEN_COLORS[0]

    def update_render(self, camera_center):
        pass

class EgoDriver(XYSeperateDriver):
    def __init__(self, 
                x_sigma, y_sigma,
                **kwargs):

        x_driver = PDDriver(sigma=x_sigma, p_des=0., a_max=1.0, axis=0, k_p=2.0, k_d=5.0, **kwargs)
        y_driver =  PDDriver(sigma=y_sigma,p_des=0., a_max=1.0, axis=1, k_p=2.0, k_d=5.0, **kwargs)
        super(EgoDriver, self).__init__(x_driver,y_driver,**kwargs)

    def apply_action(self, action):
        self.x_driver.p_des = self.car.position[0] + action[0]
        self.y_driver.p_des = action[1]*LANE_WIDTH + 0.5*LANE_WIDTH

    def observe(self, cars, road):
        self.x_driver.observe(cars, road)
        self.y_driver.observe(cars, road)

    def setup_render(self, viewer):
        self.car._color = BLUE_COLORS[0]

class HighWay(TrafficEnv):
    def __init__(self,
                 obs_noise=0.,
                 x_actions=[-1., 0., 1.],
                 y_actions=[0,1],
                 driver_sigma = 0.,
                 v0_sigma = 0.,
                 x_des_sigma = 0.,
                 y_cost=0.01,
                 collision_cost=2.,
                 survive_reward=0.01,
                 goal_reward=2.,
                 init_ast_action_scale=0.7,
                 ast_action_scale=1.0,
                 road=Road([RoadSegment([(-LANE_LENGTH/2.,0.),(LANE_LENGTH/2.,0.),
                                (LANE_LENGTH/2.,2*LANE_WIDTH),(-LANE_LENGTH/2.,2*LANE_WIDTH)])]),
                 num_updates=1,
                 dt=0.1,
                 **kwargs):

        self.obs_noise = obs_noise
        self.x_actions = x_actions
        self.y_actions = y_actions
        # we use target value instead of target change so system is Markovian
        self.rl_actions = list(itertools.product(x_actions,y_actions))
        self.num_updates = num_updates

        self.y_cost = y_cost
        self.collision_cost = collision_cost
        self.survive_reward = survive_reward
        self.goal_reward = goal_reward

        self.init_ast_action_scale = init_ast_action_scale
        self.ast_action_scale = ast_action_scale

        self.bound = 30.
        self.lane_start = 0
        self.lane_goal = 1
        self.max_veh_num = 5

        self._collision = False
        self._goal = False
        self._terminal = False
        self.ast_action = None
        self.log_trajectory_pdf = 0.0

        self.car_length = 5.0
        self.car_width = 2.0
        self.car_max_accel = 5.0
        self.car_max_speed = 2.0
        self.car_max_rotation = 0.
        self.car_expose_level = 4
        self.min_front_gap = 1.1*self.car_length
        self.driver_sigma = driver_sigma
        self.v0_sigma = v0_sigma
        self.x_des_sigma = x_des_sigma

        super(HighWay, self).__init__(
            road=road,
            cars=[],
            drivers=[],
            dt=dt,
            **kwargs,)

    def update(self, action):
        rl_action = self.rl_actions[action]
        self._drivers[0].apply_action(rl_action)
        for i,driver in enumerate(self._drivers[1:]):
            driver.apply_ast_action(0.)

        # recorder intentios at the begining
        for driver in self._drivers:
            driver.observe(self._cars, self._road)

        self._goal = False
        self._collision = False
        self._terminal = False
        for i_update in range(self.num_updates):
            if i_update > 0:
                for driver in self._drivers:
                    driver.observe(self._cars, self._road)
            self._actions = [driver.get_action() for driver in self._drivers]
            [action.update(car, self.dt) for (car, action) in zip(self._cars, self._actions)]

            ego_car = self._cars[0]
            for car in self._cars[1:]:
                if ego_car.check_collision(car):
                    self._collision = True
                    return

            if (ego_car.position[1] > 1.4*LANE_WIDTH) \
                and (ego_car.position[1] < 1.6*LANE_WIDTH):
                self._goal = True
                return

            if (ego_car.position[0] >= self.bound) \
                or (ego_car.position[0] <= -self.bound):
                self._terminal = True
                return

    def is_terminal(self):
        return (self._collision or self._goal or self._terminal)

    def get_info(self):
        info = {}

        if self._collision:
            info['event']='collision'
        elif self._goal:
            info['event']='goal'
        elif self._terminal:
            info['event']='terminal'
        else:
            info['event']='nothing'

        return info

    def observe(self):
        obs = np.zeros(int(4*self.max_veh_num))
        for car in self._cars:
            i = int(car._idx*4)
            obs[i] = car.position[0]/self.bound + np.random.uniform(-1.,1.)*self.obs_noise
            obs[i+1] = car.position[1]/LANE_WIDTH + np.random.uniform(-1.,1.)*self.obs_noise
            obs[i+2:i+4] = car.velocity/self.car_max_speed + np.random.uniform(-1.,1.,2)*self.obs_noise

        obs = np.copy(obs)
        return obs

    @property
    def observation_space(self):
        low = -np.ones(int(4*self.max_veh_num))
        high = np.ones(int(4*self.max_veh_num))
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self):
        return spaces.Discrete(len(self.rl_actions))

    def get_reward(self):
        reward = 0.
        action = self._actions[0]
        ego_car = self._cars[0]
        x, y = ego_car.position[0], ego_car.position[1]
        v_x, v_y = ego_car.velocity[0], ego_car.velocity[1]

        y_start = self.lane_start*LANE_WIDTH+0.5*LANE_WIDTH
        y_goal = self.lane_goal*LANE_WIDTH+0.5*LANE_WIDTH
        y_cost = -np.abs(y_goal-y)/np.abs(y_goal-y_start)
        reward += self.y_cost*y_cost

        if self._collision:
            reward -= self.collision_cost
        elif self._goal:
            reward += self.goal_reward
        else:
            reward += self.survive_reward
        return reward

    def add_car(self, idx, x, y, vx, vy, x_des, target_lane, theta, x_sigma, y_sigma):
        car = Car(idx=idx, length=self.car_length, width=self.car_width, color=random.choice(RED_COLORS),
                          max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                          max_rotation=self.car_max_rotation,
                          expose_level=self.car_expose_level)
        driver = EnvDriver(target_lane=target_lane, 
                            min_front_gap=self.min_front_gap,
                            x_des=x_des,
                            x_sigma=x_sigma, y_sigma=y_sigma,
                            idx=idx, car=car, dt=self.dt
                            ) 
        car.set_position(np.array([x, y]))
        car.set_velocity(np.array([vx, vy]))
        car.set_rotation(theta)

        self._cars.append(car)
        self._drivers.append(driver)
        # driver.observe(self._cars, self._road)
        return car, driver

    def _reset(self):
        self._collision = False
        self._goal = False
        self._terminal = False
        self.ast_action = None
        self.log_trajectory_pdf = 0.0

        self._cars, self._drivers = [], []
        x_0 = 0.
        y_0 = 0.5*LANE_WIDTH
        car = Car(idx=0, length=self.car_length, width=self.car_width, color=random.choice(BLUE_COLORS),
                          max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                          max_rotation=self.car_max_rotation,
                          expose_level=self.car_expose_level)
        driver = EgoDriver(x_sigma=0., y_sigma=0.,
                            idx=0,car=car,dt=self.dt)
        car.set_position(np.array([x_0, y_0]))
        car.set_velocity(np.array([0., 0.]))
        car.set_rotation(0.)
        self._cars.append(car)
        self._drivers.append(driver)
        # randomly generate surrounding cars and drivers   
        x = np.random.uniform(-self.bound, -1.5*self.car_length)
        y = 0.5*LANE_WIDTH
        self.add_car(idx=1, x=x, y=y, vx=self.v0_sigma*np.random.normal(), vy=0., 
                        x_des=self.x_des_sigma*np.random.normal(),
                        target_lane=0, theta=0., 
                        x_sigma=self.driver_sigma, y_sigma=0.)

        x = np.random.uniform(1.5*self.car_length, self.bound)
        y = 0.5*LANE_WIDTH
        self.add_car(idx=2, x=x, y=y, vx=self.v0_sigma*np.random.normal(), vy=0., 
                        x_des=self.x_des_sigma*np.random.normal(),
                        target_lane=0, theta=0., 
                        x_sigma=self.driver_sigma, y_sigma=0.)

        x = np.random.uniform(-self.bound, -0.5*self.car_length)
        y = 1.5*LANE_WIDTH
        self.add_car(idx=3, x=x, y=y, vx=self.v0_sigma*np.random.normal(), vy=0., 
                        x_des=self.x_des_sigma*np.random.normal(), 
                        target_lane=0, theta=0., 
                        x_sigma=self.driver_sigma, y_sigma=0.)

        x = np.random.uniform(0.5*self.car_length, self.bound)
        y = 1.5*LANE_WIDTH
        self.add_car(idx=4, x=x, y=y, vx=self.v0_sigma*np.random.normal(), vy=0., 
                        x_des=self.x_des_sigma*np.random.normal(), 
                        target_lane=0, theta=0., 
                        x_sigma=self.driver_sigma, y_sigma=0.)

        return None

    def setup_viewer(self):
        from traffic import rendering
        self.viewer = rendering.Viewer(1200, 800)
        self.viewer.set_bounds(-40.0, 40.0, -20.0, 20.0)

    def get_camera_center(self):
        return np.array([0.,6.0])

    def update_extra_render(self, extra_input):
        # lane marker line
        start = np.array([-0.5*LANE_LENGTH,LANE_WIDTH]) - self.get_camera_center()
        end = np.array([0.5*LANE_LENGTH,LANE_WIDTH]) - self.get_camera_center()
        attrs = {"color":(1.,1.,1.),"linewidth":4.}
        self.viewer.draw_line(start, end, **attrs)

        start = np.array([-self.bound,0.]) - self.get_camera_center()
        end = np.array([-self.bound,2.*LANE_WIDTH]) - self.get_camera_center()
        attrs = {"color":(1.,0.,0.),"linewidth":4.}
        self.viewer.draw_line(start, end, **attrs)
        start = np.array([self.bound,0.]) - self.get_camera_center()
        end = np.array([self.bound,2.*LANE_WIDTH]) - self.get_camera_center()
        attrs = {"color":(1.,0.,0.),"linewidth":4.}
        self.viewer.draw_line(start, end, **attrs)

    @property
    def ast_observation_space(self):
        low = -np.ones(int(4*self.max_veh_num))
        high = np.ones(int(4*self.max_veh_num))
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def ast_action_space(self):
        high = np.ones(1*(self.max_veh_num-1))
        return spaces.Box(-high, high, dtype=np.float32)

    def get_observation(self):
        return self.observe()

    def ast_get_observation(self):
        return self.observe()

    def ast_reset(self, s_0=None):
        # s_0 not used

        self.ast_init_step = True

        self._collision = False
        self._goal = False
        self._terminal = False
        self.ast_action = None
        self.log_trajectory_pdf = 0.0
        self._cars, self._drivers = [], []

        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.close()
            self.viewer = None

        return np.zeros(self.ast_observation_space.low.size), \
            np.zeros(self.observation_space.low.size)

    def ast_step(self, action, ast_action):
        ast_action = np.clip(ast_action, -1., 1.)
        self.ast_action = ast_action
        if self.ast_init_step:
            ast_action = ast_action * self.init_ast_action_scale
            # the first step is for the ast to set initial positions
            # and the actual reset for the env
            self.ast_init_step = False

            self._collision = False
            self._goal = False
            self._terminal = False
            self.log_trajectory_pdf = 0.0

            self._cars, self._drivers = [], []
            x_0 = 0.
            y_0 = 0.5*LANE_WIDTH
            car = Car(idx=0, length=self.car_length, width=self.car_width, color=random.choice(BLUE_COLORS),
                              max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                              max_rotation=self.car_max_rotation,
                              expose_level=self.car_expose_level)
            driver = EgoDriver(x_sigma=0., y_sigma=0.,
                                idx=0,car=car,dt=self.dt)
            car.set_position(np.array([x_0, y_0]))
            car.set_velocity(np.array([0., 0.]))
            car.set_rotation(0.)
            self._cars.append(car)
            self._drivers.append(driver)
            # randomly generate surrounding cars and drivers   
            x = ((-self.bound)+(-1.5*self.car_length))/2. \
                + ast_action[0]*((-1.5*self.car_length)-(-self.bound))/2.
            y = 0.5*LANE_WIDTH
            # idx, x, y, vx, vy, target_lane, theta, x_sigma, y_sigma
            self.add_car(idx=1, x=x, y=y, vx=self.v0_sigma*np.random.normal(), vy=0., 
                        x_des=self.x_des_sigma*np.random.normal(),
                        target_lane=0, theta=0., 
                        x_sigma=self.driver_sigma, y_sigma=0.)

            x = ((1.5*self.car_length)+(self.bound))/2. \
                + ast_action[1]*((self.bound)-(1.5*self.car_length))/2.
            y = 0.5*LANE_WIDTH
            self.add_car(idx=2, x=x, y=y, vx=self.v0_sigma*np.random.normal(), vy=0., 
                        x_des=self.x_des_sigma*np.random.normal(),
                        target_lane=0, theta=0., 
                        x_sigma=self.driver_sigma, y_sigma=0.)

            x = ((-self.bound)+(-0.5*self.car_length))/2. \
                + ast_action[2]*((-0.5*self.car_length)-(-self.bound))/2.
            y = 1.5*LANE_WIDTH
            self.add_car(idx=3, x=x, y=y, vx=self.v0_sigma*np.random.normal(), vy=0., 
                        x_des=self.x_des_sigma*np.random.normal(),
                        target_lane=1, theta=0., 
                        x_sigma=self.driver_sigma, y_sigma=0.)

            x = ((0.5*self.car_length)+(self.bound))/2. \
                + ast_action[3]*((self.bound)-(0.5*self.car_length))/2.
            y = 1.5*LANE_WIDTH
            self.add_car(idx=4, x=x, y=y, vx=self.v0_sigma*np.random.normal(), vy=0., 
                        x_des=self.x_des_sigma*np.random.normal(),
                        target_lane=1, theta=0., 
                        x_sigma=self.driver_sigma, y_sigma=0.)

        else:
            ast_action = ast_action * self.ast_action_scale
            rl_action = self.rl_actions[action]
            self._drivers[0].apply_action(rl_action)
            for i,driver in enumerate(self._drivers[1:]):
                driver.apply_ast_action(ast_action[i])

            # recorder intentios at the begining
            for driver in self._drivers:
                driver.observe(self._cars, self._road)

            self._goal = False
            self._collision = False
            self._terminal = False
            for i_update in range(self.num_updates):
                if i_update > 0:
                    for driver in self._drivers:
                        driver.observe(self._cars, self._road)
                self._actions = [driver.get_action() for driver in self._drivers]
                [action.update(car, self.dt) for (car, action) in zip(self._cars, self._actions)]

                ego_car = self._cars[0]
                for car in self._cars[1:]:
                    if ego_car.check_collision(car):
                        self._collision = True
                        break

                if (ego_car.position[1] > 1.4*LANE_WIDTH) \
                    and (ego_car.position[1] < 1.6*LANE_WIDTH):
                    self._goal = True
                    break

                if (ego_car.position[0] >= self.bound) \
                    or (ego_car.position[0] <= -self.bound):
                    self._terminal = True
                    break

        return self.ast_get_observation(), self.get_observation(), self.is_terminal()

    def ast_is_goal(self):
        return self._collision

    def ast_get_reward_info(self):
        is_goal = self.ast_is_goal()
        if is_goal:
            dist = 0.0
        else:
            ego_car_x = self._cars[0].position[0]
            ego_lane_id = which_lane(self._cars[0])
            min_distance = np.inf
            for car in self._cars[1:]:
                lane_id = which_lane(car)
                if lane_id == ego_lane_id:
                    distance = np.abs(car.position[0] - ego_car_x)
                    if distance < min_distance:
                        min_distance = distance
            dist = min_distance
        # prob = -1/(self.wind_force_mag*self.wind_force_mag)*np.abs(self.ast_action)+1/self.wind_force_mag
        prob = np.product(norm.pdf(self.ast_action))
        self.log_trajectory_pdf += np.log(prob)
        # print("prob: ",prob)
        # print("log_t_pdf: ",self.log_trajectory_pdf)
        return dict(
            is_goal = is_goal,
            dist = dist,
            prob = prob,
            log_trajectory_pdf = self.log_trajectory_pdf,
            )

if __name__ == '__main__':
    import time
    import pdb
    env = HighWay(num_updates=1, driver_sigma=0.1, 
                    obs_noise=0.,
                    )
    obs = env.reset()
    img = env.render()
    done = False
    maximum_step = 100
    t = 0
    cr = 0.

    while True:  #not done: 
        action = input("Action in {}\n".format(env.rl_actions))
        action = int(action)
        while action < 0:
            t = 0
            cr = 0.
            env.reset()
            env.render()
            action = input("Action\n")
            action = int(action)
        t += 1
        obs, reward, done, info = env.step(action)
        print('t: ', t)
        print('action: ',action)
        print('obs: ', obs)
        print('reward: ', reward)
        print('info: ', info)
        cr += reward
        env.render()
        # pdb.set_trace()
        time.sleep(0.1)
        if (t > maximum_step) or done:
            print('cr: ',cr)
            pdb.set_trace()
            # if env._collision or env._outroad:
            #     pdb.set_trace()
            t = 0
            cr = 0.
            env.reset()
            env.render()
    env.close()
