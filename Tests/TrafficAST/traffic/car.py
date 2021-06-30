import numpy as np
import scipy.spatial.distance as ssd
from shapely.geometry.polygon import Polygon
from shapely.ops import nearest_points

import gym
from gym import spaces
from gym.utils import seeding

from traffic.constants import *

class Car:

    def __init__(self, idx, length, width, color, max_accel, max_speed, max_rotation, expose_level):
        self._idx = idx
        self._length = length
        self._width = width
        self._color = color
        self._arr_color = (0.8, 0.8, 0.8)
        self._max_accel = max_accel
        self._max_speed = max_speed
        self._max_rotation = max_rotation
        self._expose_level = expose_level

        self._position = None
        self._velocity = None
        self._rotation = None
        self._action = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def observation_space(self, cars, road, include_x=False):
        if include_x:
            high = [
                np.inf, np.inf, self._max_speed,
                self._max_speed
            ]
            # px, py,vx,vy
        else:
            high = [np.inf, self._max_speed,
                    self._max_speed]
            # py, vx, vy
        for car in cars:
            if car is self:
                continue
            high += [np.inf, np.inf,
                         self._max_speed, self._max_speed]  
            # relatvie px1,py,vx,vy
        high = np.array(high)
        low = -high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def action_space(self):
        return spaces.Box(low=-self._max_accel, high=self._max_accel, shape=(2,),
                          dtype=np.float32)

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    @property
    def position(self):
        assert self._position is not None
        return self._position

    @property
    def velocity(self):
        assert self._velocity is not None
        return self._velocity

    @property
    def rotation(self):
        assert self._rotation is not None
        return self._rotation

    @property
    def heading(self):
        assert self._velocity is not None
        return np.arctan2(self._velocity[1],self._velocity[0])

    def set_position(self, x_2):
        assert x_2.shape == (2,)
        self._position = np.array(x_2)
        return self._position

    def set_velocity(self, v_2):
        assert v_2.shape == (2,)
        self._velocity = np.array(v_2)
        speed = np.linalg.norm(self._velocity)
        if speed > self._max_speed:
            self._velocity = self._velocity / speed * self._max_speed
        return self._velocity

    def set_rotation(self, rotation):
        self._rotation = rotation
        return self._rotation

    def observe(self, cars, road, include_x=False):
        if include_x:
            ob = self.position
        else:
            ob = np.array(self.position[1])
        ob = np.append(ob, self.velocity)

        for car in cars:
            if car is self:
                continue
            if car._expose_level >= 2:
                ob = np.append(ob, car.position - self.position)
            if car._expose_level >= 4:
                ob = np.append(ob, car.velocity - self.velocity)
        return np.copy(ob)

    def info(self, cars):
        return {}

    def get_vertices(self):
        phi = np.arctan2(self._width,self._length)
        l = np.sqrt(self._width**2+self._length**2)/2.
        x = self._position[0]
        y = self._position[1]
        theta = self._rotation
        cxs = [x+l*np.cos(theta+phi),x+l*np.cos(theta-phi),x-l*np.cos(theta+phi),x-l*np.cos(theta-phi)]
        cys = [y+l*np.sin(theta+phi),y+l*np.sin(theta-phi),y-l*np.sin(theta+phi),y-l*np.sin(theta-phi)]
        return np.array(list(zip(cxs,cys)))

    def get_distance(self,car,axis):
        # positive: end-to-end distance
        # negative: overlap
        vertices1 = self.get_vertices()[:,axis]
        max1 = np.max(vertices1)
        min1 = np.min(vertices1)
        vertices2 = car.get_vertices()[:,axis]
        max2 = np.max(vertices2)
        min2 = np.min(vertices2)
        distance = (np.maximum(max1,max2)-np.minimum(min1,min2))-((max1-min1)+(max2-min2))
        return distance

    def check_collision(self,car):
        polygon1 = Polygon(self.get_vertices())
        polygon2 = Polygon(car.get_vertices())
        return polygon1.intersects(polygon2)   

    def get_closest_points(self, car):
        polygon1 = Polygon(self.get_vertices())
        polygon2 = Polygon(car.get_vertices())
        p1, p2 = nearest_points(polygon1, polygon2)
        return np.array([p1.x,p1.y]), np.array([p2.x,p2.y])

    def setup_render(self, viewer):
        from traffic import rendering
        car_poly = [[-self._length / 2.0, -self._width / 2.0],
                    [self._length / 2.0, -self._width / 2.0],
                    [self._length / 2.0, self._width / 2.0],
                    [-self._length / 2.0, self._width / 2.0]]
        arr_poly = [[0., -self._width / 4.0],
                    [self._length / 2.0, -self._width / 4.0],
                    [self._length / 2.0, self._width / 4.0],
                    [0., self._width / 4.0]]
        self.geom = rendering.make_polygon(car_poly)
        self.xform = rendering.Transform()
        self.geom.set_color(*self._color)
        self.geom.add_attr(self.xform)
        viewer.add_geom(self.geom)

        self.arr_geom = rendering.make_polygon(arr_poly)
        self.arr_xform = rendering.Transform()
        self.arr_geom.set_color(*self._arr_color)
        self.arr_geom.add_attr(self.arr_xform)
        viewer.add_geom(self.arr_geom)

    def update_render(self, camera_center):
        self.xform.set_translation(*(self.position - camera_center))
        self.xform.set_rotation(self._rotation)
        self.geom.set_color(*self._color)
        self.arr_xform.set_translation(*(self.position - camera_center))
        self.arr_xform.set_rotation(self._rotation)
        self.arr_geom.set_color(*self._arr_color)

    def remove_render(self, viewer):
        viewer.geoms.remove(self.geom)
        viewer.geoms.remove(self.arr_geom)

