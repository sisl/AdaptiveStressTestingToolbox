import numpy as np
from traffic.actions.action import Action

class XYAccelAction(Action):
    def __init__(self,a_x,a_y):
        self.a_x = a_x
        self.a_y = a_y

    def update(self,car,dt):
        position_old = car.position
        velocity_old = car.velocity
        rotation_old = car.rotation
        accel = np.array([self.a_x, self.a_y])
        velocity = car.set_velocity(car.velocity + accel * dt)
        car.set_position(position_old+0.5*(velocity_old+velocity)*dt)

        # TODO
        rotation = car.heading
        max_ang = car._max_rotation
        if abs(rotation_old) <= max_ang: # rotation close to 0
            if abs(rotation) > max_ang:
                rotation = np.clip(max_ang,-max_ang,max_ang)
        elif (rotation > 0.) and (np.pi-rotation >= max_ang):
            rotation = np.pi - max_ang
        elif (rotation < 0.) and (rotation-(-np.pi) >= max_ang):
            rotation = -np.pi + max_ang
        car.set_rotation(rotation)