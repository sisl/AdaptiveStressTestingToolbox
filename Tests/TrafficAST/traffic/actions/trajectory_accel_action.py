import numpy as np
from traffic.actions.action import Action

class TrajectoryAccelAction:
    def __init__(self,a_s,a_t,trajectory):
        self.a_s = a_s
        self.a_t = a_t
        self.trajectory = trajectory

    def update(self,car,dt):
        s, t, theta, curv = self.trajectory.xy_to_traj(car.position)

        v_x, v_y = car.velocity[0], car.velocity[1]
        v_s = v_x*np.cos(theta) + v_y*np.sin(theta)
        v_t = -v_x*np.sin(theta) + v_y*np.cos(theta)
        phi = np.arctan2(v_t, v_s)

        v_s_new = v_s + self.a_s*dt
        v_t_new = v_t + self.a_t*dt
        velocity_new = car.set_velocity(np.array([v_s_new, v_t_new]))
        v_s_new, v_t_new = velocity_new[0], velocity_new[1]
        s_new = s + 0.5*(v_s+v_s_new)*dt
        t_new = t + 0.5*(v_t+v_t_new)*dt

        x_new, y_new, theta_new, curv_new = self.trajectory.traj_to_xy(np.array([s_new, t_new]))
        v_x_new = v_s_new*np.cos(theta_new) - v_t_new*np.sin(theta_new)
        v_y_new = v_s_new*np.sin(theta_new) + v_t_new*np.cos(theta_new)

        car.set_position(np.array([x_new,y_new]))
        car.set_velocity(np.array([v_x_new, v_y_new]))
        phi_new = np.arctan2(v_t_new, v_s_new)

        max_ang = car._max_rotation #np.pi/18.
        if phi_new > max_ang:
            car.set_rotation(theta_new+max_ang)
        elif phi_new < -max_ang:
            car.set_rotation(theta_new-max_ang)
        else:
            car.set_rotation(theta_new+phi_new)
