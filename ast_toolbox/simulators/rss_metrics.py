from copy import copy
import numpy as np
from enum import Enum


# The state of a car at the specified timestep
class CarState:
    def __init__(self, t, x, y, vx, vy, ax, ay):
        self.t = t
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.ax, self.ay = ax, ay


# Swap the cars and flip the coordinate system to consider the reverse orientation
def swap_and_flip_y(c1, c2):
    c1, c2 = copy(c2), copy(c1)
    c1.y, c1.vy, c1.ay = -c1.y, -c1.vy, -c1.ay
    c2.y, c2.vy, c2.ay = -c2.y, -c2.vy, -c2.ay
    return c1, c2


def swap(c1, c2):
    return copy(c2), copy(c1)


# The params required to define lateral safe distance
class LateralParams:
    def __init__(self, rho, a_lat_max_acc, a_lat_min_brake, mu):
        self.rho = rho  # Response time
        self.a_lat_max_acc = a_lat_max_acc  # Maximum acceleration before response time s
        self.a_lat_min_brake = a_lat_min_brake  # Min required braking speed
        self.mu = mu  # Buffer distance


# The params require to define longitudnal safe distance
class LongitudinalParams:
    def __init__(self, rho, a_max_brake, a_max_acc, a_min_brake1, a_min_brake2, mu):
        self.rho = rho  # Response time
        self.a_max_brake = a_max_brake  # Maximum braking acceleration of leading car
        self.a_max_acc = a_max_acc  # Maximum acceleration before response time ends
        self.a_min_brake1 = a_min_brake1  # Minimum required braking from car 1 (In RSS paper called a_min_brake_correct)
        self.a_min_brake2 = a_min_brake2  # Minimum required braking from car 2 (In RSS paper called a_min_brake)
        self.mu = mu  # Buffer distance


# Types of responses
# Note that Proper + ImproperX = ImproperX
# and ImproperLongitudinal + ImproperLateral = ImproperBoth

class Response(Enum):
    Proper = 0
    ImproperLongitudinal = 1
    ImproperLateral = 2
    ImproperBoth = 3

# Dangerous descriptors
class Dangerous(Enum):
    Safe = 0
    LongitudinallyDangerous = 1
    LaterallyDangerous = 2
    Dangerous = 3


# Take in two trajectories (Car1 and car2)
# Return an array of ints characterizing the danger level at each timestep
def characterize_danger(traj1, traj2, lat_params, long_params):
    assert len(traj1) == len(traj2)
    n = len(traj1)
    d = np.zeros(n, dtype=Dangerous)
    for i in range(n):
        assert traj1[i].t == traj2[i].t
        d[i] = is_dangerous(traj1[i], traj2[i], lat_params, long_params)
    return d


# Take in two trajectories (car1 and car2)
# Return an array of RESPONSE characterizing the behavior at each timestep for each car
def characterize_response(traj1, traj2, lat_params, long_params):
    assert len(traj1) == len(traj2)
    n = len(traj1)
    resp1, resp2 = np.zeros(n, dtype=Response), np.zeros(n, dtype=Response)
    reasons1, reasons2 = ["undef" for i in range(n)], ["undef" for i in range(n)]
    tb_lat, tb_long = find_blame_times(traj1, traj2, lat_params, long_params)
    for i in range(n):
        assert traj1[i].t == traj2[i].t
        resp1[i], reasons1[i], resp2[i], reasons2[i] = \
            is_proper(tb_lat, tb_long, traj1[i], traj2[i], lat_params, long_params)
    return resp1, reasons1, resp2, reasons2


# Find the blame time in the trajectory
def find_blame_times(traj1, traj2, lat_params, long_params):
    tb_lat, tb_long = float('inf'), float('inf')
    for i in range(len(traj1)):
        c1, c2 = traj1[i], traj2[i]
        t = c1.t
        if is_long_dangerous(c1, c2, long_params) and tb_long == float('inf'):
            tb_long = t

        if is_lat_dangerous(c1, c2, lat_params) and tb_lat == float('inf'):
            tb_lat = t
    return tb_lat, tb_long


# Get whether the response is proper for each car at the specified state
def is_proper(tb_lat, tb_long, c1, c2, lat_params, long_params):
    tb = max(tb_lat, tb_long)
    prefix = ""
    flipped = False
    if tb == tb_lat:
        if c1.x > c2.x:
            prefix = "Note: Swapped c1 (x=" + str(c1.x) + ") and c2 (x=" + str(c2.x) + \
                     ") so that c1 is on the left of c2. "
            c1, c2 = swap(c1, c2)
            flipped = True

        lat1, reason1, lat2, reason2 = is_proper_lat(tb, c1, c2, lat_params, prefix)
        resp1 = Response.Proper if lat1 else Response.ImproperLateral
        resp2 = Response.Proper if lat2 else Response.ImproperLateral
    else:
        if c1.vy <=0:
            prefix = "Note: Swapped (and flipped y dimension) c1 (vy=" + str(c1.vy) + ") and c2 (vy=" + str(c2.vy) + \
                     ") so that c1 has positive velocity. "
            c1, c2 = swap_and_flip_y(c1, c2)
            flipped = True

        long1, reason1, long2, reason2 = is_proper_long(tb, c1, c2, long_params, prefix)
        resp1 = Response.Proper if long1 else Response.ImproperLongitudinal
        resp2 = Response.Proper if long2 else Response.ImproperLongitudinal
    if flipped:
        return resp2, reason2, resp1, reason1
    else:
        return resp1, reason1, resp2, reason2


# Returns a tuple containing whether each car had a proper lateral response at the given timestep
# c1 is to the left of c2
def is_proper_lat(tb, c1, c2, p, prefix = ""):
    t = c1.t
    # Before the blame time any response is appropriate
    if t < tb:
        reason = prefix + "Current t=" + str(t) + ", is before blame time, tb=" + str(tb)
        return True, reason, True, reason
    elif t >= tb and t <= tb+p.rho:
        # Both cars can do any lateral action as long as |a| < a_lat_max_acc
        reason1 =  prefix + "After blame time before response: c1.ax=" + str(c1.ax) +  " and p.a_lat_max_acc=" + \
                   str(p.a_lat_max_acc) + ". A proper response has abs(c1.ax) < p.a_lat_max_acc"
        reason2 = prefix + "After blame time before response: c2.ax=" + str(c2.ax) + " and p.a_lat_max_acc=" + \
                  str(p.a_lat_max_acc) + ". A proper response has abs(c2.ax) < p.a_lat_max_acc"
        return abs(c1.ax) < p.a_lat_max_acc, reason1, abs(c2.ax) < p.a_lat_max_acc, reason2
    else:
        is_proper_c1 = None
        reason1 = "Error"
        if c1.vx > 0:
            # c1 must apply lateral acceleration of at most -a_lat_min_brake
            reason1 = prefix + "In response time (with c1.vx=" + str(c1.vx) + " > 0): c1.ax=" + str(c1.ax) + \
                      " and -p.a_lat_min_brake=" + str(-p.a_lat_max_acc) + \
                      ". A proper response requires c1.ax <= -p.a_lat_min_brake"
            is_proper_c1 = c1.ax <= -p.a_lat_min_brake
        else:
            # After a lateral velocity of 0 has been reached then c1 can have any non-positive lateral velocity
            # (acceleration?)
            reason1 = prefix + "In response time (with c1.vx=" + str(c1.vx) + \
                      "<=0). A proper response requires any non-positive lateral velocity"
            is_proper_c1 = c1.vx <= 0

        is_proper_c2 = None
        reason2 = "Error"
        if c2.vx < 0:
            # c2 must apply a lateral accerlation of at least a_lat_min_brake
            reason2 = prefix +  "In response time (with c2.vx=" + str(c2.vx) + " < 0): c2.ax=" + str(c2.ax) + \
                      " and p.a_lat_min_brake=" + str(p.a_lat_max_acc) + \
                      ". A proper response requires c2.ax >= p.a_lat_min_brake"
            is_proper_c2 = c2.ax >= p.a_lat_min_brake
        else:
            # c2 cn have any non-negative lateral velocity (acceleration?)
            reason2 = prefix + "In response time (with c2.vx=" + str(c2.vx) + \
                      ">=0). A proper response requires any non-negative lateral velocity"
            is_proper_c2 = c2.vx >= 0

        return is_proper_c1, reason1, is_proper_c2, reason2


# Returns a tuple containing whether each car had a proper longitudinal response at the given timestep
def is_proper_long(tb, c1, c2, p, prefix=""):
    # Enforce the condition of moving in the appropriate directions
    t = c1.t
    # Before the blame time any response is appropriate
    if t < tb:
        reason = prefix + "Current t=" + str(t) + ", is before blame time, tb=" + str(tb)
        return True, reason, True, reason

    # Get the current separation and min safe distance
    d, dmin = abs(c2.y - c1.y), longitudinal_dmin(c1.vy, c2.vy, p)

    if same_dir(c1.vy, c2.vy):
        prefix = prefix + "Same direction (c1.vy=" + str(c1.vy) + ", c2.vy=" + str(c2.vy) + ")"
        if t >= tb and t <= tb+p.rho:
            # c1's acceleration must be at most a_max_acc
            # c2's acceleration must be at least -a_max_brake
            reason1 = prefix + "After blame time before response: c1.ay=" + str(c1.ay) + \
                      ", p.a_max_acc=" +  str(p.a_max_acc) + ". A proper response requires c1.ay <= p.a_max_acc"
            reason2 = prefix + "After blame time before response: c2.ay=" + str(c2.ay) + \
                      ", -p.a_max_brake=" + str(-p.a_max_brake) + ". A proper response requires c2.ay >= -p.a_max_brake"
            return c1.ay <= p.a_max_acc, reason1, c2.ay >= -p.a_max_brake, reason2
        elif is_dangerous_d(d, dmin) and t > tb + p.rho:
            # c1's acceleration must be at most -a_min_brake
            # c2's acceleration must be at least -a_max_brake'
            reason1 = prefix + "In response time (still dangerous): c1.ay=" + str(c1.ay) + \
                      ", -p.a_min_brake1=" + str(-p.a_min_brake1) + \
                      ". A proper response requires c1.ay <= -p.a_min_brake1"
            reason2 = prefix + "In response time (still dangerous): c2.ay=" + str(c2.ay) + \
                      ", -p.a_max_brake=" + str(-p.a_max_brake) + ". A proper response requires c2.ay >= -p.a_max_brake"
            return c1.ay <= -p.a_min_brake1, reason1, c2.ay >= -p.a_max_brake, reason2
        elif not is_dangerous_d(d, dmin) and t > tb + p.rho:
            # c1's acceleration must be non-positive
            # c2's acceleration must be non-negative
            reason1 = prefix + "In response time (no longer dangerous): c1.ay=" + str(c1.ay) + \
                      ". A proper response requires c1.ay be non-positive"
            reason2 = prefix + "In response time (no longer dangerous): c2.ay=" +  str(c2.ay) + \
                      ". A proper response requires c2.ay be non-negative"
            return c1.ay <= 0, reason1, c2.ay >= 0, reason2
        else:
            raise Exception("Invalid state")
    #elif opp_dir(c1.vy, c2.vy):
    else:
        prefix = prefix  + "Opposite direction (c1.vy=" + str(c1.vy) + ", c2.vy=" +  str(c2.vy) + ")"
        if t >= tb and t <= tb+p.rho:
            # c1's acceleration must be at most a_max_acc
            # c2's accerlation must be at least -a_max_acc
            reason1 = prefix +  "After blame time before response: c1.ay=" + str(c1.ay) + ", p.a_max_acc=" + \
                      str(p.a_max_acc) + ". A proper response requires c1.ay <= p.a_max_acc"
            reason2 = prefix + "After blame time before response: c2.ay=" + str(c2.ay) + ", -p.a_max_acc=" + \
                      str(-p.a_max_acc) + ". A proper response requires c2.ay >= -p.a_max_acc"
            return c1.ay <= p.a_max_acc, reason1, c2.ay >= -p.a_max_acc, reason2

        is_proper_c1 = None
        reason1 = "Error"
        if c1.vy > 0 and t > tb + p.rho:
            # c1's acceleration must be at most -a_min_brake1
            reason1 = prefix + "In response time (with c1.vy = " + str(c1.vy) + " > 0): c1.ay=" + str(c1.ay) + \
                      ", -p.a_min_brake1=" + str(-p.a_min_brake1) +  \
                      ". A proper response requires c1.ay <= -p.a_min_brake1"
            is_proper_c1 = c1.ay <= -p.a_min_brake1
        elif c1.vy <= 0 and t > tb + p.rho:
            # c1's acceleration can be any non-positive acceleration
            reason1 = prefix + "In response time (with c1.vy = " + str(c1.vy) + " <= 0): c1.ay=" + str(c1.ay) + \
                      ". A proper response requires that c1 have any non-positive acceleration"
            is_proper_c1 = c1.ay <= 0
        else:
            raise Exception("Invalid state")

        if c2.vy < 0 and t > tb + p.rho:
            # c2's acceleration must be at most -a_min_brake2
            reason2 = prefix + "In response time (with c2.vy = " + str(c2.vy) + " < 0): c2.ay=" + str(c2.ay) + \
                      ", p.a_min_brake2=" + str(p.a_min_brake2) + ". A proper response requires c2.ay >= p.a_min_brake2"
            is_proper_c2 = c2.ay >= p.a_min_brake2
        elif c2.vy >= 0 and t > tb + p.rho:
            # c2's acceleration can be any non-nonegative acceleration
            reason2 = prefix + "In response time (with c2.vy = " + str(c2.vy) + " >= 0): c2.ay=" + str(c2.ay) + \
                      ". A proper response requires that c2 have any non-negative acceleration"
            is_proper_c2 = c2.ay >= 0
        else:
            raise Exception("Invalid State")

        return is_proper_c1, reason1, is_proper_c2, reason2
    # else:
    #     raise Exception("Improper velocity configuration")


# returns true if the vehicles are moving in the same direction (positive) or one is stopped
def same_dir(v1, v2):
    return v1 >= 0 and v2 >= 0


# returns true if the vehicles are moving in the opposite direction or if one is stopped
def opp_dir(v1, v2):
    return v1 >=0 and v2 <= 0


# Given the car distance d, and the min safe distance dmin, return if d is safe
def is_dangerous_d(d, dmin):
    return d <= dmin


# Get the danger level of a state in both the longitudinal and lateral directions
def is_dangerous(c1, c2, lat_params, long_params):
    return Dangerous(is_long_dangerous(c1, c2, long_params)*Dangerous.LongitudinallyDangerous.value + \
           is_lat_dangerous(c1, c2, lat_params)*Dangerous.LaterallyDangerous.value)


# Check if a longitudinal configuration is dangerous
def is_long_dangerous(c1, c2, long_parms):
    d = abs(c2.y - c1.y)
    dmin = longitudinal_dmin(c1.vy, c2.vy, long_parms)
    return is_dangerous_d(d, dmin)


# Check if a lateral configuration is dangerous
def is_lat_dangerous(c1, c2, lat_params):
    d = abs(c2.x - c1.x)
    # Enforce the condition that c1 is to the left of c2
    if c1.x < c2.x:
        dmin = lateral_dmin(c1.vx, c2.vx, lat_params)
    else:
        dmin = lateral_dmin(c2.vx, c1.vx, lat_params)
    return is_dangerous_d(d, dmin)


# Get the minimum longitudinal difference, taking into account the relative direction of the cars.
def longitudinal_dmin(v1, v2, long_params):
    if same_dir(v1, v2):
        return longitudinal_dmin_same_dir(v1, v2, long_params)
    elif opp_dir(v1, v2):
        return longitudinal_dmin_opp_dir(v1, v2, long_params)
    elif v1 <= 0 and v2 >= 0:
        return longitudinal_dmin_same_dir(v1, v2, long_params)
    else:
        # The rear car is in reverse so now its considered the front car
        return longitudinal_dmin(-v2, -v1, long_params)


# Get the minimum longitudnal safe distance when cars are moving in the same direction given the velocities and
# the stopping params
def longitudinal_dmin_same_dir(v1, v2, p):
    dmin = p.mu + v1*p.rho + 0.5*p.a_max_acc*p.rho**2 + (v1 + p.rho*p.a_max_acc)**2/(2*p.a_min_brake1) - \
           v2**2/(2*p.a_max_brake)
    dmin = max(dmin, 0)
    return dmin


# Get the minimum longitudnal safe distance when cars are moving in opposite directions given the velocities and the
# stopping params
def longitudinal_dmin_opp_dir(v1, v2,  p):
    v1rho = v1 + p.rho*p.a_max_acc
    v2rho = abs(v2) + p.rho*p.a_max_acc
    dmin = p.mu + (v1 + v1rho)*p.rho/2 + v1rho**2/(2*p.a_min_brake1) + (abs(v2) + v2rho)*p.rho/2 + \
           v2rho**2/(2*p.a_min_brake2)
    return dmin


# Get the minimum lateral safe distance based on car velocities and stopping parameters
# Note that c1 is to the left of c2
def lateral_dmin(v1, v2, p):
    v1rho = v1 + p.rho*p.a_lat_max_acc
    v2rho = v2 - p.rho*p.a_lat_max_acc
    dmin = (v1 + v1rho)*p.rho/2 + v1rho**2 / (2*p.a_lat_min_brake) - ((v2 + v2rho)*p.rho/2 -
                                                                     v2rho**2/(2*p.a_lat_min_brake))
    dmin = max(0,dmin) + p.mu
    return dmin