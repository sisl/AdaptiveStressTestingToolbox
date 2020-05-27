class StateActionStateNode:
    def __init__(self, n, r):
        self.n = n  # UInt64
        self.r = r  # Float64

    def __init__(self):
        self.n = 0
        self.r = 0.0


class StateActionNode:
    def __init__(self, s, n, q):
        self.s = s  # Dict{State,StateActionStateNode}
        self.n = n  # UInt64
        self.q = q  # Float64

    def __init__(self):
        self.s = {}
        self.n = 0
        self.q = 0.0


class StateNode:
    def __init__(self, a, n, v):
        self.a = a  # Dict{Action,StateActionNode}
        self.n = n  # UInt64
        self.v = v

    def __init__(self):
        self.a = {}
        self.n = 0
        self.v = 0.0
