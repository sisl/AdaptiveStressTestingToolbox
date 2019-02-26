class StateActionStateNode:
	def __init__(self):
		self.n = 0 #UInt64
		self.r = 0.0 #Float64

class StateActionNode:
	def __init__(self):
		self.s = {} #Dict{State,StateActionStateNode}
		self.n = 0 #UInt64
		self.q = 0.0 #Float64

class StateNode:
	def __init__(self):
		self.a = {} #Dict{Action,StateActionNode}
		self.n = 0 #UInt64
		self.v = 0.0 #float64