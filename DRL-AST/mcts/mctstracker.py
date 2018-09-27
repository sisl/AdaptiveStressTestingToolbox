class MCTSTracker:
	def __init__(self, actions, q_values, q_values2_rev):
		self.actions = actions
		self.q_values = q_values
		self.q_values2_rev = q_values2_rev
	def __eq__(self,other):
		return self.actions == other.actions
	def empty(self):
		self.actions.clear()
		self.q_values.clear()
		self.q_values2_rev.clear()
	def __hash__(self):
		return hash(tuple(self.actions))
	def push_action(self, a):
		self.actions.append(a)
	def push_q_value(self, q):
		self.q_values.append(q)
	def push_q_value2(self, q2):
		self.q_values2_rev.append(q2)
	def append_actions(self, a):
		self.actions.extend(a)
	def append_q_values(self, q):
		self.q_values.extend(q)
	def combine_q_values(self):
		if len(self.q_values2_rev) > 0:
			self.q_values.extend(list(reversed(self.q_values2_rev)))
			self.q_values2_rev.clear()
	def get_actions(self):
		return self.actions
	def get_q_values(self):
		return self.q_values

def MCTSTrackerInit():
	return MCTSTracker([],[],[])




