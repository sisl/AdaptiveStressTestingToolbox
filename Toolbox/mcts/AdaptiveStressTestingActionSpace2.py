import copy
import mcts.MDP as MDP
import mcts.MCTSdpw as MCTSdpw
import numpy as np

class ASTParams:
	def __init__(self,max_steps):
		self.max_steps = max_steps

# def ASTParamsInit():
# 	return ASTParams(0,DEFAULT_RSGLENGTH,0,None)

class AdaptiveStressTestAS:
	def __init__(self,p,env):
		self.params = p
		self.env = env
		self.sim_hash = hash(0)

		self.transition_model = self.transition_model()

		self.step_count = 0
		self._isterminal = False
		self._reward = 0.0

	def reset_setp_count(self):
		self.step_count = 0
	def initialize(self):
		self._isterminal = False
		self._reward = 0.0
		return self.env.reset()
	def update(self,action):
		self.step_count += 1
		obs, reward, done, info = self.env.step(action.get())
		self._isterminal = done
		self._reward = reward
		return obs, reward, done, info
	def isterminal(self):
		return self._isterminal
	def get_reward(self):
		return self._reward
	def transition_model(self):
		return transition_model_action_space(self)
	def random_action(self):
		return ASTAction(self.env.action_space.sample())
	def explore_action(self,s,tree):
		return ASTAction(self.env.action_space.sample())

class ASTState:
	def __init__(self,t_index,hash,parent,action):
		self.t_index = t_index
		self.hash = hash
		self.parent = parent
		self.action = action
	def __hash__(self):
		if self.parent is None:
			return hash((self.t_index,None,hash(self.action)))
		else:
			return hash((self.t_index,self.parent.hash,hash(self.action)))
	def __eq__(self,other):
		return hash(self) == hash(other)

def ASTStateInit(t_index,parent,action):
	obj = ASTState(t_index,0,parent,action)
	obj.hash = hash(obj)
	return obj

class ASTAction:
	def __init__(self,action):
		self.action = action
	def __hash__(self):
		return hash(tuple(self.action))
	def __eq__(self,other):
		return np.array_equal(self.action, other.action)
	def get(self):
		return self.action

def transition_model(ast):
	def get_initial_state():
		ast.t_index = 1
		ast.initialize()
		# s = ASTStateInit(ast.t_index, None, ASTAction(rsg=copy.deepcopy(ast.initial_rsg)))
		s = ASTStateInit(ast.t_index, None, None)
		ast.sim_hash = s.hash
		return s
	def get_next_state(s0,a0):
		assert ast.sim_hash == s0.hash
		ast.t_index += 1
		ast.update(a0)
		s1 = ASTStateInit(ast.t_index, s0, a0)
		ast.sim_hash = s1.hash
		r = ast.get_reward()
		return s1, r
	def isterminal(s):
		assert ast.sim_hash == s.hash
		return ast.isterminal()
	def go_to_state(target_state):
		s = get_initial_state()
		actions = get_action_sequence(target_state)
		R = 0.0
		for a in actions:
			s,r = get_next_state(s, a)
			R += r
		assert s == target_state
		return R, actions
	return MDP.TransitionModel(get_initial_state, get_next_state, isterminal, ast.params.max_steps, go_to_state)

def get_action_sequence(s):
	actions = []
	while not s.parent is None:
		actions.append(s.action)
		s = s.parent
	actions = list(reversed(actions))
	return actions