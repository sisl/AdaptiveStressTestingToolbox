import numpy as np
import copy

class RSG:
	def __init__(self,state):
		self.state = state
	def __eq__(self,other):
		return np.array_equal(self.state,other.state)
	def __hash__(self):
		return hash(tuple(self.state))
	def length(self):
		return len(self.state)
	def next1(self):
		self.state = np.uint32(list(map(hash_uint32,self.state)))
	def next(self):
		rsg1 = copy.deepcopy(self)
		rsg1.next1()
		return rsg1
	def set_from_seed(self, length, seed):
		self.state = seed_to_state_itr(length,seed)

def RSGInit(state_length=1,seed=0):
	return RSG(seed_to_state_itr(state_length,seed))

def seed_to_state_itr(state_length,seed):
	state = []
	seedi = seed
	for i in range(state_length):
		# print(seedi)
		state.append(seedi)
		seedi = hash_uint32(seedi)
	return np.array(state,dtype=np.uint32)

def isequal(rsg1,rsg2):
	return rsg1 == rs2

def hash_uint32(x):
	# print("x: ",x,"hash x: ",hash(str(x)))
	return np.uint32(hash(str(x)) & 0x00000000FFFFFFFF)

def set_gv_rng_state(a):
	if type(a) == np.uint32:
		return np.random.seed(np.array([a],dtype=np.uint32))
	else:
		return np.random.seed(a)

def set_global(rsg):
	return set_gv_rng_state(rsg.state)

