import time
import mcts.mctstracker as mctstracker
import mcts.BoundedPriorityQueues as BPQ
import numpy as np

class DPWParams:
	def __init__(self, d, ec, n, k, alpha, clear_nodes, top_k): #like constructor self must be as the first
		self.d = d #search depth
		self.ec = ec #exploration constant
		self.n = n #number of iterations
		self.k = k
		self.alpha = alpha
		self.clear_nodes = clear_nodes
		self.top_k = top_k

class DPWModel:
	def __init__(self, model, getAction, getNextAction):
		self.model = model
		self.getAction = getAction #expert action used in rollout
		self.getNextAction = getNextAction #exploration strategy

class StateActionStateNode:
	def __init__(self, n ,r):
		self.n = n #UInt64
		self.r = r #Float64
	def __init__(self):
		self.n = 0
		self.r = 0.0

class StateActionNode:
	def __init__(self, s, n, q):
		self.s = s #Dict{State,StateActionStateNode}
		self.n = n #UInt64
		self.q = q #Float64
	def __init__(self):
		self.s = {}
		self.n = 0
		self.q = 0.0

class StateNode:
	def __init__(self, a, n):
		self.a = a #Dict{Action,StateActionNode}
		self.n = n #UInt64
	def __init__(self):
		self.a = {}
		self.n = 0

class DPW:
	def __init__(self, s, p, f, tracker, top_paths):
		self.s = s #Dict{State,StateNode}
		self.p = p #DPWParams
		self.f = f #DPWModel
		self.tracker = tracker #MCTSTracker
		self.top_paths = top_paths #BoundedPriorityQueue

def DPWInit(p,f,top_paths):
	s = {}
	p = p
	f = f
	tracker = mctstracker.MCTSTrackerInit()
	# top_paths = BPQ.BoundedPriorityQueue(p.top_k)
	top_paths = top_paths
	return DPW(s,p,f,tracker,top_paths)

def saveBackwardState(dpw, old_d, new_d, s_current):
    if not (s_current in old_d):
    	return new_d
    s = s_current
    while s != None:
        new_d[s] = old_d[s]
        s = s.parent 
    return new_d

def saveForwardState(old_d, new_d, s):
    if not (s in old_d):
    	return new_d
    new_d[s] = old_d[s]
    for sa in old_d[s].a.values():
        for s1 in sa.s.keys():
            saveForwardState(old_d,new_d,s1)
    return new_d

def saveState(dpw, old_d, s):
    new_d = {}
    saveBackwardState(dpw, old_d, new_d, s)
    saveForwardState(old_d, new_d, s)
    return new_d

def trace_q_values(dpw, s_current):
    q_values = []
    if not (s_current in  dpw.s):
    	return q_values
    s = s_current
    while s.parent != None:
        q = dpw.s[s.parent].a[s.action].q
        q_values.append(q)
        s = s.parent 
    return list(reversed(q_values))

def selectAction(dpw, s, verbose=False):
	if dpw.p.clear_nodes:
		new_dict = saveState(dpw,dpw.s,s)
		dpw.s.clear()
		dpw.s = new_dict

	d = dpw.p.d
	starttime_us = time.time()*1e6
	for i in range(dpw.p.n):
		#print("i: ",i)
		R, actions = dpw.f.model.goToState(s)
		dpw.tracker.empty()
		dpw.tracker.append_actions(actions)
		qvals = trace_q_values(dpw, s)
		dpw.tracker.append_q_values(qvals)

		R += simulate(dpw, s, d, verbose = verbose)
		dpw.tracker.combine_q_values()
		dpw.top_paths.enqueue(dpw.tracker, R, make_copy=True)
	dpw.f.model.goToState(s)
	print("Size of sdict: ", len(dpw.s))
	cS = dpw.s[s]
	A = list(cS.a.keys())
	nA = len(A)
	Q = np.zeros(nA)
	for i in range(nA):
		Q[i] = cS.a[A[i]].q
	assert len(Q) != 0
	i = np.argmax(Q)
	return A[i]

def simulate(dpw, s, d, verbose=False):
	# print("simulate start: ",d)
	# print("s: ",s)
	# print("s parent: ",s.parent)
	if (d == 0) | dpw.f.model.isEndState(s):
		# print("simulate end d==0 or terminal")
		return 0.0
	if not (s in dpw.s):
		dpw.s[s] = StateNode()
		# print("rollout")
		return rollout(dpw,s,d)
	dpw.s[s].n += 1
	if len(dpw.s[s].a) < dpw.p.k*dpw.s[s].n**dpw.p.alpha:
		# print("new action: ",dpw.p.k*dpw.s[s].n**dpw.p.alpha)
		a = dpw.f.getNextAction(s,dpw.s)
		# print("new action: ",a.get())
		if not (a in dpw.s[s].a):
			dpw.s[s].a[a] = StateActionNode()
	else:
		# print("old action")
		cS = dpw.s[s]
		A = list(cS.a.keys())
		nA = len(A)
		UCT = np.zeros(nA)
		nS = cS.n
		for i in range(nA):
			cA = cS.a[A[i]]
			assert nS > 0
			assert cA.n > 0
			UCT[i] = cA.q + dpw.p.ec*np.sqrt(np.log(nS)/float(cA.n))
		a = A[np.argmax(UCT)]

	dpw.tracker.push_action(a)
	qval = dpw.s[s].a[a].q
	dpw.tracker.push_q_value(qval)

	sp,r = dpw.f.model.getNextState(s,a)
	# print("new sp: ",sp in dpw.s.keys())
	if not (sp in dpw.s[s].a[a].s):
		dpw.s[s].a[a].s[sp] = StateActionStateNode()
		dpw.s[s].a[a].s[sp].r = r
		dpw.s[s].a[a].s[sp].n = 1
	else:
		dpw.s[s].a[a].s[sp].n += 1


	q = r + simulate(dpw,sp,d-1)
	cA = dpw.s[s].a[a]
	cA.n += 1
	cA.q += (q-cA.q)/float(cA.n)
	dpw.s[s].a[a] = cA

	# print("simulate end")
	return q

def rollout(dpw, s, d):
	# print("rollout start, d is ",d)
	if (d == 0) | dpw.f.model.isEndState(s):
		# print("rollout end d==0 or terminal")
		return 0.0
	else:
		a = dpw.f.getAction(s,dpw.s)
		dpw.tracker.push_action(a)
		sp,r = dpw.f.model.getNextState(s,a)
		qval = (r+rollout(dpw,sp,d-1))
		dpw.tracker.push_q_value2(qval)
		# print("rollout end, d is ",d)
		return qval














