import numpy as np

class Walk1DParams:
	def __init__(self,startx=1.0,threshx=10.0,endtime=20,logging=False):
		self.startx = startx
		self.threshx = threshx
		self.endtime = endtime
		self.logging = logging

class Walk1DSim:
	def __init__(self,p,x,t,sigma,log):
		self.p = p
		self.x = x
		self.t = t
		self.sigma = sigma
		self.log = log

def Walk1DSimInit(params,sigma):
	return Walk1DSim(params,params.startx,0,sigma,[])

def initialize(sim):
	sim.t=0
	sim.x=sim.p.startx
	sim.log.clear()
	if sim.p.logging:
		sim.log.append(sim.x)

def update(sim):
	sim.t += 1
	r = np.random.normal(scale=sim.sigma)
	sim.x += r
	prob = 1/(sim.sigma*np.sqrt(2*np.pi))*np.exp(-(r-0.0)**2/(2*sim.sigma**2))
	dist = max(sim.p.threshx-abs(sim.x),0.0)
	if sim.p.logging:
		sim.log.append(sim.x)
	return prob,isevent(sim),dist

def isevent(sim):
	return abs(sim.x) >= sim.p.threshx

def isterminal(sim):
	return isevent(sim) | (sim.t >= sim.p.endtime)