import AdaptiveStressTesting as AST
import ASTSim
import MCTSdpw
import AST_MCTS
import numpy as np
import Walk1D


def test_ast():
	MAXTIME = 25 #sim endtime
	RNG_LENGTH = 2
	SIGMA = 1.0 #standard deviation of Gaussian
	SEED = 0 

	sim_params = Walk1D.Walk1DParams()
	sim_params.startx = 1.0
	sim_params.threshx = 10.0
	sim_params.endtime = MAXTIME
	sim_params.logging = True

	sim = Walk1D.Walk1DSimInit(sim_params,SIGMA)
	ast_params = AST.ASTParams(MAXTIME,RNG_LENGTH,SEED,None)
	ast = AST.AdaptiveStressTest(ast_params, sim, Walk1D.initialize, Walk1D.update, Walk1D.isterminal)
	return ast

def run_test():
	MAXTIME = 25 #sim endtime
	RNG_LENGTH = 2
	SIGMA = 1.0 #standard deviation of Gaussian
	SEED = 0 

	sim_params = Walk1D.Walk1DParams()
	sim_params.startx = 1.0
	sim_params.threshx = 10.0
	sim_params.endtime = MAXTIME
	sim_params.logging = True

	sim = Walk1D.Walk1DSimInit(sim_params,SIGMA)
	ast_params = AST.ASTParams(MAXTIME,RNG_LENGTH,SEED,None)
	ast = AST.AdaptiveStressTest(ast_params, sim, Walk1D.initialize, Walk1D.update, Walk1D.isterminal)

	ASTSim.sample(ast)

	#macts_params = MCTSdpw.DPWParams(50,100.0,100,0.5,0.85,1.0,0.0,True,1.0e308,np.uint64(0),10)
	macts_params = MCTSdpw.DPWParams(50,100.0,50,1000.0,0.85,1.0,0.0,True,1.0e308,np.uint64(0),10)
	result = AST_MCTS.stress_test(ast,macts_params)
	#reward, action_seq = result.rewards[1], result.action_seqs[1]
	return result,ast

def show_log(x,stress_test_num):
	MAXTIME = 25 #sim endtime
	RNG_LENGTH = 2
	SIGMA = 1.0 #standard deviation of Gaussian
	SEED = 0 

	sim_params = Walk1D.Walk1DParams()
	sim_params.startx = x
	sim_params.threshx = 10.0
	sim_params.endtime = MAXTIME
	sim_params.logging = True

	sim = Walk1D.Walk1DSimInit(sim_params,SIGMA)
	ast_params = AST.ASTParams(MAXTIME,RNG_LENGTH,SEED,None)
	ast = AST.AdaptiveStressTest(ast_params, sim, Walk1D.initialize, Walk1D.update, Walk1D.isterminal)

	#ASTSim.sample(ast)

	#macts_params = MCTSdpw.DPWParams(50,100.0,100,0.5,0.85,1.0,0.0,True,1.0e308,np.uint64(0),10)
	macts_params = MCTSdpw.DPWParams(50,100.0,1000,0.5,0.85,1.0,0.0,True,1.0e308,np.uint64(0),10)
	if stress_test_num == 2:
		result = AST_MCTS.stress_test2(ast,macts_params,False)
	else:
		result = AST_MCTS.stress_test(ast,macts_params,False)
	#reward, action_seq = result.rewards[1], result.action_seqs[1]
	ASTSim.play_sequence(ast,result.action_seqs[0])
	print(ast.sim.log)
	print(result.rewards)

