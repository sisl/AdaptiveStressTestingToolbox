import mylab.mcts.AdaptiveStressTesting as AST
import mylab.mcts.ASTSim as ASTSim
import mylab.mcts.MCTSdpw as MCTSdpw
import mylab.mcts.AST_MCTS as AST_MCTS
from mylab.mcts import tree_plot

class MCTS:
	"""
	MCTS
	"""
	def __init__(
			self,
		    env,
			stress_test_num,
			max_path_length,
			ec,
			n_itr,
			k,
			alpha,
			clear_nodes,
			log_interval,
		    top_paths,
		    log_tabular=True,
		    plot_tree=False,
		    plot_path=None,
		    plot_format='png'
			):
		self.env = env
		self.stress_test_num = stress_test_num
		self.max_path_length = max_path_length
		self.macts_params = MCTSdpw.DPWParams(max_path_length,ec,n_itr,k,alpha,clear_nodes)
		self.log_interval = log_interval
		self.top_paths = top_paths
		self.log_tabular = log_tabular
		self.plot_tree=plot_tree
		self.plot_path=plot_path
		self.plot_format=plot_format

	def init(self):
		ast_params = AST.ASTParams(self.max_path_length,self.log_interval,self.log_tabular)
		self.ast = AST.AdaptiveStressTest(p=ast_params, env=self.env, top_paths=self.top_paths)

	def train(self):
		self.init()
		if self.plot_tree:
			if self.stress_test_num == 2:
				result,tree = AST_MCTS.stress_test2(self.ast,self.macts_params,self.top_paths,verbose=False, return_tree=True)
			else:
				result,tree = AST_MCTS.stress_test(self.ast,self.macts_params,self.top_paths,verbose=False, return_tree=True)
		else:
			if self.stress_test_num == 2:
				result = AST_MCTS.stress_test2(self.ast,self.macts_params,self.top_paths,verbose=False, return_tree=False)
			else:
				result = AST_MCTS.stress_test(self.ast,self.macts_params,self.top_paths,verbose=False, return_tree=False)
		self.ast.params.log_tabular = False
		for (i,action_seq) in enumerate(result.action_seqs):
			reward, _ = ASTSim.play_sequence(self.ast,action_seq,sleeptime=0.0)
			print("predic reward: ",result.rewards[i])
			print("actual reward: ",reward)	
		if self.plot_tree:
			tree_plot.plot_tree(tree,d=self.max_path_length,path=self.plot_path,format=self.plot_format)