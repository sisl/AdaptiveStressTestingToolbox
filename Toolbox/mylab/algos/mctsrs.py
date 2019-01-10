import mcts.AdaptiveStressTestingRandomSeed as AST_RS
from mylab.algos.mcts import MCTS

class MCTSRS(MCTS):
	"""
	MCTS with Random Seed as action
	"""
	def __init__(self,
		seed,
		rsg_length,
		**kwargs):
		self.seed = seed
		self.rsg_length = rsg_length
		super(MCTSRS, self).__init__(**kwargs)

	def init(self):
		ast_params = AST_RS.ASTParams(self.max_path_length,self.rsg_length,self.seed,self.log_interval,self.log_tabular)
		self.ast = AST_RS.AdaptiveStressTest(p=ast_params, env=self.env, top_paths=self.top_paths)