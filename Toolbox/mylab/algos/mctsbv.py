import mylab.mcts.AdaptiveStressTestingBlindValue as AST_BV
from mylab.algos.mcts import MCTS

class MCTSBV(MCTS):
	"""
	MCTS with Blind Value
	"""
	def __init__(self,
		M,
		**kwargs):
		"""
		:param M: the number of randon decisions generated for the action pool
		:return: No return value.
		"""
		self.ec = kwargs['ec']
		self.M = M
		super(MCTSBV, self).__init__(**kwargs)

	def init(self):
		ast_params = AST_BV.ASTParams(self.max_path_length,self.ec,self.M,self.log_interval,self.log_tabular)
		self.ast = AST_BV.AdaptiveStressTestBV(p=ast_params, env=self.env, top_paths=self.top_paths)