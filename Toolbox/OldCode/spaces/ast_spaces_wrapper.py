from mylab.spaces.ast_spaces import ASTSpaces

class ASTSpacesWrapper(ASTSpaces):
	def __init__(self, env):
		self._action_space = env.ast_action_space
		self._observation_space = env.ast_observation_space
		super().__init__()

	@property
	def action_space(self):
	    return self._action_space

	@property
	def observation_space(self):
		return self._observation_space