from gym.envs.registration import register

# logger = logging.getLogger(__name__)

register(
    id='GoExploreAST-v0',
    entry_point='ast_toolbox.envs:GoExploreASTEnv',
    nondeterministic = True,
)

register(
    id='GoExploreAST-v1',
    entry_point='ast_toolbox.envs:Custom_GoExploreASTEnv',
    nondeterministic = True,
)
# register(
#     id='SoccerEmptyGoal-v0',
#     entry_point='gym_soccer.envs:SoccerEmptyGoalEnv',
#     timestep_limit=1000,
#     reward_threshold=10.0,
#     nondeterministic = True,
# )
#
# register(
#     id='SoccerAgainstKeeper-v0',
#     entry_point='gym.envs:SoccerAgainstKeeperEnv',
#     timestep_limit=1000,
#     reward_threshold=8.0,
#     nondeterministic = True,
# )