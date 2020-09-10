"""Environments for formulating validation as an AST reinforcement learning problem."""
import importlib
__all__ = []
if importlib.util.find_spec('bsddb3') is not None:
    # Only load bsddb3 dependent packages if it is installed
    from ast_toolbox.envs.go_explore_ast_env import GoExploreASTEnv  # noqa
    from ast_toolbox.envs.go_explore_ast_env import Custom_GoExploreASTEnv  # noqa

    __all__ += ['GoExploreASTEnv', 'Custom_GoExploreASTEnv']

from .ast_env import ASTEnv  # noqa

__all__ += ['ASTEnv']
