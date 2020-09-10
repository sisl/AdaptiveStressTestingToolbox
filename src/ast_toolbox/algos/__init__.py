"""Algorithms for solving AST formulated RL problems."""
import importlib

from .ga import GA  # noqa
from .gasm import GASM  # noqa
from .mcts import MCTS  # noqa
from .mctsbv import MCTSBV  # noqa
from .mctsrs import MCTSRS  # noqa

__all__ = ['GA', 'GASM', 'MCTS', 'MCTSBV', 'MCTSRS']

if importlib.util.find_spec('bsddb3') is not None:
    # Only load bsddb3 dependent packages if it is installed
    from .go_explore import GoExplore  # noqa
    from .backward_algorithm import BackwardAlgorithm  # noqa

    __all__ += ['GoExplore', 'BackwardAlgorithm']

# from .random_search import RandomSearch
# from .trpo import TRPO
