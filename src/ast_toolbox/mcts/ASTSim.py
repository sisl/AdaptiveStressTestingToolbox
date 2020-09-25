import ast_toolbox.mcts.MDP as MDP


class AcionSequence:
    """Sturcture storing the actions sequences.

    Parameters
    ----------
    sequence : list
        The list of actions.
    index : int, optional
        The initial action index in the sequence.
    """

    def __init__(self, sequence, index=0):
        self.sequence = sequence
        self.index = index


def action_seq_policy(action_seq, s):
    """The policy wrapper for the action sequence.

    Parameters
    ----------
    action_seq : :py:class:`ast_toolbox.mcts.ASTSim.AcionSequence`
        The action sequence.
    s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The AST state.

    Returns
    ----------
    action : `ast_toolbox.mcts.AdaptiveStressTesting.ASTAction`
        The AST action.
    """
    action = action_seq.sequence[action_seq.index]
    action_seq.index += 1
    return action


def play_sequence(ast, actions, verbose=False, sleeptime=0.0):
    """Rollout the action sequence.

    Parameters
    ----------
    ast : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.AdaptiveStressTest`
        The AST object.
    actions : list
        The action sequence.
    verbose : bool, optional
        Whether to log the rollout information.
    sleeptime: float, optional
        The pause time between each step.

    Returns
    ----------
    rewards : list[float]
        The rewards.
    actions2 : list
        The action sequence of the path. Should be the same as the input actions.
    """
    rewards, actions2 = MDP.simulate(ast.transition_model, AcionSequence(actions), action_seq_policy, verbose=verbose, sleeptime=sleeptime)
    assert actions == actions2
    return rewards, actions2
