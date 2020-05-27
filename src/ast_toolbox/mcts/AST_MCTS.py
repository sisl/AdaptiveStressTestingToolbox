import numpy as np

import ast_toolbox.mcts.MCTSdpw as MCTSdpw
import ast_toolbox.mcts.MDP as MDP


class StressTestResults:
    def __init__(self, rewards, action_seqs, q_values):
        self.rewards = rewards
        self.action_seqs = action_seqs
        self.q_values = q_values


def StressTestResultsInit(k):
    rewards = np.zeros(k)
    action_seqs = [None] * k
    q_values = [None] * k
    return StressTestResults(rewards, action_seqs, q_values)


def rollout_getAction(ast):
    def rollout_policy(s, tree):
        return ast.random_action()
    return rollout_policy


def explore_getAction(ast):
    def explore_policy(s, tree):
        return ast.explore_action(s, tree)
    return explore_policy


def stress_test(ast, mcts_params, top_paths, verbose=True, return_tree=False):
    dpw_model = MCTSdpw.DPWModel(ast.transition_model, rollout_getAction(ast), explore_getAction(ast))
    tree = MCTSdpw.DPWTree(mcts_params, dpw_model)
    (mcts_reward, action_seq) = MDP.simulate(tree.f.model, tree, MCTSdpw.selectAction, verbose=verbose)
    results = StressTestResultsInit(top_paths.N)

    results = ast.top_paths
    if return_tree:
        return results, tree.s_tree
    else:
        return results


def stress_test2(ast, mcts_params, top_paths, verbose=True, return_tree=False):
    mcts_params.clear_nodes = False
    mcts_params.n *= ast.params.max_steps

    dpw_model = MCTSdpw.DPWModel(ast.transition_model, rollout_getAction(ast), explore_getAction(ast))
    tree = MCTSdpw.DPWTree(mcts_params, dpw_model)

    s = tree.f.model.getInitialState()
    MCTSdpw.selectAction(tree, s, verbose=verbose)

    results = ast.top_paths
    if return_tree:
        return results, tree.s_tree
    else:
        return results
