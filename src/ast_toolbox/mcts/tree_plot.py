import uuid

import pydot


def get_root(tree):
    """Get the root node of the tree.

    Parameters
    ----------
    tree : dict
        The tree.

    Returns
    ----------
    s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The root state.
    """
    for s in tree.keys():
        if s.parent is None:
            return s


def s2node(s, tree):
    """Transfer the AST state to pydot node.

    Parameters
    ----------
    s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The AST state.
    tree : dict
        The tree.

    Returns
    ----------
    node : :py:class:`pydot.Node`
        The pydot node.
    """
    if s in tree.keys():
        return pydot.Node(str(uuid.uuid4()), label='n=' + str(tree[s].n))
    else:
        return None


def add_children(s, s_node, tree, graph, d):
    """Add successors of s into the graph.

    Parameters
    ----------
    s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The AST state.
    s_node : :py:class:`pydot.Node`
        The pydot node corresponding to s.
    tree : dict
        The tree.
    graph : :py:class:`pydot.Dot`
        The pydot graph.
    d : int
        The depth.
    """
    if d > 0:
        for a in tree[s].a.keys():
            n = tree[s].a[a].n
            q = tree[s].a[a].q
            assert len(tree[s].a[a].s.keys()) == 1
            for ns in tree[s].a[a].s.keys():
                ns_node = s2node(ns, tree)
                if ns_node is not None:
                    graph.add_node(ns_node)
                    graph.add_edge(pydot.Edge(s_node, ns_node, label="n=" + str(n) + " a=" + str(a.get()) + " q=" + str(q)))
                    # graph.add_edge(pydot.Edge(s_node, ns_node))
                    add_children(ns, ns_node, tree, graph, d - 1)


def plot_tree(tree, d, path, format="svg"):
    """Plot the tree.

    Parameters
    ----------
    tree : dict
        The tree.
    d : int
        The depth.
    path : str
        The plotting path.
    format : str
        The plotting format.
    """
    graph = pydot.Dot(graph_type='digraph')
    root = get_root(tree)
    root_node = s2node(root, tree)
    graph.add_node(root_node)
    add_children(root, root_node, tree, graph, d)
    filename = path + "." + format
    if format == "svg":
        graph.write(filename)
    elif format == "png":
        graph.write(filename)
