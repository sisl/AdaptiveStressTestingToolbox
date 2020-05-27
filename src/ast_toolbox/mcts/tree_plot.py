import uuid

import pydot


def get_root(tree):
    for s in tree.keys():
        if s.parent is None:
            return s


def s2node(s, tree):
    if s in tree.keys():
        return pydot.Node(str(uuid.uuid4()), label='n=' + str(tree[s].n))
    else:
        return None


def add_children(s, s_node, tree, graph, d):
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
