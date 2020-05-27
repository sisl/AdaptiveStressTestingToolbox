import uuid

import matplotlib as mpl
import pydot
from matplotlib import pyplot as plt

mpl.use('Agg')


def get_root(tree):
    for s in tree.keys():
        if s.parent is None:
            return s


def s2node(s, tree):
    if s in tree.keys():
        # return pydot.Node(str(uuid.uuid4()),label='n='+str(tree[s].n))
        return pydot.Node(str(uuid.uuid4()), label=str(tree[s].v))
    else:
        return None


def add_children(s, s_node, tree, graph, d):
    if d > 0:
        for a in tree[s].a.keys():
            tree[s].a[a].n
            tree[s].a[a].q
            assert len(tree[s].a[a].s.keys()) == 1
            for ns in tree[s].a[a].s.keys():
                ns_node = s2node(ns, tree)
                if ns_node is not None:
                    graph.add_node(ns_node)
                    # graph.add_edge(pydot.Edge(s_node, ns_node, label="n="+str(n)+" q="+str(q)))
                    graph.add_edge(pydot.Edge(s_node, ns_node, label=str(ns.action[0])))
                    add_children(ns, ns_node, tree, graph, d - 1)


def get_node_num_next(s, tree, depths, nodeNums, d):
    d = d + 1
    if (len(depths) <= d) and (len(tree[s].a.keys()) > 0):
        depths.append(d)
        nodeNums.append(0)

    for a in tree[s].a.keys():
        for ns in tree[s].a[a].s.keys():
            nodeNums[d] += 1
            if ns in tree.keys():
                get_node_num_next(ns, tree, depths, nodeNums, d)


def plot_tree(tree, d, path, format="svg"):
    graph = pydot.Dot(graph_type='digraph')
    root = get_root(tree)
    root_node = s2node(root, tree)
    graph.add_node(root_node)
    add_children(root, root_node, tree, graph, d)
    filename = path + "." + format
    if format == "svg":
        graph.write_svg(filename)
    elif format == "png":
        graph.write_png(filename)


def plot_node_num(tree, path, format="svg"):
    root = get_root(tree)
    depths = [0]
    nodeNums = [1]
    d = 0
    get_node_num_next(root, tree, depths, nodeNums, d)
    filename = path + "." + format
    plt.plot(depths, nodeNums)
    plt.xlabel('Depth')
    plt.ylabel('Node Number')
    plt.savefig(filename)
