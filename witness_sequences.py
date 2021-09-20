from itertools import chain, product, combinations
import copy
from matplotlib.pyplot import get

from numpy import add
from solve_tsp import *
from heuristics.two_opt import *
from heuristics.three_opt import *
from heuristics.lk import *
from collections import defaultdict
import networkx as nx
#import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout


# Node class: useful for building witness DAG below
class Node(int):
    def __new__(cls, *args, **kwargs):
        return  super(Node, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args):
        self.children = [None]
        


np.random.seed(0)

nvert = 50
graph, tour = generate_random_tour(nvert, rng_seed=22, kind="Euclidean")
#graph, tour = generate_random_tour(nvert, rng_seed=8715, kind="Unit")
#graph = Graph.from_tspfile("instances/kn57.tsp")

vs = np.random.permutation(graph.vertices)
nvert = len(vs)
tour = [frozenset({u, v}) for u, v in zip(vs, vs[1:])] + [frozenset({vs[-1], vs[0]})]
initial_tour = copy.deepcopy(tour)

lb = 0

# number of steps
steps = 4000

ks = []

total_gain = 0
gains = []
lengths = [path_length(tour, graph.weights)]

# list of cut/joined edge pairs per step
xsys = []

for step in range(steps):


    #new_tour = lk_iteration(tour, graph.weights, kmax=np.inf)
    #new_tour = lk_iteration(tour, graph.weights, kmax=2)
    new_tour = two_opt_iteration(tour, graph.weights)
    #new_tour = three_opt_iteration(tour, graph.weights)
    #new_tour = pure_three_opt_iteration(tour, graph.weights)
   
    # find edges cut during iteration
    cut, joined = edge_difference(tour, new_tour)
    k = len(cut)

    xsys.append((cut, joined))

    length = path_length(new_tour, graph.weights)
    gain = path_length(tour, graph.weights) - length
    gains.append(gain)
    total_gain += gain

    print(f"k = {k} | gain: {gain:.5f} | total gain: {total_gain:.5f} | length: {length:.5f} | lower bound: {lb:.5f}")
    
    lengths.append(path_length(new_tour, graph.weights))

    if k == 0:
        break

    old_tour = tour
    tour = new_tour


# Find the next step in the sequence (after i)
# that removes an edge added in i
def find_next_link(i, xsys, link):
    link_found = False
    s1 = xsys[i]
    x1, y1 = s1
    for j, s2 in enumerate(xsys[i+1:]):
        x2, y2 = s2
        if set(x2).intersection(set(y1)) == link:
            link_found = True
            break
    if link_found:
        return j+i+1
    else:
        return None


# build witness DAG recursively from a root
def witness_dag_from_root(root, xsys, get_links=lambda y: map(lambda e: set({e}), y), strict=False):
    """Builds the witness DAG as in Englert (2014) from a sequence of 2-opt steps"""

    # add a node for each edge in the initial tour with arbitrary timestamp {1,..,n}
    _, y = xsys[root]

    links = [find_next_link(root, xsys, s) for s in get_links(y)]
    links = list(filter(lambda x: x is not None, links))

    # if strict, then require all children of s to be present
    # otherwise, allow any links
    # NB: in Englert (2014), they build a strict DAG
    if strict and len(links) < len(y):
        return root
    elif not strict and len(links) == 0:
        return root
    else:
        root.children = [witness_dag_from_root(Node(l), xsys, strict=strict) for l in links]
        return root


def create_witness_dag(xsys, get_links=lambda y: map(lambda e: set({e}), y), strict=False):

    nodes = set()

    for i in range(len(xsys)):
        nodes.add(witness_dag_from_root(Node(i), xsys, get_links=get_links, strict=strict))

    return sorted(list(nodes))


# Returns the shortest path length rom
# a node to any leaf
# Incorrectly called the "height" in Englert (2014)
def minheight(node):
    if None in node.children:
        return 0
    else:
        return 1 + min([minheight(c) for c in node.children])

    
def unroll_dag(nodes):
    unrolled = {int(n) for n in nodes}
    arcs = set()
    
    for n in nodes:
        if None not in n.children:
            arcs = arcs.union({(int(n), int(c)) for c in n.children})

    return unrolled, arcs


def get_sub_dag(node, nodes, arcs, k):
    """Obtain the sub-DAG at node, with path length k"""
    assert minheight(node) >= k-1

    if k == 1:
        return nodes, arcs

    existant_children = list(filter(lambda c: c is not None, node.children))
    if len(existant_children) == 0:
        return nodes, arcs

    nodes += [int(c) for c in existant_children]
    arcs += [(int(node), int(c)) for c in existant_children]

    n, a = zip(*[get_sub_dag(c, nodes, arcs, k-1) for c in existant_children])
    return set(chain.from_iterable(n)), set(chain.from_iterable(a))


def unroll_subdag(root, nodes, arcs, new_nodes, new_arcs, k):
    if k == 0:
        return new_nodes, new_arcs

    if type(root) != int:
        child_arcs = list(filter(lambda a: a[0] == int(root.strip("'")), arcs))
    else:
        child_arcs = list(filter(lambda a: a[0] == root, arcs))


    for ca in child_arcs:
        c = str(ca[1])

        if f"{c}" in new_nodes:
            c += "'"
            if c in new_nodes:
                c += "'"

        new_nodes.add(c)
        new_arcs.add((f"{root}", c))

        subnodes, subarcs = unroll_subdag(c, nodes, arcs, new_nodes, new_arcs, k-1)

        new_nodes = new_nodes.union(subnodes)
        new_arcs = new_arcs.union(subarcs)

    return new_nodes, new_arcs



dag = create_witness_dag(xsys[:-1], get_links=lambda y: list(map(set, combinations(y, 1))), strict=True)
nodes, arcs = unroll_dag(dag)

root = 0
height = 5
n, a = get_sub_dag(dag[root], [dag[root]], [], height)
n2, a2 = unroll_subdag(root, n, a, {f"{root}"}, set(), height-1)

nodes, arcs = unroll_dag(dag)


# visualize a graph
plt.figure()

g = nx.DiGraph()

g.add_edges_from(a2)

nx.nx_agraph.write_dot(g,'test.dot')
pos = graphviz_layout(g, prog='dot')

nx.draw_networkx_nodes(g, pos, node_size=70)
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_labels(g, pos, font_size=8)

plt.show()