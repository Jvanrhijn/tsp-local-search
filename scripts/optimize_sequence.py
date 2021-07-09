from itertools import chain, product
import copy

from numpy import add
from solve_tsp import *
from heuristics.two_opt import *
from heuristics.three_opt import *
from heuristics.lk import *
from collections import defaultdict
import networkx as nx
#import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout



np.random.seed(0)

nvert = 20
#graph, tour = generate_random_tour(nvert, rng_seed=22, kind="Euclidean")
graph, tour = generate_ktree(25, 20, rng_seed=0, kind="Euclidean")

#fig, ax = plt.subplots(1)
#plot_tour(ax, graph.edges, graph, alpha=1)
#plt.show()

print(graph.vertices)

# delete a couple of edges

#graph, tour = generate_random_tour(nvert, rng_seed=15, kind="Unit")
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
    #new_tour = lk_iteration(tour, graph.weights, kmax=3)
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

#plt.figure()
#plt.plot(gains)
#
#plt.figure()
#plt.plot(lengths)


def find_linked_pairs(xsys):
    linked_pairs = []
    seen = set()

    for i, (x0, y0) in enumerate(xsys[:-1]):

        # if Si was already processed, skip it
        if i in seen:
            continue

        s0 = frozenset({frozenset(x0), frozenset(y0)})
        v0 = set(chain.from_iterable([*y0]))
            
        for j, (x1, y1) in enumerate(xsys[i+1:]):

            # if Sj was already processed, skip it
            if j+i+1 in seen or i in seen:
                continue

            s1 = frozenset({frozenset(x1), frozenset(y1)})
            v1 = set(chain.from_iterable([*x1]))
            intersect = set(y0).intersection(set(x1))
            #if v0 == v1 and s0 != s1:
            #if set(y0) == set(x1) and s0 != s1:
            #if intersect:

            if len(intersect) >= 2 and s0 != s1:
            #if v1 == v0 and s0 != s1:
                if i not in seen and j+i+1 not in seen:
                    seen.add(i); seen.add(j+i+1)
                    linked_pairs.append((s0, s1))

                    # if Si is paired off, we continue to S_(i+1)
                    continue

    return linked_pairs


def find_linked_triples(xsys):
    linked_triples = []
    seen = set()

    for i, (x0, y0) in enumerate(xsys):
        s0 = frozenset({frozenset(x0), frozenset(y0)})
        v0 = set(chain.from_iterable([*x0]))

        for j, (x1, y1) in enumerate(xsys[i+1:]):

            s1 = frozenset({frozenset(x1), frozenset(y1)})
            v1 = set(chain.from_iterable([*y1]))

            if set(y0) == set(x1) and s0 != s1:

                for k, (x2, y2) in enumerate(xsys[j+1:]):
                    s2 = frozenset({frozenset(x2), frozenset(y2)})
                    v2 = set(chain.from_iterable([*x2]))

                    if set(y1) == set(x2) and s1 != s2 and s0 != s2:

                        if i not in seen and j+i+1 not in seen and k+j+i+2 not in seen:
                            seen.add(i); seen.add(j+i+1); seen.add(k+j+i+2)
                            linked_triples.append((s0, s1, s2))

    return linked_triples


def find_2_linked_triples(xsys):
    linked_triples = []
    seen = set()

    for i, (x0, y0) in enumerate(xsys):

        if i in seen:
            continue

        s0 = [frozenset(x0), frozenset(y0)]
        v0 = set(chain.from_iterable([*x0]))
        
        for j in range(i+1, len(xsys)):

            x1, y1 = xsys[j]
            s1 = [{frozenset(x1), frozenset(y1)}]
            intersect1 = set(y0).intersection(set(x1))

            # if no intersection or steps are identical: move on
            if len(intersect1) != 2 or s0 == s1:
                continue

            for k in range(j+1, len(xsys)):

                x2, y2 = xsys[k]
                s2 = [frozenset(x2), frozenset(y2)]
                intersect2 = set(y1).intersection(set(x2))

                # if no intersection or any of the steps are identical: move on
                if len(intersect2) != 2 or s1 == s2 or s0 == s2:
                    continue
                
                # this statement guarantees disjointness of the found triples
                # only if none of Si, Sj, Sk have been added to any triple before,
                # then we add (Si, Sj, Sk)
                if i not in seen and j not in seen and k not in seen:

                    seen = seen.union({i, j, k})
                    linked_triples.append((s0, s1, s2))

    return linked_triples


def count_i_linked(xsys, i):
    l = []

    for j, s0 in enumerate(xsys):
        for s1 in xsys[j+1:]:
            l.append((s0, s1))

    def linked(pair):
        s0, s1 = pair
        x0, y0 = s0
        x1, y1 = s1
        return len(set(y0).intersection(set(x1))) == i

    return list(filter(linked, l))


def find_next_link(i, xsys, links):
    link_found = False
    s1 = xsys[i]
    x1, y1 = s1
    for j, s2 in enumerate(xsys[i+1:]):
        x2, y2 = s2
        if len(set(x2).intersection(set(y1))) == links:
            link_found = True
            break
    if link_found:
        return j+i+1, s2
    else:
        return None


def find_chain_starting_at(i, subchain, xsys, links):
    if i == len(xsys)-1:
        return subchain

    res = find_next_link(i, xsys, links)

    if res:
        subchain.append(res[1])
        return find_chain_starting_at(res[0], subchain, xsys, links)
    else:
        return subchain


def create_witness_dag(xsys, initial_tour):
    """Builds the witness DAG as in Englert (2014) from a sequence of 2-opt steps"""

    # add a node for each edge in the initial tour with arbitrary timestamp {1,..,n}
    #nodes = set(copy.deepcopy(initial_tour))
    n = len(initial_tour)

    nodes = list(range(1, n+1))
    arcs = set()
    leaves = {edge: idx for idx, edge in zip(nodes, initial_tour)}

    for i, s in enumerate(xsys):

        x, y = s
        for k in range(1, len(x)+1):
            nodes.append(n + len(x)*i + k)

        #fprev, gprev = x
        #f, g = y

        removed_idxs = [leaves[e] for e in x]
        #i1, i2 = leaves[fprev], leaves[gprev]

        for removed_edge in x:
            del leaves[removed_edge]

        new_idxs = [n + len(x)*i + k for k in range(1, len(x)+1)]
        #j1 = n + 2*i + 1
        #j2 = n + 2*i + 2

        for added_edge, idx in zip(y, new_idxs):
            leaves[added_edge] = idx
        #leaves[f] = j1
        #leaves[g] = j2

        #arcs = arcs.union({(fprev, g), (gprev, g), (fprev, f), (gprev, f)})
        #arcs = arcs.union({(i1, j1), (i1, j2), (i2, j1), (i2, j2)})
        arcs = arcs.union(set(product(removed_idxs, new_idxs)))
    
    return nodes, arcs


nodes, arcs = create_witness_dag(xsys[:-1], initial_tour)

plt.figure()

g = nx.DiGraph()

g.add_edges_from(arcs)

nx.nx_agraph.write_dot(g,'test.dot')
pos = graphviz_layout(g, prog='dot')
#
nx.draw_networkx_nodes(g, pos, node_size=35)
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_labels(g, pos, font_size=5)

count_disjoint = np.array([len(count_i_linked(xsys[:i], 0)) for i in range(len(xsys))])
count_2linked = np.array([len(count_i_linked(xsys[:i], 2)) for i in range(len(xsys))])
count_1linked = np.array([len(count_i_linked(xsys[:i], 1)) for i in range(len(xsys))])

ts = np.arange(1, len(xsys)+1)

#plt.figure()
#plt.plot(ts, count_2linked, label="2-linked")
#plt.plot(ts, count_1linked, label="1-linked")
##plt.plot(ts, count_disjoint, label="0-linked")
#plt.plot(ts, (ts-1)*(2*ts - nvert) / (nvert + 2*ts - 2))
#plt.legend()

#linked_pairs = find_linked_pairs(xsys[:len(xsys)//4])

num_steps = len(xsys)
ts = np.array(list(range(1, num_steps, 5)))
num_linked_pairs = []

for t in ts:
    linked_pairs = find_2_linked_triples(xsys[:t])
    num_linked_pairs.append(len(linked_pairs))


plt.figure()
plt.plot(ts, np.array(num_linked_pairs))# / np.array(ts))
plt.xlabel("Sequence length")
plt.ylabel("Number of disjoint 2-linked triples")
#plt.plot(ts, 3*ts / 11)

#fig, ax = plt.subplots(1)
#
#edges1 = set(chain.from_iterable(linked_pairs[-1][0]))
#edges2 = set(chain.from_iterable(linked_pairs[-1][1]))
##edges3 = set(chain.from_iterable(linked_pairs[2][2]))
##
#linked_edges = edges1.intersection(edges1).intersection(edges2)
#
##plot_tour(ax, graph.edges, graph, alpha=0.01)
#plot_tour(ax, edges1, graph, color="black")
#plot_tour(ax, edges2, graph, color="red")
##plot_tour(ax, edges3, graph, color="blue")
#plot_tour(ax, linked_edges, graph, color="green")
#

#print(num_steps, num_linked_pairs, int((2*num_steps - nvert) / 7))

fig, ax = plt.subplots(1)
plot_tour(ax, tour, graph, alpha=1)
plot_tour(ax, graph.edges, graph, alpha=0.05)

print(len(xsys))


plt.show()