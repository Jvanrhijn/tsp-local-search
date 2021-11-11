import math
from itertools import chain, product, combinations
from functools import reduce
import copy
from os import path
from matplotlib.pyplot import get

from numpy import add
from solve_tsp import *
from util.kmeans import *
from heuristics.two_opt import *
from heuristics.three_opt import *
from heuristics.lk import *
from heuristics.simulated_annealing import *
from collections import defaultdict
import networkx as nx
#import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout



np.random.seed(43532)

local_maxima_encountered = 0
local_minima_encountered = 0

nvert = 50
#graph, tour = generate_random_tour(nvert, rng_seed=22, kind="Euclidean")
graph, tour = generate_random_tour(nvert, rng_seed=8, kind="Euclidean")
start_tour = copy.deepcopy(tour)
#graph = Graph.from_tspfile("instances/a280.tsp")

vs = np.random.permutation(graph.vertices)
nvert = len(vs)
tour = [frozenset({u, v}) for u, v in zip(vs, vs[1:])] + [frozenset({vs[-1], vs[0]})]
initial_tour = copy.deepcopy(tour)

lb = 0

# number of steps
nsteps = 1000

ks = []

total_gain = 0
total_uphill = 0
gains_by_k = [[] for _ in range(nvert)]
lengths = [path_length(tour, graph.weights)]
gain_uphill = []
gain_downhill = []

# list of cut/joined edge pairs per step
xsys = []
steps = []
a = 1

best_so_far = np.inf
halfway_tour = None

temperature = lambda t: a / np.log(t + 2)
#temperature = lambda t: a / (t + 1)

for step in range(nsteps):


    #new_tour = lk_iteration(tour, graph.weights, kmax=np.inf, greedy=True)
    new_tour = sa_iteration(tour, graph, temperature(step), pick_neighbor=pick_neighbor_3opt)
    #new_tour = lk_iteration(tour, graph.weights)
    #new_tour = two_opt_iteration(tour, graph.weights)
    #new_tour = two_opt_iteration_reverse(tour, graph.weights)
    #new_tour = three_opt_iteration(tour, graph.weights)
    #new_tour = pure_three_opt_iteration(tour, graph.weights)
   
    # find edges cut during iteration
    cut, joined = edge_difference(tour, new_tour)
    k = len(cut)

    if step == nsteps // 10:
        halfway_tour = new_tour

    xsys.append((cut, joined))
    steps.append(f"{cut}, {joined}")

    length = path_length(new_tour, graph.weights)
    gain = path_length(tour, graph.weights) - length
    #if k >= 2:
    gains_by_k[k].append(gain)
    total_gain += gain
    
    if gain < -1e-10:
        gain_uphill.append(abs(gain))
    if gain > 1e-20:
        gain_downhill.append(gain)

    best_so_far = min(best_so_far, length)

    is_2_optimal = abs(path_length(two_opt_iteration(new_tour, graph.weights), graph.weights) - path_length(tour, graph.weights)) < 1e-10
    is_2_maximal = abs(path_length(two_opt_iteration_reverse(new_tour, graph.weights), graph.weights) - path_length(tour, graph.weights)) < 1e-5

    local_maxima_encountered += int(is_2_maximal)
    local_minima_encountered += int(is_2_optimal)

    print(f"k = {k} | gain: {gain:.5f} | total gain: {total_gain:.5f} | length: {length:.5f} | best: {best_so_far:.5f} | 2-opt enc: {local_minima_encountered} | temperature: {temperature(step):.5f} | ave uph g: {np.mean(gain_uphill):.5f} | ave downh g: {np.mean(gain_downhill):.5f}")

    
    lengths.append(path_length(new_tour, graph.weights))


    old_tour = tour
    tour = new_tour


tup = np.arange(1, len(gain_uphill)+1)
guph = np.array([sum(gain_uphill[:i]) for i in tup])

fig, ax = plt.subplots(1, ncols=3)


ax[0].set_title(f"$T = \infty$, $J(x) = {path_length(start_tour, graph.weights):.2f}$")
plot_tour(ax[0], start_tour, graph)
ax[0].xaxis

#ax[1].set_title(f"$T = {temperature(nsteps//10):.4f}$, $J(x) = {path_length(halfway_tour, graph.weights):.2f}$")
#plot_tour(ax[1], halfway_tour, graph)

ax[2].set_title(f"$T = {temperature(nsteps):.4f}$, $J(x) = {path_length(tour, graph.weights):.2f}$")
plot_tour(ax[2], tour, graph)

for a in ax:
    a.xaxis.set_ticks([])
    a.yaxis.set_ticks([])

plt.tight_layout()

plt.show()