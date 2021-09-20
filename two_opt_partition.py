import math

from solve_tsp import solve_tsp
import numpy as np
import matplotlib.pyplot as plt

from heuristics.two_opt import two_opt_iteration
from graph.graph import *
from solve_tsp import *
from util.kmeans import *

np.random.seed(44355)

nvert = 200
s = int(math.floor(nvert / math.log2(nvert)))
print(s)

graph, tour = generate_random_tour(nvert, kind="Euclidean", rng_seed=10)
opt, it = solve_tsp(graph, tour, optimizer=two_opt_iteration)

x = graph.vertices[:nvert//2]

pos = graph.positions
#left = graph.vertices[:nvert//2]
#right = list(set(graph.vertices) - set(left))

# partition the vertices using Karp's algorithm
#left = [v for v in graph.vertices if pos[v][0] < 0.5]
#right = [v for v in graph.vertices if pos[v][0] >= 0.5]
clusters, _, _ = kmeans(list(pos.values()), s)

sets = []
for i in range(s):
    sets.append(list(filter(lambda v: list(pos[v]) in list(map(list, clusters[i])), graph.vertices)))

print(max(list(map(len, sets))), nvert // s)


subgraphs = [graph.vertex_induced_subgraph(x) for x in sets]
tours = [[frozenset({u, v}) for u, v in zip(x, x[1:])] + [frozenset({x[-1], x[0]})] for x in sets]

sub_opt = [solve_tsp(sg, t, optimizer=two_opt_iteration)[0] for sg, t in zip(subgraphs, tours)]

fig, ax = plt.subplots(ncols=2)
for so in sub_opt:
    plot_tour(ax[0], so, graph)

plot_tour(ax[1], opt, graph)

partitioned_tour_length_ub = sum(path_length(subtour, graph.weights) for subtour in sub_opt) #+ s*math.sqrt(2)
print(f"{partitioned_tour_length_ub:.5f}    {path_length(opt, graph.weights):.5f}")

plt.show()