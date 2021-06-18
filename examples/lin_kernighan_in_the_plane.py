import numpy as np
import matplotlib.pyplot as plt

from graph.graph import Graph
from heuristics.lk import lk_iteration
from solve_tsp import solve_tsp
from util.util import path_length, unflatten_tour, plot_tour


"""This example shows how to optimize a tour in the plane, using the Lin-Kernighan algorithm"""

np.random.seed(0)

# Generate a random complete graph from placing points in [0, 1]^2
num_vertices = 100
positions = np.random.uniform(size=(num_vertices, 2), low=0, high=1)

vertices = list(range(num_vertices))
points = {v: p for v, p in zip(vertices, positions)}

g = Graph.fully_connected(vertices)
g.place_vertices(points)

initial_tour = unflatten_tour(vertices)
initial_length = path_length(initial_tour, g.weights)

# 'greedy' keyword: greedy=False means the best improving edge is determined
# at every iteration; this is the default
# True means the first improving edge is used
# 
tour, num_iterations = solve_tsp(g, initial_tour, lambda *x: lk_iteration(*x, greedy=False), it_max=1000)
final_length = path_length(tour, g.weights)

fig, ax = plt.subplots(ncols=2)

plot_tour(ax[0], initial_tour, g)
ax[0].set_title(f"Length: {initial_length:.2f}")

plot_tour(ax[1], tour, g)
ax[1].set_title(f"Length: {final_length:.2f}")

plt.show()
