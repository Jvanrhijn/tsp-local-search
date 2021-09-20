from os import path
import numpy as np

from solve_tsp import *
from heuristics.two_opt import *
from heuristics.three_opt import *
from heuristics.lk import *
from heuristics.simulated_annealing import *


np.random.seed(52)

steps = 40000
eq_steps = 40
d = 1
temperature = 1
best_cost = np.inf

nvert = 100
graph, tour = generate_random_tour(nvert, rng_seed=22, kind="Euclidean")
costs = [path_length(tour, graph.weights)]

neighbors = generate_neighbors(tour, graph)

#plt.axis([0, steps, 0, path_length(tour, graph.weights)+10])


for t in range(steps):
    temperature = d / (1 + math.log2(t+1))
    best_cost = min(costs[-1], best_cost)

    #temperature *= 1 + (path_length(tour, graph.weights) - best_cost) / path_length(tour, graph.weights)

    for _ in range(eq_steps):
        tour = sa_iteration(tour, graph, temperature, pick_neighbor=pick_neighbor_2opt)
    #tour = two_opt_iteration(tour, graph.weights)
    
    costs.append(path_length(tour, graph.weights))

    #plt.scatter(t, costs[-1], s=3, color="blue")
    #plt.pause(0.01)
    print(f"{costs[-1]:.5f}    {best_cost:.5f}")

plt.figure()
plt.plot(costs)
plt.show()