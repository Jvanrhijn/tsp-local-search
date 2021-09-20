import math
from os import path

import tqdm

from solve_tsp import solve_tsp
import numpy as np
import matplotlib.pyplot as plt

from heuristics.two_opt import two_opt_iteration
from heuristics.lk import lk_iteration
from heuristics.simulated_annealing import *
from graph.graph import *
from solve_tsp import *
import random


np.random.seed(35)


d = 0.1
temperature = d

fraction_improving = []
mingain = np.inf
changing = []

nsamples = 1

nuphills = []

for _ in tqdm.tqdm(range(nsamples)):

    nuphill = []
    repetitions = 0

    nvert = 25
    graph, tour = generate_random_tour(nvert, kind="Euclidean", rng_seed=1234)

    init_length = path_length(tour, graph.weights)

    total_improvement = 0
    total_worsening = 0
    worsening = []
    improvement = []

    steps = 100

    best_length = path_length(tour, graph.weights)

    for t in range(steps):
        temperature = d / (t+1)

        new_tour = sa_iteration(tour, graph, temperature, transition_function=boltzmann, pick_neighbor=pick_neighbor_2opt)

        is_2_optimal = (path_length(two_opt_iteration(new_tour, graph.weights), graph.weights) == path_length(tour, graph.weights))

        x, y = edge_difference(tour, new_tour)
        gain = path_length(tour, graph.weights) - path_length(new_tour, graph.weights)

        if gain < -1e-10:
            mingain = min(mingain, abs(gain))
            total_worsening += abs(gain)
            worsening.append(total_worsening)
        elif gain > 1e-10:
            mingain = min(mingain, abs(gain))
            total_improvement += gain
            improvement.append(total_improvement)
        else:
            repetitions += 1

        x, y = edge_difference(tour, new_tour)

        best_length = min(best_length, path_length(new_tour, graph.weights))
        length = path_length(new_tour, graph.weights)

        #print(f"step: {t+1:5d} | gain: {gain:.5f} | length: {length:.5f} | best: {best_length:.5f} | total gain: {total_improvement:.5f} | total worsening: {total_worsening:.6f} | net gain: {total_improvement-total_worsening:.5f} | uphill: {len(worsening)} | min gain: {mingain:.10f} | repetitions: {repetitions}")

        changing.append(len(worsening) + len(improvement))
        nuphill.append(len(worsening))

        if is_2_optimal:
            break

        tour = new_tour
    
    nuphills.append(nuphill[-1])


print(mingain)
nuphills = np.array(nuphills)

ks = np.arange(1, steps+1)
tailp = []

for k in ks:
    tailp.append(sum(nuphills >= k) / len(nuphills))

plt.figure()
#plt.hist(nuphills, bins=10)
plt.loglog(ks, tailp)
plt.loglog(ks, 1/ks)
plt.show()