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


np.random.seed(35)


#nverts = [10, 20, 30]
nverts = range(10, 55, 5)



def steps_to_reach_2opt(tour, graph, steps=40000):

    #d = 100 / len(graph.vertices)**3
    d = 1
    repetitions = 0
    temperature = d


    for t in range(steps):
        #temperature = d / (np.log(t+2))
        #temperature = d / (t+1)
        #temperature = d / (np.log((t+2) * np.log(t + 2)))
        temperature = d / np.log(t+2)

        #new_tour = two_opt_iteration(tour, graph.weights)
        new_tour = sa_iteration(tour, graph, temperature, transition_function=boltzmann, pick_neighbor=pick_neighbor_2opt)
        #new_tour = lk_iteration(tour, graph.weights)

        gain = path_length(tour, graph.weights) - path_length(new_tour, graph.weights)
        if abs(gain) <= 1e-10:
            repetitions += 1

        is_2_optimal = (path_length(two_opt_iteration(new_tour, graph.weights), graph.weights) == path_length(tour, graph.weights))

        tour = new_tour

        if is_2_optimal:
            break
        
    return t+1# - repetitions


nsamples = 10

nsteps = []


for nvert in nverts:

    steps = 0

    print(nvert)

    for ns in tqdm.tqdm(range(nsamples)):

        graph, tour = generate_random_tour(nvert, kind="Unit", rng_seed=(nvert + ns))
        
        steps += steps_to_reach_2opt(tour, graph)

    nsteps.append(steps / nsamples)


plt.figure()
plt.loglog(nverts, nsteps, 'o', linestyle='-')
plt.grid()
plt.show()