from heuristics.two_opt import two_opt_swap
from heuristics.three_opt import three_opt
import math
import numpy as np
from util.util import *


def generate_neighbors(tour, graph):
    weights = graph.weights
    edges = set(weights.keys())
    neighbors = []

    for i in range(len(tour)):

        p = remove_edge_from_tour(tour, i)

        for j in range(i+1, len(tour)-2):

            pnew = break_and_reconnect_path(p, j)

            tour_new = close_path(pnew)

            if not tour_valid(tour_new, edges):
                continue

            neighbors.append(tour_new)

    return neighbors



def pick_neighbor_2opt(tour, graph):
    n = len(graph.vertices)
    i = np.random.choice(list(range(n-1)))
    j = np.random.choice(list(set(range(n-1)) - {i, (i-1) % n, (i+1) % n}))
    return two_opt_swap(tour, i, j)


def pick_neighbor_3opt(tour, graph):
    i, j, k = np.random.choice(list(range(len(tour))), size=3, replace=False)

    flat_tour = flatten_tour(tour)
    candidates = three_opt(flat_tour, i, j, k)
    choice =  np.random.choice(list(range(len(candidates))))
    candidate = unflatten_tour(candidates[choice])[:-1]

    return candidate


def boltzmann(temperature, j1, j0):
        return math.exp(-1/temperature * (max(0, j1 - j0)))


def fermi(temperature, j1, j0):
    return 2 / (1 + math.exp(1/temperature * max(0, j1 - j0)))


def transition(initial, graph, temperature, pick_neighbor=pick_neighbor_2opt, transition_function=boltzmann):

    length = path_length(initial, graph.weights)
    # pick any neighbor at random
    candidate = pick_neighbor(initial, graph)
    cost = path_length(candidate, graph.weights)

    # decide transition

    # if the candidate is better than the current solution:
    if cost < length:
        return candidate

    # otherwise, only transition according to
    # Boltzmann factor
    #transition_prob = math.exp(-1/temperature * (max(0, cost - length)))
    transition_prob = transition_function(temperature, cost, length)

    r = np.random.uniform()
    if transition_prob > r:
        return candidate
    else:
        return initial


def sa_iteration(initial, graph, temperature, pick_neighbor=pick_neighbor_2opt, transition_function=boltzmann):
    return transition(initial, graph, temperature, pick_neighbor=pick_neighbor, transition_function=transition_function)