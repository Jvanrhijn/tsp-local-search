import math
from itertools import chain, product, combinations
from functools import reduce
import copy
import tqdm
from matplotlib.pyplot import get
import scipy.integrate as integ

from numpy import add
from solve_tsp import *
from util.kmeans import *
from util.util import *
from heuristics.two_opt import *
from heuristics.three_opt import *
from heuristics.lk import *
from heuristics.simulated_annealing import *
from collections import defaultdict
import networkx as nx
#import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout



np.random.seed(43532)


nvert = 7
#graph, tour = generate_random_tour(nvert, rng_seed=22, kind="Euclidean")
graph, tour = generate_random_tour(nvert, rng_seed=435, kind="Unit")
#graph = Graph.from_tspfile("instances/a280.tsp")

vs = np.random.permutation(graph.vertices)
nvert = len(vs)
tour = [frozenset({u, v}) for u, v in zip(vs, vs[1:])] + [frozenset({vs[-1], vs[0]})]
initial_tour = copy.deepcopy(tour)

lb = 0

# number of steps
nsteps = 1000



temperature = lambda t, a: a
#temperature = lambda t, a: a/(t+1)
#temperature = lambda t, a: a/np.log(t+3)


def downhill_neighborhood_size(tour, graph):
    l0 = path_length(tour, graph.weights)
    ndown = 0
    for i in range(len(tour)):
        for j in range(i+2, len(tour)-1):
            candidate = two_opt_swap(tour, i, j)
            l = path_length(candidate, graph.weights)
            ndown += l < l0
    return ndown


def run_sa(steps, tour, a):

    gain_uphill = []
    gain_downhill = []
    nup = [0]
    ndown = [0]
    lens = []


    best_so_far = np.inf

    for step in range(steps):

        new_tour = sa_iteration(tour, graph, temperature(step, a))

        length = path_length(new_tour, graph.weights)
        gain = path_length(tour, graph.weights) - length

        nup.append(nup[-1] + int(gain < -1e-10))
        ndown.append(ndown[-1] + int(gain > 1e-10))

        if gain < -1e-10:
            gain_uphill.append(abs(gain))
        if gain > 1e-10:
            gain_downhill.append(gain)

        best_so_far = min(best_so_far, length)

        tour = new_tour

        lens.append(length)

    return nup, ndown, gain_uphill, gain_downhill, lens


#temps = [0.1, 0.2, 0.5, 0.7, 1.0]
#temps = np.array(list(np.linspace(0.01, 1, 10)) + list(np.linspace(1.1, 10, 10)))
#temps = np.array(list(np.linspace(0.001, 0.1, 10)))
#temps = [1]
betas = np.linspace(1, 10, 10)
#temps = np.linspace(1/nvert, 10/nvert , 10)
temps = 1/betas

fraction_nup = []
lensss = []

for a in temps:

    nups = []
    ndowns = []
    nminuss = []
    lenss = []

    # run a couple of times with random tours
    for _ in tqdm.tqdm(range(40)):
        vs = np.random.permutation(graph.vertices)
        nvert = len(vs)
        tour = [frozenset({u, v}) for u, v in zip(vs, vs[1:])] + [frozenset({vs[-1], vs[0]})]

        nup, ndown, gain_uphill, gain_downhill, lens = run_sa(nsteps, tour, a)

        tup = np.arange(1, len(gain_uphill)+1)
        guph = np.array([sum(gain_uphill[:i]) for i in tup])

        td = np.arange(1, len(gain_downhill)+1)
        gd = np.array([sum(gain_downhill[:i]) for i in td])

        lenss.append(lens[-len(lens)//2:])

        nups.append(nup)
        ndowns.append(ndown)

    nups = np.array(nups)
    ndowns = np.array(ndowns)
    lenss = np.array(lenss)

    nups = np.mean(nups, axis=0)
    ndowns = np.mean(ndowns, axis=0)
    lensss.append(np.mean(lenss))


    fraction_nup.append(nups[-1] / nsteps)


betas = 1/temps
lens_z = nvert / betas + nvert / (np.exp(betas) - 1)

plt.figure()
plt.plot(temps, lensss, 'o')
plt.plot(temps, nvert / betas + abs(min(nvert/betas) - min(lensss)))
plt.show()