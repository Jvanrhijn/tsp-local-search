import numpy as np
import tqdm
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.special import factorial
import math

from solve_tsp import *


def tour_length(tour, weights):
    edges = map(frozenset, zip(tour, tour[1:] + [tour[0]]))
    return sum([weights[e] for e in edges])



n = 6
nsamples = 400
zav_log = 0
betas, dbeta = np.linspace(0.01, n, 100, retstep=True)
zmean = 0.5 * factorial(n-1) * ((1 - np.exp(-betas)) / betas)**n

for it in tqdm.tqdm(range(nsamples)):
    graph, _ = generate_random_tour(n, rng_seed=it, kind="Unit")

    tours = list(map(lambda t: [graph.vertices[0]] + list(t), permutations(graph.vertices[1:])))
    tour_lengths = list(map(lambda t: tour_length(t, graph.weights), tours))


    # div 2 for double counting
    z = np.array([sum(math.exp(-beta * j) for j in tour_lengths) for beta in betas]) / 2


    zav_log += np.log(z) / nsamples


zav_logderiv = (zav_log[1:] - zav_log[:-1]) / dbeta
zav_logderiv2 = (zav_logderiv[1:] - zav_logderiv[:-1]) / dbeta


zav_log_ub = np.log(0.5 * factorial(n-1) * ((1 - np.exp(-betas)) / betas)**n)
zav_log_ub_deriv = (zav_log_ub[1:] - zav_log_ub[:-1]) / dbeta
zav_log_ub_deriv2 = (zav_log_ub_deriv[1:] - zav_log_ub_deriv[:-1]) / dbeta

plt.figure()
plt.plot(betas[:-1], -zav_logderiv, linestyle="-", label=r"$-d_\beta \mathbb{E} (\ln Z)$")
plt.plot(betas, -(n / (np.exp(betas) - 1) - n / betas), label=r"$-d_\beta \ln \mathbb{E}(Z)$")
plt.xlabel(r"$\beta$")
plt.legend()

plt.figure()
plt.plot(betas[:-2], zav_logderiv2, linestyle="-", label=r"$d_\beta^2 \mathbb{E} (\ln Z)$")
plt.plot(betas, n* (1/betas**2 - 1 / (np.exp(betas) - 1) - 1 / (np.exp(betas) - 1)**2), linestyle="-", label=r"$d_\beta^2 (\ln \mathbb{E}(Z))$")
plt.xlabel(r"$T$")
plt.legend()
plt.show()
plt.show()