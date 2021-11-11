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
tmax = n**10
betamax = np.log(tmax)
nsamples = 10
zav_log = 0
betas, dbeta = np.linspace(0.1, betamax, 100, retstep=True)
zmean = 0.5 * factorial(n-1) * ((1 - np.exp(-betas)) / betas)**n
j_pth_central_moment = 0

p = 2
jmin = 0

for it in tqdm.tqdm(range(nsamples)):
    graph, _ = generate_random_tour(n, rng_seed=it, kind="Unit")

    tours = list(map(lambda t: [graph.vertices[0]] + list(t), permutations(graph.vertices[1:])))
    tour_lengths = list(map(lambda t: tour_length(t, graph.weights), tours))


    # div 2 for double counting
    z = np.array([sum(math.exp(-beta * j) for j in tour_lengths) for beta in betas]) / 2

    zav_log += np.log(z) / nsamples

    j_av = np.array([sum(j * math.exp(-beta*j) for j in tour_lengths) / z[idx] for idx, beta in enumerate(betas)]) / (2)
    j_pth_central_moment += np.array([
        sum((j - 0*j_av[idx])**p * math.exp(-beta*j) / z[idx] for j in tour_lengths) for idx, beta in enumerate(betas)
    ]) / (2 * nsamples)

    jmin += min(tour_lengths) / nsamples


zav_logderiv = (zav_log[1:] - zav_log[:-1]) / dbeta
zav_logderiv2 = (zav_logderiv[1:] - zav_logderiv[:-1]) / dbeta
zav_logderiv3 = (zav_logderiv2[1:] - zav_logderiv2[:-1]) / dbeta
zav_logderiv4 = (zav_logderiv3[1:] - zav_logderiv3[:-1]) / dbeta
zav_logderiv5 = (zav_logderiv4[1:] - zav_logderiv4[:-1]) / dbeta


zav_log_ub = np.log(0.5 * factorial(n-1) * ((1 - np.exp(-betas)) / betas)**n)
zav_log_ub_deriv = (zav_log_ub[1:] - zav_log_ub[:-1]) / dbeta
zav_log_ub_deriv2 = (zav_log_ub_deriv[1:] - zav_log_ub_deriv[:-1]) / dbeta
zav_log_ub_deriv3 = (zav_log_ub_deriv2[1:] - zav_log_ub_deriv2[:-1]) / dbeta
zav_log_ub_deriv4 = (zav_log_ub_deriv3[1:] - zav_log_ub_deriv3[:-1]) / dbeta
zav_log_ub_deriv5 = (zav_log_ub_deriv4[1:] - zav_log_ub_deriv4[:-1]) / dbeta

plt.figure()
plt.plot(betas[:-1], -zav_logderiv, linestyle="-", label=r"$-d_\beta \mathbb{E} (\ln Z)$")
plt.plot(betas, -(n / (np.exp(betas) - 1) - n / betas), label=r"$-d_\beta \ln \mathbb{E}(Z)$")
plt.xlabel(r"$\beta$")
plt.legend()

plt.figure()
plt.plot(betas[:-2], zav_logderiv2, linestyle="-", label=r"$d_\beta^2 \mathbb{E} (\ln Z)$")
plt.plot(betas, n* (1/betas**2 - 1 / (np.exp(betas) - 1) - 1 / (np.exp(betas) - 1)**2), linestyle="-", label=r"$d_\beta^2 (\ln \mathbb{E}(Z))$")
plt.xlabel(r"$\beta$")
plt.legend()

plt.figure()
plt.plot(betas[:], j_pth_central_moment[:])
#plt.plot(betas[:-p], n**p / betas[:-p]**p, label="UB")
s = n*(n-1)*(1/betas - 1/(np.exp(betas)-1))**2 + n * (2/betas * (1/betas - 1/(np.exp(betas)-1)) - 1/(np.exp(betas)-1))
plt.plot(betas, jmin**p + 2*s + 0*jmin*np.sqrt(s))
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\mu_p$")
plt.legend()

plt.show()