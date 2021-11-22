import math
from scipy.stats import kstat
from itertools import chain, product, combinations
from functools import reduce
import copy
import tqdm
from matplotlib.pyplot import get
import scipy.integrate as integ

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
graph, tour = generate_random_tour(nvert, rng_seed=435, kind="Unit")
#graph = Graph.from_tspfile("instances/a280.tsp")

vs = np.random.permutation(graph.vertices)
nvert = len(vs)
tour = [frozenset({u, v}) for u, v in zip(vs, vs[1:])] + [frozenset({vs[-1], vs[0]})]
initial_tour = copy.deepcopy(tour)

lb = 0

# number of steps
nsteps = 100

# number of steps to stay at the same temperature
neq = 1


a = 1
#temperature = lambda t: a / (np.log(t + 2))**np.log(np.log((t + 2)))
#temperature = lambda t: a / (t + 2)
temperature = lambda t: a / np.log(t // neq + 2)
#temperature = lambda t: a


def run_sa(steps, tour):

    lens = []
    gain_uphill = []
    gain_downhill = []
    nup = [0]
    ndown = [0]


    best_so_far = np.inf

    for step in range(steps):

        #new_tour = sa_iteration(tour, graph, temperature(step), lazy=True)
        new_tour = sa_iteration(tour, graph, temperature(step), lazy=True, pick_neighbor=pick_neighbor_3opt)

        length = path_length(new_tour, graph.weights)
        gain = path_length(tour, graph.weights) - length

        nup.append(nup[-1] + int(gain < -1e-10))
        ndown.append(ndown[-1] + int(gain > 1e-10))

        if gain < -1e-10:
            gain_uphill.append(abs(gain))
            #gain_downhill.append(0)
        if gain > 1e-10:
            gain_downhill.append(gain)
            #gain_uphill.append(0)
        #else:
        #    gain_downhill.append(0)
        #    gain_uphill.append(0)


        best_so_far = min(best_so_far, length)
        lens.append(length)

        #is_2_optimal = abs(path_length(two_opt_iteration(new_tour, graph.weights), graph.weights) - path_length(tour, graph.weights)) < 1e-10

        tour = new_tour

        final_weights = [graph.weights[e] for e in tour]

        #if is_2_optimal:
        #    break

    return nup, ndown, gain_uphill, gain_downhill, lens, final_weights


uphills = []
downhills = []
nups = []
ndowns = []
lenss = []
weightss = []

nsamples = 1000
# run a couple of times with random tours
for i in tqdm.tqdm(range(nsamples)):
    #vs = np.random.permutation(graph.vertices)
    #nvert = len(vs)
    #tour = [frozenset({u, v}) for u, v in zip(vs, vs[1:])] + [frozenset({vs[-1], vs[0]})]
    graph, tour = generate_random_tour(nvert, rng_seed=0, kind="Unit")
    vs = np.random.permutation(graph.vertices)
    nvert = len(vs)
    tour = [frozenset({u, v}) for u, v in zip(vs, vs[1:])] + [frozenset({vs[-1], vs[0]})]
    initial_tour = copy.deepcopy(tour)

    nup, ndown, gain_uphill, gain_downhill, lens, weights = run_sa(nsteps, tour)

    tup = np.arange(1, len(gain_uphill)+1)
    guph = np.array([sum(gain_uphill[:i]) for i in tup])

    td = np.arange(1, len(gain_downhill)+1)
    gd = np.array([sum(gain_downhill[:i]) for i in td])

    uphills.append(guph)
    downhills.append(gd)
    
    nups.append(nup)
    ndowns.append(ndown)

    lenss.append(lens)

    weightss.append(weights)

nuphill = min([len(u) for u in uphills])
uphills = np.array([u[:nuphill] for u in uphills])
weightss = np.array(weightss)

nups = np.array(nups)
ndowns = np.array(ndowns)

uphill_mean = np.mean(uphills, axis=0)

nups = np.mean(nups, axis=0)
ndowns = np.mean(ndowns, axis=0)

ndhill = min([len(u) for u in downhills])
downhills = np.array([u[:ndhill] for u in downhills])

downhill_mean = np.mean(downhills, axis=0)

lenss_mean = np.mean(np.array(lenss), axis=0)
lenss_var = np.var(np.array(lenss), axis=0)


tup = np.arange(1, len(uphill_mean)+1)
td = np.arange(1, len(downhill_mean)+1)

w = weightss.reshape((1, weightss.size))

#uph_bound = integ.cumtrapz(temperature(tup), x=tup, initial=0)


ts = np.arange(1, len(lenss_mean)+1)
temps = temperature(ts)

fig, ax = plt.subplots(1, ncols=2)
ax[0].plot(ts[10:], lenss_mean[10:], label="Mean tour length")
#plt.plot(ts, nvert * (temps - 1 / (np.exp(1/temps) - 1)), label=r"$\Theta(an/\log(t))$")
ax[0].plot(ts[10:], nvert * (temps[10:]), label=r"$\Theta(an/\log(t))$")
ax[0].set_ylim(0)
ax[0].set_xlabel("t")
ax[0].set_ylabel("Tour length")
ax[0].legend()

ax[1].plot(ts[10:], lenss_var[10:], label="Tour variance")
#ax[1].plot(ts[10:], 1*nvert * (temps[10:]**2), label=r"$\Theta(a^2n/\log^2(t))$")
ax[1].plot(ts[10:], 1*nvert * (temps[10:]**2), label=r"$\Theta(a^2n/\log^2(t))$")
ax[1].set_ylim(0)
ax[1].set_xlabel("t")
ax[1].set_ylabel("Tour variance")
ax[1].legend()

plt.tight_layout()


plt.figure()
js = np.array(lenss)[:, -1]

delta = 0.1
mu = js.mean()
jsmall = js[js <= (1-delta)*mu]
psmall = len(jsmall) / len(js)

print(f"psmall = {psmall:.10f}    chernoff: {math.exp(-delta**2*mu/2):.10f}")

w0 = weightss[:, :35].sum(axis=1)
w1 = weightss[:, 35:].sum(axis=1)

print(np.mean(w0*w1))
print(np.mean(w0)*np.mean(w1))

plt.hist(w.T, bins=20)
#plt.axvline(1.5*nvert*temperature(nsteps), color='k')
#plt.axvline(js.mean(), color='k', linestyle='dashed', linewidth=1)


plt.show()

#plt.figure()
#plt.title("uphill")
#plt.plot(tup, uphill_mean, label="Expected uphill gain")
#plt.plot(tup, 1.1*tup / np.log(tup+2), label=r"$\Theta(t/\log(t))$")
#plt.plot(tup, a * tup, label=r"$\Theta(t)$")
##plt.plot(tup, uph_bound)
#plt.xlabel("t"); plt.ylabel("Gain")
#plt.legend()
#
#plt.figure()
#plt.title("downhill")
#plt.plot(td, downhill_mean)
#plt.plot(td, td / np.log(td+2))
##plt.plot(tdown, tdown / np.log(tdown))
#
#ts = np.arange(1, nsteps + 2)
#
#xs = np.arange(1, 10*nsteps +2)
#
#plt.figure()
##plt.plot(ts, nups / ts**0.8)
#
#exp = (np.log(nups[-1] - np.log(nups[3*len(nups)//4])) / (np.log(ts[-1] - np.log(ts[3*len(ts)//4]))))
#
#plt.plot(ts, nups, label=r"$T_+$")
#plt.plot(ts, ts**(exp + 0.01), label=f"$\Theta$(t^{exp + 0.01:.3f})")
##plt.loglog(xs, xs / np.log(xs+2)**3)
##plt.plot(ts, 5*np.sqrt(ts))
#plt.xlabel("t")
#plt.legend()
#
#plt.show()