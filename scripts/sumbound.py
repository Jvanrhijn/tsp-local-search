import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

np.random.seed(0)

n = 4000
k = 3

deltas = []

def edges_cross(red1, red2):
    line1 = LineString(red1)
    line2 = LineString(red2)
    return line1.intersection(line2).geom_type == "Point"


def gain(reds, blues, points):
    red_weights = [np.linalg.norm(points[i] - points[j]) for (i, j) in reds]
    blue_weights = [np.linalg.norm(points[i] - points[j]) for (i, j) in blues]
    return sum(red_weights) - sum(blue_weights)


def decomposition(red_edges, blue_edges):
    vs = list(range(2*len(red_edges)))

    decomps = []

    for i in vs[:len(red_edges)]:

        u = vs[i]
        v = u + 3

        cycle_vs = list(range(u, v+1))    
        subcycle = list(zip(cycle_vs, cycle_vs[1:]))
    
        reds = set(red_edges).intersection(set(subcycle))
        blues = set(blue_edges).intersection(set(subcycle))

        if len(reds) < len(blues):
            reds = reds.union({(u, v)})
        else:
            blues = blues .union({(u, v)})

        decomps.append((reds, blues))

    return decomps


for _ in range(n):

    points = np.random.normal(size=(2*k, 2))

    odds = list(range(1, 2*k, 2))
    evens = list(range(0, 2*k, 2))

    red_edges = [(i, (i+1) % (2*k)) for i in odds]
    blue_edges = [(i, (i+1) % (2*k)) for i in evens]

    red_weights = [np.linalg.norm(points[i] - points[j]) for (i, j) in red_edges]
    blue_weights = [np.linalg.norm(points[i] - points[j]) for (i, j) in blue_edges]

    delta = gain(red_edges, blue_edges, points)

    d = decomposition(red_edges, blue_edges)

    gains = [gain(r, b, points) for r, b in d]

    if delta > 0:
        deltas.append(delta)
        print(gains)


epss = np.linspace(0, 0.5, 100)

ps = []

for e in epss:
    nsmall = sum(deltas <= e)
    ps.append(nsmall / len(deltas))


plt.figure()

plt.plot(epss, ps)
plt.plot(epss, epss)

plt.show()