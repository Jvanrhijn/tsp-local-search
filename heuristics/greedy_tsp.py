import numpy as np
import math
import matplotlib.pyplot as plt
from util import *


def greedy(vertices, weights):
    # start from initial vertex

    tours = []
    
    for v in vertices:

        vs = set(vertices)
        seen = {v}
        tour = []

        while len(tour) < len(vertices)-1:
            distances = {u: weights[(u, v)] for u in vs - seen if u != v}
            u = min(distances, key=distances.get)
            tour.append((v, u)) 
            v = u
            seen.add(u)

        # close tour
        tour.append((tour[-1][1], tour[0][0]))

        tours.append(tour)

    best = np.array([path_length(t, weights) for t in tours]).argmin()

    return tours[best]
