import numpy as np
import math
import matplotlib.pyplot as plt
from util import *


def distance(u, v, weights):
    if u == v:
        return 0
    return weights[frozenset({u, v})]


def reverse_segment_if_better(tour, i, j, k, weights):
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    # Given tour [...A-B...C-D...E-F...]
    A, B, C, D, E, F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
    d0 = distance(A, B, weights) + distance(C, D, weights) + distance(E, F, weights)
    d1 = distance(A, C, weights) + distance(B, D, weights) + distance(E, F, weights)
    d2 = distance(A, B, weights) + distance(C, E, weights) + distance(D, F, weights)
    d3 = distance(A, D, weights) + distance(E, B, weights) + distance(C, F, weights)
    d4 = distance(F, B, weights) + distance(C, D, weights) + distance(E, A, weights)

   # if d0 > d1:
   #     tour[i:j] = reversed(tour[i:j])
   #     return -d0 + d1
   # elif d0 > d2:
   #     tour[j:k] = reversed(tour[j:k])
   #     return -d0 + d2
   # elif d0 > d4:
   #     tour[i:k] = reversed(tour[i:k])
   #     return -d0 + d4
    if d0 > d3:
        tmp = tour[j:k] + tour[i:j]
        tour[i:k] = tmp
        return -d0 + d3
    return 0


def all_segments(n: int):
    """Generate all segments combinations"""
    return ((i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n + (i > 0)))



def flatten_tour(tour):
    # flatten vertex list
    flat = [next(iter(tour[-1].intersection(tour[0])))]
    for i, e in enumerate(tour[1:]):
        f = tour[i]
        link = e.intersection(f)
        flat.append(next(iter(link)))

    flat.append(flat[0])
    return flat


def unflatten_tour(tour):
    return list(map(frozenset, zip(tour, tour[1:] + [tour[0]])))


def three_opt(p, a, c, e, broad=False):
    """In the broad sense, 3-opt means choosing any three edges ab, cd
    and ef and chopping them, and then reconnecting (such that the
    result is still a complete tour). There are eight ways of doing
    it. One is the identity, 3 are 2-opt moves (because either ab, cd,
    or ef is reconnected), and 4 are 3-opt moves (in the narrower
    sense)."""
    n = len(p)
    #a, c, e = random.sample(range(n+1), 3)
    # without loss of generality, sort
    a, c, e = sorted([a, c, e])
    b, d, f = a+1, c+1, e+1

    if broad == True:
        which = random.randint(0, 7) # allow any of the 8
    else:
        which = random.choice([3, 4, 5, 6]) # allow only strict 3-opt

    # in the following slices, the nodes abcdef are referred to by
    # name. x:y:-1 means step backwards. anything like c+1 or d-1
    # refers to c or d, but to include the item itself, we use the +1
    # or -1 in the slice
    #if which == 0:
    #    sol = p[:a+1] + p[b:c+1]    + p[d:e+1]    + p[f:] # identity
    #elif which == 1:
    #    sol = p[:a+1] + p[b:c+1]    + p[e:d-1:-1] + p[f:] # 2-opt
    #elif which == 2:
    #    sol = p[:a+1] + p[c:b-1:-1] + p[d:e+1]    + p[f:] # 2-opt

    sols = [
        p[:a+1] + p[c:b-1:-1] + p[e:d-1:-1] + p[f:],
        #p[:a+1] + p[d:e+1]    + p[b:c+1]    + p[f:],
        p[:a+1] + p[d:e+1]    + p[c:b-1:-1] + p[f:],
        p[:a+1] + p[e:d-1:-1] + p[b:c+1]    + p[f:]
    ]

    return sols


def three_opt_iteration_better(tour, weights):
    flat_tour = flatten_tour(tour)[:-1]
    og_tour_length = path_length(tour, weights)

    for (i, j, k) in all_segments(len(tour)):
        tours = list(map(unflatten_tour, three_opt(flat_tour, i, j, k)))

        lengths = np.array(list(map(lambda p: path_length(p, weights), tours)))

        shortest = lengths.argmin()

        new_tour = tours[shortest]
        
        if path_length(new_tour, weights) < og_tour_length:
            return new_tour

        #delta = reverse_segment_if_better(flat_tour, a, b, c, weights)
        #if delta < 0:
        #    return unflatten_tour(flat_tour)

    return tour
