import numpy as np
from util.util import *


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


def three_opt(p, a, c, e):
    """In the broad sense, 3-opt means choosing any three edges ab, cd
    and ef and chopping them, and then reconnecting (such that the
    result is still a complete tour). There are eight ways of doing
    it. One is the identity, 3 are 2-opt moves (because either ab, cd,
    or ef is reconnected), and 4 are 3-opt moves (in the narrower
    sense)."""
    # without loss of generality, sort
    a, c, e = sorted([a, c, e])
    b, d, f = a+1, c+1, e+1

    sols = [
        p[:a+1] + p[c:b-1:-1] + p[e:d-1:-1] + p[f:],
        p[:a+1] + p[d:e+1]    + p[b:c+1]    + p[f:], 
        p[:a+1] + p[d:e+1]    + p[c:b-1:-1] + p[f:],
        p[:a+1] + p[e:d-1:-1] + p[b:c+1]    + p[f:]
    ]

    return sols


def three_opt_iteration(tour, weights):
    flat_tour = flatten_tour(tour)[:-1]
    og_tour_length = path_length(tour, weights)

    for (i, j, k) in all_segments(len(tour)):
        tours = list(map(unflatten_tour, three_opt(flat_tour, i, j, k)))

        lengths = np.array(list(map(lambda p: path_length(p, weights), tours)))

        shortest = lengths.argmin()

        new_tour = tours[shortest]
        
        if path_length(new_tour, weights) < og_tour_length:
            return new_tour

    return tour
