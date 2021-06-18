import numpy as np
from util.util import *
from heuristics.two_opt import two_opt_iteration


def all_segments(n: int):
    """Generate all segments combinations"""
    return ((i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n + (i > 0)))


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


def pure_three_opt_iteration(tour, weights):
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


def three_opt_iteration(tour, weights):
    og_tour_length = path_length(tour, weights)

    # try to do 2-opt first
    two_opt_tour = two_opt_iteration(tour, weights)
    if path_length(two_opt_tour, weights) < og_tour_length:
        return two_opt_tour

    # if 2-opt fails to improve, do pure 3-opt
    return pure_three_opt_iteration(tour, weights)
