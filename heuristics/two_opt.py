import numpy as np
import math
import matplotlib.pyplot as plt
from util import *


def rot(theta):
    return np.array(
            [
                [math.cos(theta), -math.sin(theta)], 
                [math.sin(theta), math.cos(theta)]
            ]
    )


def construct_worst_case_instance(n):
    dx = np.array([-1.2, 0.1])
    initial_gadget = [
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([-0.1, 1.4]),
            np.array([-1.1, 4.8])
    ]
    gadgets = [initial_gadget]
    points = initial_gadget

    for i in range(n-1):
        g = gadgets[i]
        next_gadget = [dx + 3*rot(3*math.pi/2) @ x for x in g]
        gadgets.append(next_gadget)
        points = points + next_gadget

    return points


def two_opt_swap(tour, i, k):
    assert k > i

    #p1 = tour[k+1:] + tour[:i] 
    #p2 = tour[i+1:k]

    p = remove_edge_from_tour(tour, i)
    pnew = break_and_reconnect_path(p, k)
    return close_path(pnew)


def three_opt_swap(tour, i, k, l, s=1):
    assert k > i
    assert l > k

    # swap edge i and edge k
    p1 = tour[l+1:] + tour[:i]
    p2 = tour[i+1:k]
    p3 = tour[k+1:l]

    # if any sub path is empty,
    # i, k or j were adjacent -> return the original
    # tour, since swapping does nothing
    if not p1 or not p2 or not p3:
        return tour

    if s == 1:
        ip2 = invert_ordered_path(p2)
        ip3 = invert_ordered_path(p3)

        # edges to insert between p1, p2:
        e1 = (p1[-1][1], ip2[0][0])
        e2 = (ip2[-1][1], ip3[0][0])
        e3 = (ip3[-1][1], p1[0][0])

        return p1 + [e1] + ip2 + [e2] + ip3 + [e3]

    elif s == 2: 
        e1 = (p1[-1][1], p3[0][0])
        e2 = (p3[-1][1], p2[0][0])
        e3 = (p2[-1][1], p1[0][0])

        return p1 + [e1] + p3 + [e2] + p2 + [e3]


def flatten_tour(tour):
    # flatten vertex list
    tour = list(sum(tour, ()))
    # remove consecutive doubles
    return [x[0] for x in groupby(tour)]


def unflatten_tour(tour):
    return list(zip(tour, tour[1:] + [tour[0]]))


def two_opt_iteration(tour, weights):

    length = path_length(tour, weights)

    for i in range(len(tour)):

        p = remove_edge_from_tour(tour, i)

        for j in range(i+1, len(tour)-2):

            pnew = break_and_reconnect_path(p, j)

            tour_new = close_path(pnew)

            if path_length(tour_new, weights) < length:
                return tour_new

    return tour


def three_opt_iteration(tour, weights):

    length = path_length(tour, weights)

    # first try 2-opt equivalent moves
    #new_tour = two_opt_iteration(tour, weights)
    
    #if path_length(new_tour, weights) < length:
    #    return new_tour

    # if 2-opt doesn't yield improvement, try
    # pure 3-opt

    # convert tour to ordered tour for laziness
    tour = ordered_tour(tour)
    unordered_tour = lambda t: list(map(frozenset, t))


    for i in range(1, len(tour)-2):
        for k in range(i+2, len(tour)-2):
            for l in range(k+2, len(tour)-2):

                # try both type 1 and type 2 moves
                new_tour = three_opt_swap(tour, i, k, l, s=1)

                new_distance = path_length(unordered_tour(new_tour), weights)

                if new_distance < length:
                    return unordered_tour(new_tour)
                    
                else:
                    new_tour = three_opt_swap(tour, i, k, l, s=2)
                    new_distance = path_length(unordered_tour(new_tour), weights)

                    if new_distance < length:
                        return unordered_tour(new_tour)

    return unordered_tour(tour)