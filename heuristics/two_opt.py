from util.util import *
from util.util import remove_edge_from_tour


def two_opt_swap(tour, i, k):
    if k < i:
        k, i = i, k

    p = remove_edge_from_tour(tour, i)
    pnew = break_and_reconnect_path(p, k)

    return close_path(pnew)


def two_opt_iteration_general(tour, weights, comp=lambda x, y: x < y):

    edges = set(weights.keys())

    length = path_length(tour, weights)

    for i in range(len(tour)):

        p = remove_edge_from_tour(tour, i)

        for j in range(i+1, len(tour)-2):

            pnew = break_and_reconnect_path(p, j)

            tour_new = close_path(pnew)

            if not tour_valid(tour_new, edges):
                continue

            #if path_length(tour_new, weights) < length:
            if comp(path_length(tour_new, weights), length):
                return tour_new

    return tour


def two_opt_iteration(tour, weights):
    return two_opt_iteration_general(tour, weights)


def two_opt_iteration_reverse(tour, weights):
    return two_opt_iteration_general(tour, weights, comp=lambda x, y: x >= y)