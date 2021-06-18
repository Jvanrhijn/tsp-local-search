import numpy as np
import math
from util.util import *


def two_opt_swap(tour, i, k):
    assert k > i

    p = remove_edge_from_tour(tour, i)
    pnew = break_and_reconnect_path(p, k)
    return close_path(pnew)


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