import numpy as np

from util.util import *


def optimize_path_greedy(path, weights, depth, tour_length, kmax=np.inf, restricted=None):

    if restricted is None:
        restricted = set()

    if depth >= kmax:
        return path

    pl = path_length(path, weights)

    for i in range(len(path)):
        e = path[i]

        if head(e, path[(i+1) % len(path)]) not in restricted:
            new_path = break_and_reconnect_path(path, i)
        else:
            continue

        if path_length(new_path, weights) < pl:

            new_tour = close_path(new_path)

            if path_length(new_tour, weights) < tour_length:
                return new_path

            else:
                succ = new_path[(i + 1) % len(path)]
                x = head(e, succ)
                return optimize_path(new_path, weights, depth+1, tour_length, kmax=kmax, restricted=restricted.union({x}))

    return path


def optimize_path(path, weights, depth, tour_length, kmax=np.inf, restricted=None):

    if restricted is None:
        restricted = set()

    if depth + 1 >= kmax:
        return path

    pl = path_length(path, weights)

    paths = []
    path_lengths = []

    for i in range(1, len(path)-1):
        e = path[i]

        if head(e, path[i+1]) not in restricted:
            new_path = break_and_reconnect_path(path, i)
        else:
            new_path = path

        paths.append(new_path)
        path_lengths.append(path_length(new_path, weights))

    best_path = np.array(path_lengths).argmin()
    best_edge = path[best_path + 1]
    new_path = paths[best_path]
    new_tour = close_path(paths[best_path])

    if path_length(new_tour, weights) < tour_length:
        return paths[best_path]

    elif path_lengths[best_path] < pl:
        succ = path[best_path + 2]
        x = head(best_edge, succ)

        return optimize_path(paths[best_path], weights, depth+1, tour_length, kmax=kmax, restricted=restricted.union({x}))

    else:
        return path


def lk_iteration(tour, weights, kmax=np.inf, greedy=False):

    edges = set(weights.keys())

    # hacky: perform LK for both the clockwise
    # and counterclockwise tour

    forward = lk_iteration_pre(tour, weights, kmax=kmax, greedy=greedy)
    backward = lk_iteration_pre(invert_path(tour), weights, kmax=kmax, greedy=greedy)

    forward_len = path_length(forward, weights) if tour_valid(forward, edges) else np.inf
    backward_len = path_length(backward, weights) if tour_valid(backward, edges) else np.inf

    return forward if forward_len < backward_len else backward


def lk_iteration_pre(tour, weights, kmax=np.inf, greedy=False):

    tour_length = path_length(tour, weights)

    for i, e in enumerate(tour):

        path = remove_edge_from_tour(tour, i)

        if greedy:
            optimal_path = optimize_path_greedy(path, weights, 0, tour_length, kmax=kmax)
        else:
            optimal_path = optimize_path(path, weights, 0, tour_length, kmax=kmax)

        new_tour = close_path(optimal_path)
        
        # check if new tour has a shorter length than the
        # starting tour
        new_tour_length = path_length(new_tour, weights)

        if new_tour_length < tour_length:
            return new_tour

    return tour

