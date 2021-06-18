import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product, groupby
import random
import copy
import networkx as nx


def invert_path(path):
    # path is a list of edges
    #return list(map(lambda x: x[::-1], path[::-1]))
    return path[::-1]


def invert_ordered_path(path):
    return list(map(lambda x: x[::-1], path[::-1]))


def compute_distance(tour):
    distance = 0
    for x1, x2 in zip(tour, tour[1:]):
        distance += np.linalg.norm(x2 - x1)**2
    distance += np.linalg.norm(tour[0] - tour[-1])**2
    return distance


def plot_tour(ax, tour, graph, alpha=1, color="black"):

    for edge in tour:
        u, v = edge
        xu = graph.positions[u]
        xv = graph.positions[v]
        ax.plot(*zip(xu, xv), alpha=alpha, color=color)
        ax.plot([xu[0]], [xu[1]], color="red", marker="o")
        ax.plot([xv[0]], [xv[1]], color="red", marker="o")


def mark_edge(ax, edge, positions, *args, **kwargs):
    v0, v1 = edge
    x1 = positions[v0]
    x2 = positions[v1]
    ax.plot([x1[0], x2[0]], [x1[1], x2[1]], *args, **kwargs)


def path_length(edges, weights):
    return sum(weights[e] for e in edges)


def get_positions(tour, positions):
    tour_vertices = [x[0] for x in groupby(list(sum(tour, ())))]
    return [positions[v] for v in tour_vertices]


def perturb_points(points, sigma=1):
    n = len(points)
    return [p + sigma*np.random.normal(size=p.shape, scale=sigma) for p in points]


def nearest_neighbor(tour):
    node = tour[0]
    new_tour = [node]
    for i in range(1, len(tour)):
        distances = np.array([np.linalg.norm(node - n) for n in tour[i:]])
        nn = distances.argmin()
        new_tour.append(tour[nn+1])
        node = new_tour[-1]
    return new_tour


def draw_tour(vertices, edges, tour, weights):

    graph = nx.Graph()
    
    for e in edges:
        graph.add_edge(*e, weight=weights[e])
    
    pos = nx.spring_layout(graph, seed=7)
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="red")

    # draw edges in tour
    #nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color="black")
    nx.draw_networkx_edges(graph, pos, edgelist=tour, edge_color="red")

    nx.draw_networkx_labels(graph, pos, font_size=10)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def head(e0, e1):
    # e1 succeeds e0 in a path
    return next(iter(e0 - e0.intersection(e1)))


def tail(e0, e1):
    # e1 succeeds e0 in a path
    return next(iter(e0.intersection(e1)))


def remove_edge_from_tour(tour, i):
    p1 = tour[:i]
    p2 = tour[i+1:]
    return p2 + p1


def break_and_reconnect_path(path, i):
    edge = path[i]

    # hacky but works
    x = head(edge, path[(i+1) % len(path)])
    y = head(path[(i+1) % len(path)], edge)
    b = head(path[0], path[1])
    e = head(path[-1], path[-2])

    if i > 0 and i < len(path)-1:
        new_path = path[:i] + [frozenset((x, e))] + invert_path(path[i+1:])
    else:
        new_path = path

    return new_path


def close_path(path):
    # reattach to close tour
    first = head(path[0], path[1])
    last = head(path[-1], path[-2])
    return path + [frozenset((first, last))]


def ordered_tour(tour):
    temp_otour = list(map(tuple, tour))
    otour = []

    for i, edge in enumerate(temp_otour[1:]):
        if temp_otour[i][1] not in edge:
            otour.append(tuple(reversed(temp_otour[i])))
        else:
            otour.append(temp_otour[i])
    
    otour.append(temp_otour[-1] if temp_otour[-1][0] == temp_otour[-2][1] else tuple(reversed(temp_otour[-1])))

    return otour