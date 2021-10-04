import numpy as np
import copy

from util.util import *
from heuristics.greedy_tsp import greedy as greedy_tsp
from graph.graph import Graph
from util.prim import *


# prefer concorde
try:
    from concorde.tsp import TSPSolver
    CONCORDE_AVAILABLE = True
    #CONCORDE_AVAILABLE = False 
except ImportError:
    CONCORDE_AVAILABLE = False

if CONCORDE_AVAILABLE == False:
    try:
        from python_tsp.heuristics import solve_tsp_simulated_annealing as solve_tsp_lib
        SOLVER_AVAILABLE = False 
    except ImportError:
        SOLVER_AVAILABLE = False


def generate_random_tour(nvert, rng_seed=None, kind="Euclidean"):
    # set up tour
    gen = np.random.default_rng(seed=rng_seed)

    graph = Graph.fully_connected([f"v{i}" for i in range(nvert)])
    
    if kind == "Euclidean":
        # randomly place vertices in the unit square
        positions = {v: gen.uniform(size=2) for v in graph.vertices}
        graph.place_vertices(positions)

    elif kind == "Unit":
        graph.weights = {e: gen.uniform(low=0, high=1) for e in graph.edges}
        # symmetrize
        #for u, v in graph.edges:
        #    graph.weights[frozenset((u, v))] = graph.weights[frozenset((v, u))]
    
    # randomly generate initial tour
    tour = copy.deepcopy(graph.vertices)
    tour = list(map(frozenset, zip(tour, tour[1:] + [tour[0]])))

    return graph, tour


def generate_ktree(nvert, k, rng_seed=None, kind="Euclidean"):
    gen = np.random.RandomState(seed=rng_seed)

    assert k <= nvert + 1

    base_vertices = [f"v{i}" for i in range(k+1)]
    # identify the subset of vertices to which all added
    # vertices will be neighbors
    subset = base_vertices[:k-2]

    graph = Graph.fully_connected(copy.deepcopy(base_vertices))
    
    for i in range(k+1, nvert):
        # add new vertex
        graph.vertices.append(f"v{i}")

        # add edges to create a k-tree
        # for each new vertex v, connect it to exactly k-2 vertices S 
        # from the base graph in such a way that S and {v} together
        # form a clique.

        # connect each new vertex to exactly k-2 of the base vertices
        graph.edges += [frozenset({f"v{i}", v}) for v in subset]

    # finally, connect all added vertices in a cycle
    added_vs = [f"v{i}" for i in range(k+1, nvert)]
    cycle = copy.deepcopy(added_vs)

    graph.edges += [frozenset({u, v}) for u, v in zip(cycle, cycle[1:])] + [frozenset({cycle[-1], cycle[0]})]
    
    # a hamiltonian cycle is given by the outer cycle excepting one edge, plus an edge to v0, plus a hamiltonian
    # path through the original graph ending at any vertex of the neighbor subset
    tour = cycle

    start = subset[0]
    end = subset[1]

    tour.append(start)
    remaining = subset[2:]
    tour += remaining
    tour += base_vertices[k-2:]
    tour.append(end)

    tour = [frozenset({u, v}) for u, v in zip(tour, tour[1:])] + [frozenset({tour[-1], tour[0]})]

    if kind == "Unit":
        for edge in graph.edges:
            graph.set_weight(gen.uniform(low=0, high=1), edge)
    elif kind == "Euclidean":
        positions = {v: gen.uniform(size=2) for v in graph.vertices}
        graph.place_vertices(positions)
    else:
        raise ValueError("Weight distribution not supported")

    return graph, tour


def solve_tsp(graph, tour, optimizer, it_max=1000): #tour, weights, it_max=1000):

    for i in range(it_max):

        new_tour = optimizer(tour, graph.weights)

        if path_length(tour, graph.weights) <= path_length(new_tour, graph.weights):
            break
        else:
            tour = new_tour

    return new_tour, i + 1


def solve_ensemble(optimizer, num_tours, nvert, seeds=None, greedy=False, compute_lb=True, kind="Euclidean", tour_generator=generate_random_tour):

    if not seeds:
        seeds = [None]*num_tours

    len_original = np.zeros(num_tours)
    len_optimal = np.zeros(num_tours)
    dts = np.zeros(num_tours)
    lower_bounds = np.zeros(num_tours)

    for i in range(num_tours):
        graph, tour = tour_generator(nvert, rng_seed=seeds[i], kind=kind)
        
        if compute_lb:
            if CONCORDE_AVAILABLE:
                lb = path_length(tsp_concorde(graph), graph.weights)
            elif SOLVER_AVAILABLE:
                lb = path_length(tsp_solver(graph), graph.weights)
            else:
                lb = tsp_lower_bound(graph)
        else:
            lb = np.inf

        if greedy:
            tour = greedy_tsp(graph.vertices, graph.weights)

        len_original[i] = path_length(tour, graph.weights)

        optimal_tour, dt = solve_tsp(graph, tour, optimizer)
        len_optimal[i] = path_length(optimal_tour, graph.weights)
        lower_bounds[i] = lb

        dts[i] = dt

    return len_original, len_optimal, dts, lower_bounds


def average_case_time(num_tours, nverts, optimizer, greedy=False, kind="Euclidean", tour_generator=generate_random_tour):
    dts = np.zeros(nverts.shape)

    for i, nv in enumerate(nverts):
        seeds = list(range(i*num_tours, (i+1)*num_tours))
        len_orig, len_optimal, dt, lower_bounds \
            = solve_ensemble(optimizer, num_tours, nv, seeds=seeds, greedy=greedy, compute_lb=False, kind=kind, tour_generator=tour_generator)
        dts[i] = np.mean(dt)

    return dts


def average_case_ratio(num_tours, nverts, optimizer, greedy=False, kind="Euclidean"):
    ratios = np.zeros(nverts.shape)

    for i, nv in enumerate(nverts):
        seeds = list(range(i*num_tours, (i+1)*num_tours))
        len_orig, len_optimal, dt, lbs = solve_ensemble(optimizer, num_tours, nv, seeds=seeds, greedy=greedy, kind=kind)
        ratios[i] = np.mean(len_optimal / lbs)

    return ratios


def tsp_concorde(graph):
    # solve with concorde
    pos = np.array(list(graph.positions.values()))
    xs = pos[:, 0]
    ys = pos[:, 1]

    solver = TSPSolver.from_data(xs, ys, norm="GEO")
    sol = solver.solve(verbose=False)

    optimal_tour = [graph.vertices[i] for i in sol.tour]
    return list(map(frozenset, list(zip(optimal_tour, optimal_tour[1:])) + [(optimal_tour[-1], optimal_tour[0])]))


def tsp_solver(graph):
    distance_matrix = np.zeros((len(graph.vertices), len(graph.vertices)))
    for i, u in enumerate(graph.vertices):
        for j, v in enumerate(graph.vertices):
            if i == j:
                distance_matrix[i, i] == 0
            else:
                distance_matrix[i, j] = graph.weights[frozenset({u, v})]
    
    perm, distance = solve_tsp_lib(distance_matrix)

    tour = [graph.vertices[i] for i in perm]
    tour = [frozenset({u, v}) for u, v in zip(tour, tour[1:])] + [frozenset({tour[-1], tour[0]})]
    return tour