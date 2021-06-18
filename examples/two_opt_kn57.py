"""Approximates kn57 from the instances folder using 2-opt"""

from os import path
from util.util import path_length, unflatten_tour
from heuristics.two_opt import two_opt_iteration
from heuristics.three_opt import three_opt_iteration
from heuristics.lk import lk_iteration
from solve_tsp import solve_tsp
from graph.graph import Graph


g = Graph.from_tspfile("instances/kn57.tsp")
tour = unflatten_tour(g.vertices)

init_length = path_length(tour, g.weights)

algs = ["2-opt", "3-opt", "Lin-Kernighan"]

print(f"Initial length: {init_length}")

for i, algorithm in enumerate([two_opt_iteration, three_opt_iteration, lk_iteration]):
    
    optimal_tour, iters = solve_tsp(g, tour, algorithm)
    
    final_length = path_length(optimal_tour, g.weights)
    
    print(algs[i])
    print(f"Final length:   {final_length}")

