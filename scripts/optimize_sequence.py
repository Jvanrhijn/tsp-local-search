from itertools import chain, product
import copy
from os import path

from numpy import add
from solve_tsp import *
from heuristics.two_opt import *
from heuristics.three_opt import *
from heuristics.lk import *
from collections import defaultdict
import networkx as nx
#import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout


fig, ax = plt.subplots(1)

nverts = np.arange(20, 60)
k = 15
num_tours = 40

dts = average_case_time(num_tours, nverts, two_opt_iteration, tour_generator=lambda n, rng_seed=None, kind="Euclidean": generate_ktree(n, k, kind=kind, rng_seed=rng_seed))

fig, ax = plt.subplots(1)
ax.plot(nverts, dts, 'o')
ax.grid()

plt.show()