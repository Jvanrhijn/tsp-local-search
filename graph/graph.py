from os import read
import numpy as np
from itertools import product, groupby


class Graph:

    def __init__(self, vertices, edges, weights=None):
        self.vertices = vertices
        self.edges = edges
        self.positions = None

        if weights is None:
            self.weights = {e: 1 for e in edges}
        else:
            self.weights = weights

    @classmethod
    def fully_connected(cls, vertices, weights=None):
        edges = list(map(frozenset, filter(lambda e: e[0] != e[1], product(vertices, vertices))))
        return cls(vertices, edges, weights=weights)

    @classmethod
    def from_tspfile(cls, path):
        dists = read_instance(path)

        n = dists.shape[0]
        graph = cls.fully_connected(list(range(n)))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                graph.set_weight(dists[i, j], frozenset({i, j}))

        return graph

    def set_weight(self, w, e):
        if e not in self.edges:
            raise ValueError(f"Edge {e} not present in graph")
        self.weights[e] = w

    def place_vertices(self, positions):
        if not positions:
            return

        if set(positions.keys()) != set(self.vertices):
            raise ValueError("Not all vertices assigned a position")
        self.weights = {frozenset((a, b)): np.linalg.norm(positions[a] - positions[b]) for (a, b) in self.edges}
        self.positions = positions

    def remove_vertex(self, vertex):
        vertices = list(filter(vertex.__ne__, self.vertices))
        edges = list(filter(lambda e: vertex not in e, self.edges))
        weights = {e: self.weights[e] for e in edges}
        positions = None

        if self.positions:
            positions = {v: self.positions[v] for v in vertices}

        g = Graph(vertices, edges, weights=weights)
        g.place_vertices(positions)
        return g

def read_instance(path):
    distances = []

    with open(path, "r") as file:
        for line in file:
            split = line.split()
            distances.append(list(map(int, split)))
    
    return np.array(distances)