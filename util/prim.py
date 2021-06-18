from collections import defaultdict
import heapq


def convert_graph(graph):
    vertices = graph.vertices
    weights = graph.weights

    outgraph = defaultdict(dict)

    for v in vertices:
        for edge in weights:
            if v in edge:
                outgraph[v][next(iter(edge - {v}))] = weights[edge]

    return outgraph



def create_spanning_tree(og_graph, starting_vertex):
    graph = convert_graph(og_graph)

    mst = defaultdict(set)
    visited = set([starting_vertex])
    edges = [
        (cost, starting_vertex, to)
        for to, cost in graph[starting_vertex].items()
    ]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst[frm].add(to)
            for to_next, cost in graph[to].items():
                if to_next not in visited:
                    heapq.heappush(edges, (cost, to, to_next))

    return mst, mst_cost(mst, og_graph.weights)


def mst_cost(mst, weights):
    cost = 0

    for u in mst:
        for child in mst[u]:
            cost += weights[frozenset((u, child))]
    
    return cost


def tsp_lower_bound(graph):
    bounds = []

    for v0 in graph.vertices:
        g = graph.remove_vertex(v0)

        mst, cost = create_spanning_tree(g, g.vertices[0])

        # find shortest connections from v0
        conns = sorted([graph.weights[frozenset((v0, u))] for u in g.vertices])

        bounds.append(cost + conns[0] + conns[1])

    return max(bounds)


