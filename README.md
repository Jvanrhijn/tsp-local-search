# TSP Local Search
This repository contains simple implementations of some local search algorithms for TSP. It is solely meant for research purposes, and *not* meant to be used for any practical purposes.

### Heuristics

* greedy
* 2-opt
* 3-opt
* pure 3-opt
* Lin-Kernighan

#### Greedy
The simplest TSP heuristic, also known as nearest-neighbor. This heuristic starts at an arbitrary vertex, and chooses the closest-by neighbor as its successor. This continues until all vertices have been visited, at which point it loops back to the beginning. The greedy heuristic is not very useful in practice, but it is linear in `n` and so can be used to speed up other heuristics by initializing them with a not-too-bad starting tour.

#### 2-opt
This is an implementation of the widely-used 2-opt algorithm. A 2-change is found by simply trying all pairs of edges; time complexity per iteration `O(n^2)` for a graph with `n` vertices.

#### 3-opt
This heuristic is analogous to 2-opt, but swaps out 3 edges for 3 edges. An edge that is removed may be added back again, so the possible 3-opt moves encompass all 2-opt moves as well. In this implementation, the 2-opt neighborhood is explored first, and if it is empty it moves into the 3-opt neighborhood by executing `pure 3-opt`.

#### Pure 3-opt
This is a variant of 3-opt wherein the sets of edges added and removed are disjoint, i.e. moves that are not equivalent to 2-opt. Note that there are 4 possible pure 3-opt moves once the edges to be removed are specified. Here, the move with the largest improvement is chosen.

#### Lin-Kernighan
This implements the variant of Lin-Kernighan specified [here](https://arxiv.org/abs/1003.5330). It is almost equivalent to the variant shown to be PLS complete by Papadimitriou, but written in a form easier to implement. Additionally, it is possible to specify the `greedy` option to `True`, which has the algorithm choose the first improving edge in the path optimization routine rather than the best improving edge.
