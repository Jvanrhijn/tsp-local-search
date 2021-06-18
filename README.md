# TSP Local Search
This repository contains simple implementations of some local search algorithms for TSP. It is solely meant for research purposes, and *not* meant to be used for any practical purposes.

### Heuristics

* 2-opt
* pure 3-opt
* Lin-Kernighan

#### 2-opt
This is an implementation of the widely-used 2-opt algorithm. A 2-change is found by simply trying all pairs of edges; time complexity per iteration `O(n^2)` for a graph with `n` vertices.

#### Pure 3-opt
This is a variant of 3-opt wherein the sets of edges added and removed are disjoint, i.e. moves that are not equivalent to 2-opt.

#### Lin-Kernighan
This implements the variant of Lin-Kernighan specified [here](https://arxiv.org/abs/1003.5330). It is almost equivalent to the variant shown to be PLS complete by Papadimitriou, but written in a form easier to implement.
