import numpy as np
import scipy.linalg as linalg
from util.sinkhorn_knopp import SinkhornKnopp
np.random.seed(4554365)


def generate_stochastic_matrix(n, iter_max=10):
    mat = np.random.rand(n, n)
    return mat/np.sum(mat, axis=0)[:, None]

    

def sort_by_statistic_average(matrices, distributions, statistic):
    averages = [statistic @ d for d in distributions]
    #print(averages)
    triples = zip(averages, matrices, distributions)
    return sorted(triples)[::-1]


n = 10
num = 10
t = 10

statistic = 10000*np.random.random(n)
w = np.diag(statistic)


matrices = [generate_stochastic_matrix(n) for _ in range(num)]
print([np.linalg.norm(w @ m @ np.linalg.inv(w), ord=2) for m in matrices])
print([np.linalg.norm(m, ord=2) for m in matrices])