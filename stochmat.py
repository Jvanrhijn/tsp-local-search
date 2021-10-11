import numpy as np
import scipy.linalg as linalg
from util.sinkhorn_knopp import SinkhornKnopp
np.random.seed(0)


def generate_stochastic_matrix(n, iter_max=10):
    mat = np.random.rand(n, n)
    sk = SinkhornKnopp()
    #mat = mat/np.sum(mat, axis=1)[:, None]
    return sk.fit(mat)

    

def sort_by_statistic_average(matrices, distributions, statistic):
    averages = [statistic @ d for d in distributions]
    #print(averages)
    triples = zip(averages, matrices, distributions)
    return sorted(triples)[::-1]


n = 10
num = 10
t = 10


matrices = [generate_stochastic_matrix(n) for _ in range(num)]
distributions = [np.linalg.eig(m)[1][0] for m in matrices]

#print(sum(matrices[0][0, :]))

# the random statistic to average
#statistic = np.random.rand(n)
#
#averages, matrices, distributions = zip(*sort_by_statistic_average(matrices, distributions, statistic))
#
#mu = distributions[0]
#for time in range(t):
#    mu = mu @ matrices[time]
#    #print(mu @ statistic, distributions[time] @ statistic)
#


v = linalg.eig(matrices[0], left=True)[1][1]
#v /= sum(v)
print(v)