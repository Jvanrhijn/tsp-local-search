import sys
import numpy as np
import math


# absolutely horrible
def string_starts_with_number(string):
    try:
        n = float(string.lstrip()[0])
        return True
    except:
        return False
    

def read_coordinate_file(path):
    xs = []
    ys = []
   
    
    with open(path, "r") as file:
        lines = file.readlines()
        coordinates = list(filter(string_starts_with_number, lines))

        for line in coordinates:
            l = line.split()
            xs.append(float(l[1]))
            ys.append(float(l[2]))

    points = np.zeros((len(xs), 2))
    points[:, 0] = xs
    points[:, 1] = ys

    return points


def write_distance_matrix(d, outpath):
    dt = d.T
    with open(outpath, "w") as file:
        for i, line in enumerate(dt):
            for num in line:
                file.write(str(num) + " ")
            if i != len(dt)-1:
                file.write("\n")


def get_distance_matrix(points):
    d = np.zeros((len(points), len(points)))

    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            d[i, j] = np.linalg.norm(points[i] - points[j])

    return d


def read_distance_matrix(path):
    points = read_coordinate_file(path)
    return get_distance_matrix(points)
