import numpy as np
import matplotlib.pyplot as plt


def assign_to_clusters(points, centers):
    num_clusters = len(centers)
    clusters = [[] for _ in range(num_clusters)]
    
    for x in points:

        best_cluster = np.array([np.linalg.norm(x - c) for c in centers]).argmin()
        
        clusters[best_cluster].append(x)

    return clusters


def compute_centers(clusters):
    return [sum(c) / len(c) for c in clusters]


def objective(clusters, centers):
    return sum(
        sum(np.linalg.norm(x - center) for x in cluster) \
                for center, cluster in zip(centers, clusters)
        )


def kmeans(points, num_clusters):
    centers = np.random.uniform(size=(num_clusters, 2))

    obj_prev, obj = np.inf, np.inf

    time = 0
    
    while True:
        time += 1

        clusters = assign_to_clusters(points, centers)
        centers = compute_centers(clusters)
        obj = objective(clusters, centers)

        if obj_prev == obj:
            break

        obj_prev = obj

    return assign_to_clusters(points, centers), centers, time


if __name__ == "__main__":
    num_clusters = 9
    num_points = 4000

    np.random.seed(0)

    # generate "true" cluster centers
    #centers = np.random.uniform(size=(num_clusters, 2), low=-5, high=5)

    # generate guassian clusters, points evenly distributed among
    # them
    #clusters = tuple(np.random.normal(loc=c, scale=0.375, size=(num_points // num_clusters, 2)) for c in centers)
    #points = np.concatenate(clusters)
    points = np.random.uniform(low=-1, high=1, size=(num_points, 2))

    clusters, centers, time = kmeans(points, num_clusters)

    print(f"k-means converged in {time} iterations")

    fig, ax = plt.subplots(1)
    for i, c in enumerate(map(np.array, clusters)):
        ax.scatter(c[:, 0], c[:, 1])
        ax.plot([centers[i][0]], [centers[i][1]], color="black", marker="o", markersize=10)

    plt.show()