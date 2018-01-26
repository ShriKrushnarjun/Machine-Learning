import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style

style.use("ggplot")

centers = [[1, 1, 1], [3, 10, 10], [5, 5, 5]]
X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1.5)

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
centers = ms.cluster_centers_

print(centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters : ", n_clusters_)

colors = 10 * ["r", "g", "b", "c", "k"]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], color=colors[labels[i]], marker="o")

for j in range(len(centers)):
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color="k", marker="X", s=150)

print(centers)

plt.show()
