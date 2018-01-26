import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

style.use("ggplot")

data = np.array([[1, 3],
                 [2, 4],
                 [3, 1],
                 [6, 7],
                 [5.5, 8],
                 [8, 9]])

clf = KMeans(n_clusters=4)
clf.fit(data)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.", "r.", "b.", "c.", "k."]

for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize=15)
print(centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, color="b", marker="X")
plt.show()
