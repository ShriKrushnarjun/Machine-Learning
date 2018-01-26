import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import random

style.use("ggplot")


class KMeans:
    def __init__(self, k=8, tol=0.001, max_iterations=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iterations

    def fit(self, data):
        numbers = []
        self.centroids = {}
        # Selected centroids
        while len(numbers) < self.k:
            num = random.randrange(len(data))
            if num not in numbers:
                numbers.append(num)
        o = 0
        for n in numbers:
            self.centroids[o] = data[n].tolist()
            o += 1

        for _ in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for feature in data:
                distances = [np.linalg.norm(feature - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature.tolist())

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            for c in self.centroids:
                prev_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.abs(np.sum((current_centroid[0] - prev_centroid[0]) / current_centroid[0] * 100.0)) > self.tol:
                    if np.abs(
                            np.sum((current_centroid[1] - prev_centroid[1]) / current_centroid[1] * 100.0)) > self.tol:
                        optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


data = np.array([[1, 3],
                 [2, 4],
                 [3, 1],
                 [6, 7],
                 [5.5, 8],
                 [8, 9],
                 [1, 4],
                 [8, 9],
                 [6, 5]
                 ])

colors = ["r", "g", "b", "c", "k"]

clf = KMeans(k=2)
clf.fit(data)

classification = clf.classifications
centroids = clf.centroids
print(classification)
print(centroids)

for c in classification:
    color = colors[c]
    for j in classification[c]:
        plt.scatter(j[0], j[1], s=110, color=color, marker="*")

for ce in centroids:
    plt.scatter(centroids[ce][0], centroids[ce][1], s=110, color="b", marker="X")

# for i in range(len(data)):
#     plt.scatter(data[i][0], data[i][1], color=colors[0], s=150)

predict_this = []

for p in predict_this:
    classification = clf.predict(p)
    plt.scatter(p[0], p[1], s=110, color=colors[classification], marker="*")

plt.show()
