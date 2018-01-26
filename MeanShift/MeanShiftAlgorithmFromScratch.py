import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use("ggplot")


class MeanShift:
    def __init__(self, radius=None, radius_norm_steps=100):
        self.radius = radius
        self.radius_norm_steps = radius_norm_steps

    def fit(self, data):
        centroids = {}

        weights = [i for i in range(self.radius_norm_steps)][::-1]

        if self.radius is None:
            all_points_avg = np.average(data, axis=0)
            all_points_norm = np.linalg.norm(all_points_avg)
            self.radius = all_points_norm / self.radius_norm_steps

        print(self.radius)

        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []

            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for d in data:
                    # if np.linalg.norm(centroid - d) < self.radius:
                    #     in_bandwidth.append(d)
                    distance = np.linalg.norm(centroid - d)
                    if distance == 0:
                        distance = 0.0000001
                    weight_index = int(distance / self.radius)
                    if weight_index > self.radius_norm_steps - 1:
                        weight_index = self.radius_norm_steps - 1
                    to_add = (weights[weight_index] ** 4) * [d]
                    in_bandwidth += to_add

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            to_pop = []

            for i in uniques:
                for ii in [i for i in uniques]:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                    break

            self.classification = {}
            for i in range(len(centroids)):
                self.classification[i] = []

            for featureset in data:
                distance = [np.linalg.norm(featureset - centroids[centroid]) for centroid in centroids]
                classification = distance.index(min(distance))
                self.classification[classification].append(featureset)

            if optimized:
                break

        self.centroids = centroids

    def predict(self, data):
        distance = [np.linalg.norm(data - centroid) for centroid in self.centroids]
        classification = distance.index(min(distance))
        return classification


data = np.array([[1, 3],
                 [2, 4],
                 [3, 1],
                 [6, 7],
                 [5.5, 8],
                 [8, 9],
                 [7, 2],
                 [8, 3],
                 [9, 1]])

clf = MeanShift()
clf.fit(data)
centroids = clf.centroids

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(data[:, 0], data[:, 1])

for c in centroids:
    ax.scatter(centroids[c][0], centroids[c][1], marker="*", color="k", s=150)

plt.show()
