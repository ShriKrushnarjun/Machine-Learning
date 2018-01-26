import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: "r", -1: "b"}
        if visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data

        # w,b dict
        opt_dict = {}

        all_data = []
        for classes in self.data:
            for feature_set in self.data[classes]:
                for features in feature_set:
                    all_data.append(features)

        self.max_feature_in_data = max(all_data)
        self.min_feature_in_data = min(all_data)
        all_data = None

        # Step Sizes
        step_size = [self.max_feature_in_data * 0.1,
                     self.max_feature_in_data * 0.01,
                     # Expensive Step
                     self.max_feature_in_data * 0.001,
                     # Very Expensive Step
                     self.max_feature_in_data * 0.0001]

        # Transformation of w
        trasformation = [[1, 1],
                         [-1, 1],
                         [1, -1],
                         [-1, -1]]

        # Step Size for b
        b_range_multiple = 1

        # b step multiple
        b_multiple = 5

        latest_optimum = self.max_feature_in_data * 10

        for step in step_size:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_in_data * b_range_multiple),
                                   self.max_feature_in_data * b_range_multiple,
                                   step * b_multiple):
                    for t in trasformation:
                        w_t = w * t
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                            if not found_option:
                                break
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print("Optimized a step.")
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])

            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

    def predict(self, features):
        # w.x+b
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification and self.visualization:
            self.ax.scatter(features[0], features[1], s=100, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(p[0], p[1], s=100, color=self.colors[i]) for p in data_dict[i]] for i in data_dict]

        # plotting planes
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_in_data * 0.9, self.max_feature_in_data * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # x.w + b= 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], "k")

        # x.w + b= -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], "k")

        # x.w + b= 1
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], "y--")

        plt.show()


data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8]]),

             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3]])}

svm = SupportVectorMachine()
svm.fit(data_dict)

predictions = [[2, 4],
               [3, 5],
               [8, 9],
               #               [-5, -1],
               [5, 1],
               [-1, 5],
               [5, -1],
               [6, 0],
               [8, 0],
               [2, 2.518]]

print([[i, " : ", svm.predict(i)] for i in predictions])
svm.visualize()
