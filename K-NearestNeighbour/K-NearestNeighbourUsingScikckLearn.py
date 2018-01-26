import numpy as np
from sklearn import cross_validation, neighbors
import pandas as pd

accuracies = []

for i in range(25):
    df = pd.read_csv("../breast-cancer-wisconsin.data.txt")
    df.drop(["id"], 1, inplace=True)
    df.replace("?", -99999, inplace=True)

    X = df.drop(["class"], 1)
    y = df["class"]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    # print(accuracy)

    # queries_array = np.array([[4, 2, 1, 3, 5, 6, 4, 7, 8],
    #                           [7, 8, 5, 4, 1, 2, 3, 6, 5],
    #                           [4, 5, 6, 2, 3, 6, 1, 2, 1],
    #                           [1, 2, 1, 2, 1, 2, 1, 2, 1],
    #                           [1, 2, 1, 2, 1, 2, 1, 2, 1]])
    # predictions = clf.predict(queries_array)
    # [print(queries_array[i], [predictions[i]]) for i in range(len(queries_array) - 1)]
    accuracies.append(accuracy)

print("Accuracy:", sum(accuracies) / len(accuracies))
