import pandas as pd
import numpy as np
from collections import Counter
import random

# K Should be greater than number of clusters and mostly be an odd number
# K means could fail on terabytes of datasets
# but it can improved with threded and radius attributes

accuracies = []

for i in range(25):
    def k_nearest_neighbour(data, predict, k):
        eucledian_distances = []
        for group in data:
            for features in data[group]:
                votes = np.linalg.norm(np.array(features) - np.array(predict))
                eucledian_distances.append([votes, group])
        votes = [i[1] for i in sorted(eucledian_distances)[:k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1] / k
        return vote_result, confidence


    df = pd.read_csv("../breast-cancer-wisconsin.data.txt")
    df.drop(["id"], 1, inplace=True)
    df.replace("?", -99999, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_percentage = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_percentage * len(full_data))]
    test_data = full_data[-int(test_percentage * len(full_data)):]

    for t in train_data:
        train_set[t[-1]].append(t[:-1])

    for t in test_data:
        test_set[t[-1]].append(t[:-1])

    # result, confidence = k_nearest_neighbour(train_set, test_set, k=4)
    # print(result, confidence)

    correct = 0
    total = 0

    for group in test_set:
        for features in test_set[group]:
            result, confidence = k_nearest_neighbour(train_set, features, k=5)
            if result == group:
                correct += 1
            total += 1
    # print("Accuracy:", (correct / total))
    accuracies.append(correct / total)

print("Accuracy:", sum(accuracies) / len(accuracies))
