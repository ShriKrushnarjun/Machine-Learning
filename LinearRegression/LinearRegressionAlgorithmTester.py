import LinearRegression.LinearRegressionAlgorithm as lr
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import random


def create_dataset(hm, variance, step=2, corelation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if corelation and corelation == "pos":
            val += step
        elif corelation and corelation == 'neg':
            val -= step
    xs = [i for i in range(hm)]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


for i in range(10, 100, 5):
    xs, ys = create_dataset(40, i, 2, "pos")
    predict_x = 45

    linerar_regression_classifier = lr.LinearRegression()
    linerar_regression_classifier.train(xs, ys)
    print(linerar_regression_classifier.accuracy())
    print(linerar_regression_classifier.classify(predict_x))

    style.use("fivethirtyeight")
    plt.scatter(xs, ys)
    plt.scatter(predict_x, linerar_regression_classifier.classify(predict_x))
    plt.plot(xs, linerar_regression_classifier.regression_line())
    plt.show()


# For Linear regression the variance should be less and points should be more
# For defining a relationship Number_of_points < variance is not good
#                             Number_of_points > variance is good
#                             lesser the variance more nice the output is
#                             morer the varince algorithem wil fail


