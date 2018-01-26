from statistics import mean


class LinearRegression():
    def __init__(self):
        pass

    def best_fit_slope_and_intercept(self, xs, ys):
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / (mean(xs) ** 2 - mean(xs ** 2)))
        c = mean(ys) - m * mean(xs)
        return m, c

    def squared_error(self, ys_orig, ys_regression):
        return sum((ys_orig - ys_regression) ** 2)

    def coeficient_of_determindation(self, ys_orig, ys_regression):
        ys_mean_line = [mean(ys_orig) for y in ys_orig]
        ys_regression_se = self.squared_error(ys_orig, ys_regression)
        ys_mean_se = self.squared_error(ys_orig, ys_mean_line)
        return 1 - (ys_regression_se / ys_mean_se)

    def classify(self, given_x):
        return (self.m * given_x) + self.c

    def train(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.m, self.c = self.best_fit_slope_and_intercept(xs, ys)
        self.regression_line_ys = [(self.m * x) + self.c for x in self.xs]

    def accuracy(self):
        return self.coeficient_of_determindation(self.ys, self.regression_line_ys)

    def regression_line(self):
        return self.regression_line_ys
