#!/usr/bin/python3

import abc
import math
import random

import numpy as np
from matplotlib import pyplot as plt


class Distribution(abc.ABC):
    @abc.abstractmethod
    def cdf(self, x):
        pass


class TaskDistribution(Distribution):
    def __init__(self, a):
        self.a = a

    def cdf(self, x):
        a = self.a
        c = 1 / (2 * math.exp(-a) + 2 * a)
        if x < -a:
            return c * math.exp(x)
        elif x < a:
            return c * (math.exp(-a) + x + a)
        else:
            return c * (2 * math.exp(-a) + 2 * a - math.exp(-x))


class InverseFunctionMethod(abc.ABC):
    @abc.abstractmethod
    def inverse_cdf(self, x):
        pass

    def __call__(self):
        return self.inverse_cdf(random.random())



class TaskInverseMethod(InverseFunctionMethod):
    def __init__(self, a):
        self.a = a

    def inverse_cdf(self, y):
        a = self.a
        c = 1 / (2 * math.exp(-a) + 2 * a)
        if y < c * math.exp(-a):
            return math.log(y / c)
        elif 1 - y < c * math.exp(-a):
            return -math.log((1 - y) / c)
        else:
            return (y / c - math.exp(-a) - a)


class QDistribution(Distribution):
    def cdf(self, x):
        if x < 0:
            return 0.5 * math.exp(x)
        else:
            return 0.5 * (2 - math.exp(-x))

    def f(self, x):
        return 0.5 * math.exp(-abs(x))


class QInverseMethod(InverseFunctionMethod):
    def inverse_cdf(self, y):
        if y < 0.5:
            return math.log(2 * y)
        else:
            return -math.log(2 - 2 * y)


class FiltrationMethod(abc.ABC):
    @abc.abstractmethod
    def Q(self):
        pass

    @abc.abstractmethod
    def QDist(self):
        pass

    @abc.abstractmethod
    def f(self, x):
        pass

    def r(self, x):
        return self.f(x) / self.QDist().f(x)

    @abc.abstractmethod
    def M(self):
        pass

    def __call__(self):
        M = self.M()
        Q = self.Q()
        while True:
            eta = Q()
            a = random.random()
            if self.r(eta) > M * a:
                return eta


class TaskFiltrationMethod(FiltrationMethod):
    def __init__(self, a):
        self.a = a

    def Q(self):
        return QInverseMethod()

    def QDist(self):
        return QDistribution()

    def f(self, x):
        a = self.a
        c = 1 / (2 * math.exp(-a) + 2 * a)
        if abs(x) > a:
            return c * math.exp(-abs(x))
        else:
            return c

    def M(self):
        a = self.a
        c = 1 / (2 * math.exp(-a) + 2 * a)
        return max(c / 0.5, 2 / c)



def simulate(dist, method, steps, to_save):
    x = np.linspace(-5, 5, 100)
    fig, ax = plt.subplots()
    ax.plot(x, list(map(dist.cdf, x)), label="Real CDF")

    data = [method() for i in range(steps)]
    data.sort()
    ax.plot(data, list(map(lambda i : i / steps, range(steps))), label="Heuristic CDF")

    ax.legend()
    fig.savefig(to_save)


def main():
    dist = TaskDistribution(0.5)
    simulate(dist, TaskInverseMethod(0.5), 100, "inverse_method.png")
    simulate(dist, TaskFiltrationMethod(0.5), 100, "filtration_method.png")


if __name__ == "__main__":
    main()
