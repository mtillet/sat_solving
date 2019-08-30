"""
Some functions which should eventually calculate the probability that a Grover algorithm run wrongly classify a 3SAT
instance as unstatisfiable. WIP, as it requires a more in depth study of the distribution of #3SAT problem to solve
completely.
"""

from math import asin, sqrt, sin
from scipy.stats import nbinom
import numpy as np

from sat_study import conc


def u_i(a, n, i):
    """
    Recursively compute the i-th iteration of the u serie, defined by
    u_1 = 1 - 2a/n
    u_i = 2u_(i-1)**2 - 1

    :return: u_i
    """
    if i == 1:
        return 1 - 2*a/n

    v = u_i(a, n, i-1)
    return 2*v**2 - 1


def p_failure_conditioned(a, n):
    """
    Calculates the conditioned probability of failure of Grover 3SAt solver with n variables and a solutions
    :param a: number of solutions to the problem considered
    :param n: number of variables
    :return: the probability of failure
    """
    total = 2**n
    theta_a = asin(sqrt(a/total))

    v = 1
    for i in range(1, n + 1):
        print(v)
        v *= sin((2**(n-i+1) + 1)*theta_a)

    return 1 - v


def compute_failure_probability(instance, m):

    ratio = instance.get_ratio()
    nb_variables = instance.nb_variables
    x = np.arange(0, 2**nb_variables + 1)

    p, r = load_nbinom_params(ratio, nb_variables)

    probabilities = nbinom.pmf(x, r, p)

    partial_sum = 1 - probabilities[0]

    for k in range(0, 2**nb_variables + 1):
        theta_k = np.arcsin(sqrt(k/2**nb_variables))
        probabilities[k] *= (np.cos(2*(m+1)*theta_k))**2

    final_probability = sum(probabilities[1:])/partial_sum

    return final_probability


def load_nbinom_params(ratio, nb_variables):

    if nb_variables == 16:
        x, prob = conc(["p_16_0.1_1.01.npy", "p_16_2_6.1.npy", "p_16_6_9.1.npy"])
        x = x[:-1]
        prob = prob[:-1]

        p = interpolate(ratio, x, prob)

        x, r = conc(["r_16_1_7.1.npy"])
        r = interpolate(ratio, x, r)

        return p, r

    pass


def interpolate(ratio, x, prob):
    i = 0
    while i < len(x) and x[i] < ratio:
        i += 1

    if i == len(x) or i == 0:
        return 1

    w = (ratio - x[i-1]) / (x[i] - x[i-1])

    p = prob[i-1] + (prob[i] - prob[i-1])*w

    return p




