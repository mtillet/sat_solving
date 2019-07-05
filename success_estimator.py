"""
Some functions which should eventually calculate the probability that a Grover algorithm run wrongly classify a 3SAT
instance as unstatisfiable. WIP, as it requires a more in depth study of the distribution of #3SAT problem to solve
completely.
"""

from math import asin, sqrt, sin


def u_i(a, n, i):
    """
    Recursively compute the i-th iteration of the u serie.
    :param a:
    :param n:
    :param i:
    :return:
    """
    if i == 1:
        return 1 - 2*a/n

    v = u_i(a, n, i-1)
    return 2*v**2 - 1


# u_i(5, 8, 10)


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


# print(p_failure_conditionned(8, 5))


