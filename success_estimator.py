from math import asin, sqrt, sin


def u_i(a, n, i):
    if i == 1:
        return 1 - 2*a/n

    v = u_i(a, n, i-1)
    return 2*v**2 - 1


# u_i(5, 8, 10)


def p_failure_conditionned(a, n):
    total = 2**n
    theta_a = asin(sqrt(a/total))

    v = 1
    for i in range(1, n + 1):
        print(v)
        v *= sin((2**(n-i+1) + 1)*theta_a)

    return 1 - v


# print(p_failure_conditionned(8, 5))


