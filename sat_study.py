"""
Functions designed to study the distribution of the number of solutions of random 3SAT problems
"""

from matplotlib import pyplot as plt
from scipy.stats import nbinom
import numpy as np

from three_sat_instance import ThreeSatInstance


def compute_satisfiability(nb_variables=15, ratio=2.5, iterations=15, nb_instances=100, step=0.25):
    """
    Calculate the proportion of satisfiable 3SAT instance for different ratios clauses/variables
    :param nb_variables: Number of variables to use for the problem
    :param ratio: Ratio clause/variable with which to begin
    :param iterations: Number of different ratios to use
    :param nb_instances: Number of 3SAT instance to generate to compute the proportion
    :param step: step between two successive ratios
    :return: ratios : the lists of ratios used; means: the list of the corresponding means of satisfiable instances
    """

    ratios = []
    means = []

    for i in range(iterations):
        print(i)
        ratio += step
        nb_clauses = int(ratio * nb_variables)
        satisfiable = 0
        for _ in range(nb_instances):
            instance = ThreeSatInstance()
            instance.set_random(nb_clauses, nb_variables)
            if not instance.solve_with_z3():
                satisfiable += 1
        mean = satisfiable/nb_instances
        ratios.append(ratio)
        means.append(mean)

    return ratios, means


def curve(mini, maxi):
    for i in range(mini, maxi):
        ratios, means = compute_satisfiability(i)
        plt.plot(ratios, means, label=i)
    plt.legend()
    plt.show()


def distribution(nb_variables, nb_iterations=100, ratio=5):

    nb_clauses = int(nb_variables * ratio)
    nb_sols = [0]*2**nb_variables

    for i in range(nb_iterations):
        instance = ThreeSatInstance()
        instance.set_random(nb_clauses, nb_variables)

        nb_sol = instance.solve_sharp_sat()

        nb_sols[nb_sol] += 1

    mini = 0
    maxi = 2**nb_variables - 1

    results = [nb_sols[v]/nb_iterations for v in range(mini, maxi + 1)]
    index = [i for i in range(mini, maxi + 1)]

    return index, results


def distrib_plot(nb_variables_min, nb_variables_max, ratio_min, ratio_max, step, nb_iterations=1500, show=False, mode=None):
    """
    Compute the distribution of the number of solutions to 3 SAT instances for different parameters,
    supposing it follows a negative binomial law

    :param nb_variables_min: minimum number of variables
    :param nb_variables_max: maximum number of variables
    :param ratio_min: minimum ratio
    :param ratio_max: maximum ratio value
    :param step: ratio step
    :param nb_iterations: Number of
    :param show: if True, shows the graph of the proportion of randomly picked 3SAT by number of solutions,
    and the calculated negative binomial associated
    :return:
    """

    probas = []
    r = []

    for nb_variables in range(nb_variables_min, nb_variables_max + 1):
        print("Number of variables: ", nb_variables)
        ratio = ratio_min
        while ratio < ratio_max:
            print("Current ratio:", ratio)

            index, results = distribution(nb_variables=nb_variables, nb_iterations=nb_iterations, ratio=ratio)
            if show:
                lab = "Number of variables: " + str(nb_variables) + ", Ratio: " + str(ratio)
                index = [i/2**nb_variables for i in index]
                res = [i*2**nb_variables for i in results]
                print(res)
                print(results)
                plt.plot(index, res, label=lab)
            ratio += step

            mean = sum([results[i]*i for i in range(len(results))])
            variance = sum([results[i]*nb_iterations*(i - mean)**2 for i in range(len(results))])/nb_iterations

            if variance == 0:
                # if there is no variance, the model fails
                p = 0
                n = 0
            else:
                # we assume the distribution generally follows a negative binomial law, and find its parameters
                if mean/variance < 1:
                    p = mean/variance
                    n = variance * p**2 / (1 - p)

                else:
                    p = variance/mean
                    n = mean**2 / (variance - mean)

            probas.append(p)
            r.append(n)

            if show:
                x = np.arange(0, 2**nb_variables)
                plt.plot(x/2**nb_variables, nbinom.pmf(x, n, p)*2**nb_variables, ms=8,
                         label="Calculated n: " + str(n) + " p: " + str(p))

    if show:
        plt.legend()
        plt.show()

    if mode == 'r':
        return r
    if mode == 'p' or mode is None:
        return probas
    if mode == 'rp':
        return r, probas
    if mode == "pr":
        return probas, r


def negative_binom():
    # n, p = 16, 0.95
    n = 10
    p = 0.999
    print(p)
    print(n*(1-p)/(p**2))
    fig, ax = plt.subplots(1, 1)
    x = np.arange(0, 16)
    ax.plot(x, nbinom.pmf(x - 12, n, p), ms=8, label='nbinom pmf')
    ax.plot(x, nbinom.pmf(x, n, p), ms=8)
    plt.show()


def compute_graph(nb_variables_min, nb_variables_max, ratio_min, ratio_max, step, nb_iterations=1500, mode=None):

    if mode is None:
        mode = 'p'

    for i in range(nb_variables_min, nb_variables_max + 1):
        res = distrib_plot(i, i, ratio_min, ratio_max, step, nb_iterations, show=True, mode=mode)
        if int((ratio_max - ratio_min) / step) == (ratio_max - ratio_min)/step:
            x = np.arange(ratio_min, ratio_max + step/2, step)
        else:
            x = np.arange(ratio_min, ratio_max, step)

        np.save(mode + "_" + str(i) + "_" + str(ratio_min) + "_" + str(ratio_max), np.array([x, res]))
        plt.plot(x, res, label=str(i))
    plt.legend()
    plt.show()


def hypothesis(mini, maxi, show=False):

    x = np.arange(mini, maxi, 0.001)
    val = [0]*int((maxi - mini)/0.001)
    for ind, v in enumerate(x):
        # val[ind] = np.exp(-1.7*v) + np.arctan((v-5)/6)*2/np.pi + 0.33
        # val[ind] = np.exp(-2*v) + np.arctan(v/6.5 - 0.9)*2/np.pi + 0.45

        val[ind] = min((v/2)**3.3/120 + np.exp(-8*v), 1)
        # val[ind] = min(1, np.exp(v/4)/12)
    if show:
        plt.plot(x, val, "--g", label="Hypothetical p for n=12")

    p_10 = ["p_10.npy", "p_10_7_12.1.npy"]

    x, probas = conc(p_10)
    plt.plot(x, probas, label="Calculated p for n=10")

    p_12 = ["p_12_0.01_0.5.npy", "p_12_0.5_7.1.npy", "p_12_7_8.1.npy"]

    x, probas = conc(p_12)
    x = x[5:]
    probas = probas[5:]
    x = np.append([0], x)
    probas = np.append([1], probas)

    plt.plot(x, probas, label="Calculated p for n=12", color='r')

    p_14 = ["p_14_0.07_1.npy", "p_14_1_4.1.npy", "p_14_4_10.1.npy"]
    x, prob = conc(p_14)

    x = x[3:]
    prob = prob[3:]
    x = np.append([0], x)
    x = x[:-1]
    prob = np.append([1], prob[:-1])

    plt.plot(x, prob, label="Calculated p for n=14")

    x, prob = conc(["p_16_0.1_1.01.npy", "p_16_2_6.1.npy", "p_16_6_9.1.npy"])
    x = x[:-1]
    prob = prob[:-1]
    plt.plot(x, prob, label="Calculated p for n=16", color="black")

    plt.ylabel("Associated p value")
    plt.xlabel("Ratio")
    plt.legend()
    plt.show()

    return x, val


def conc(files):

    x = [0]
    y = [1]
    for file in files:
        [a, b] = load(file)
        x = np.append(x, a)
        y = np.append(y, b)
    return x, y


def load(file):
    """
    Load a numpy file
    :param file: the name of the file, as a string
    :return: the content of the file
    """
    return np.load(file)


def show_file(file):
    [x, probas] = load(file)
    plt.plot(x, probas, label=file)
    plt.show()


# inst = create_3sat(12, 4)
# print(inst)
# print(general_solver(instance=inst, nb_variables=4))
# proba(10)
# curve(5, 17)
# poiss()
# distribution(6, nb_iterations=5000)
# distrib_plot(10, 10, 3.5, 4.5, 0.4, nb_iterations=1000, show=True)
# negat_binom()


compute_graph(16, 16, 1, 7.1, 1, nb_iterations=500, mode='r')
# show("p_4.npy")

# hypothesis(0, 10, True)
# distrib_plot(10, 13, 2, 3, 1, nb_iterations=1000, show=True)
# curve(10, 16)