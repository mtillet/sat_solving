from random import randint, sample
from itertools import product
import z3


class ThreeSatInstance:
    """
    Instance of 3SAT problem, provided with solution-counting methods
    """

    def __init__(self):

        self.clauses = []
        self.nb_variables = 0
        self.coefficient = 1

    def set_random(self, nb_clauses, nb_variables):
        """
        Create a random instance of 3SAT
        :param nb_clauses: number of clauses the instance shall have
        :param nb_variables: number of variables the instance shall have
        """

        self.clauses = []
        self.nb_variables = nb_variables

        for _ in range(nb_clauses):
            # a clause is composed of non-zero integers, each one's absolute value being the variable symbolized,
            # and a negative sign meaning the presence of a NOT
            self.clauses.append(sample(range(1, nb_variables + 1), 3))
            self.clauses[-1].sort()

            for i in range(3):
                # randomly assign signs
                self.clauses[-1][i] *= 2 * randint(0, 1) - 1

    def remove_unused_var(self):
        """
        For low ratios, it is possible some variables are not used within the clause. Since it simplifies the space of
        solutions to work without these, we remove them (and take into account their existence with the coefficient)
        """

        # print("Ratio : ", len(instance)/nb_variables)

        unused_variables = [i + 1 for i in range(self.nb_variables)]
        # we first find which variables are not used
        for clause in self.clauses:
            for var in clause:
                if abs(var) in unused_variables:
                    unused_variables.remove(abs(var))

        # then we update the instance so that the only unused variables are those with the greatest absolute value
        # (without changing the problem itself, it is only a reindexation)
        for index, i in enumerate(unused_variables):
            for clause in self.clauses:
                for ind, var in enumerate(clause):
                    if abs(var) > i - index:
                        clause[ind] -= int(var / abs(var))

        nb_useful_variables = self.nb_variables - len(unused_variables)

        # the number of solutions found for the modified problem shall be multiplied by this coefficient
        # to have the solution for the original problem
        self.coefficient = 2 ** (self.nb_variables - nb_useful_variables)
        self.nb_variables = nb_useful_variables

    def solve_sharp_sat(self, use_cautious_leap_of_faith=False):
        """
        Most general function which encapsulate all solvers for special cases
        :param use_cautious_leap_of_faith:
        :return: the number of solutions, ie the solution to the #SAT problem
        """

        ratio = len(self.clauses) / self.nb_variables

        # threshold totally arbitrary to switch to z3 rather than brute forcing
        if ratio > 7 and self.nb_variables > 12:
            leap_of_faith = use_cautious_leap_of_faith and ratio > 10
            return self.use_z3(leap_of_faith=leap_of_faith)

        else:
            return self.find_number_of_sols()

    def find_number_of_sols(self):
        """
        Try every possibilities to find the number of solutions, after simplification of the problem
        :return: the number of solutions
        """
        self.remove_unused_var()
        t = ["".join(seq)[::-1] for seq in product("01", repeat=self.nb_variables)]

        nb_sols = 0
        for val in t:
            nb_sols += self.evaluate(val)

        return nb_sols * self.coefficient

    def use_z3(self, leap_of_faith=False):
        """
        Use the z3 library to check satisfiability.

        :param leap_of_faith: USE AT YOUR OWN RISK
        For high ratios, one can assume that if an instance is satisfiable, it is likely to only have one solution
        It's absolutely false for medium and small ratios, but can provide a great acceleration, especially for instance
        with numerous variable, if used in fitting circumstances
        :return: the number of solutions (or what it is supposed to be, if leap_of_faith is set to True)
        """
        v = self.solve_with_z3()

        if v:
            # the problem is not satisfiable
            return 0
        else:
            # the problem have an unknown number of solutions
            if leap_of_faith:
                return 1
            else:
                return self.find_number_of_sols()

    def evaluate(self, val):
        """
        Check if val is a solution to instance
        :param val: a string of 0 and 1, symbolizing the values of the variables
        :return: True if it does validate, else False
        """

        for clause in self.clauses:
            total = 0

            for var in clause:
                if (var < 0 and val[-var - 1] == '0') or (var > 0 and val[var - 1] == '1'):
                    total = 1
                    continue

            if total == 0:
                return False
        return True

    def solve_with_z3(self):
        """
        Use the z3 librairy to find if the problem is satisfiable
        (but cannot tell its number of solutions if there are some)
        :return: True if the problem is NOT satisfiable, else False (if satisfiable or unknown)
        """
        z3_instance = convert_instance(self.clauses, self.nb_variables)
        return z3_instance.check() == z3.unsat

    def get_ratio(self):
        """
        Compute the ratio nb of clauses / nb of variables
        :return:
        """
        return len(self.clauses)/self.nb_variables


def convert_instance(instance, nb_variables):
    """
    Convert a ThreeSatInstance object to a z3 problem
    :param instance: the clauses
    :param nb_variables: the number of variables
    :return: a z3 problem tantamount to the 3sat problem
    """

    variables = [z3.Bool("x_%s" % i) for i in range(nb_variables)]

    s = z3.Solver()

    for clause in instance:

        s.add(z3.Or([variables[i - 1] for i in clause if i > 0] +
                    [z3.Not(variables[abs(i) - 1]) for i in clause if i < 0]))

    return s
