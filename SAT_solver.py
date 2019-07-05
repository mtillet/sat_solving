from itertools import combinations
from math import ceil, sqrt, pi
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer

# we use an implementation of the multi qubit controlled Toffoli gate more efficient than Aqua's mct, to gain some gates
from SAT_solver.utils import cnx_o


def add_3sat_clause(circ, qr, clause, index):
    cont = [qr[abs(i)] for i in clause]

    # negates the qbits without NOT on the clause
    for i in clause:
        if i > 0:
            circ.x(qr[i])

    cnx_o(circ, qr, cont, qr[index], qr[0])

    # brings the negated qubits to their original value
    for i in clause:
        if i > 0:
            circ.x(qr[i])


def add_clause_exclusive(circ, qr, clause, index):
    """
    Add a clause of the 1-exclusive SAT problem to the circuit
    :param circ: circuit
    :param qr: quantum registers
    :param clause: a clause of the SAT problem, using any number of variables
    :param index: the index of the quantum register which will carry the result of the evaluation of the clause
    :return:
    """
    # indexes of the qubits carrying the variables
    cont = [qr[abs(i)] for i in clause]

    # negates the qbits with NOT on the clause
    for i in clause:
        if i < 0:
            circ.x(qr[-i])

    # iterates on odd controlled not
    for control_len in range(1, len(clause) + 1, 2):
        conts = combinations(cont, control_len)

        for controls in conts:
            # cnx_o(circ, qr, list(controls), qr[index], qr[0])
            cnx_o(circ, qr, list(controls), qr[index], qr[0])

    # brings the negated qubits to their original value
    for i in clause:
        if i < 0:
            circ.x(qr[-i])


def sign_flip(circ, qr, controls):
    """
    Flips the sign of the states whose control qubits are all one
    :param circ: circuit
    :param qr: quantum registers
    :param controls: m qubits which ensure the correctness of the solution
    :return:
    """
    circ.u1(pi, qr[1])
    cnx_o(circ, qr, controls, qr[1], qr[0])
    circ.u1(pi, qr[1])
    cnx_o(circ, qr, controls, qr[1], qr[0])


def oracle(circ, qr, clauses, var_nb, is_exclusive):
    """
    Oracle for the problem
    :param circ: circuit
    :param qr: quantum registers
    :param clauses: lists of clauses (see ex_1_3SAT)
    :param var_nb: number of variables
    :param is_exclusive: should the problem be regular SAT or 1 exclusive ?
    :return:
    """
    nb_qbits = var_nb + len(clauses) + 1

    # adds every clause operator
    for i, clause in enumerate(clauses):
        if is_exclusive:
            add_clause_exclusive(circ, qr, clause, var_nb + i + 1)
        else:
            add_3sat_clause(circ, qr, clause, var_nb + i + 1)

    # list of the qubits carrying the result of the clauses
    clauses_checkers = [qr[i] for i in range(var_nb + 1, nb_qbits)]

    # flips the sign
    sign_flip(circ, qr, clauses_checkers)

    # removes the effect of every clause, by just applying them again
    for i, clause in enumerate(clauses):
        if is_exclusive:
            add_clause_exclusive(circ, qr, clause, var_nb + i + 1)
        else:
            add_3sat_clause(circ, qr, clause, var_nb + i + 1)


def grover_flip(circ, qr, to_flip):
    """
    Flip the amplitudes around the mean
    :param circ: circuit
    :param qr: quantum registers
    :param to_flip: list of the registers carrying the values of the variables
    :return:
    """
    for qr_i in to_flip:
        circ.h(qr_i)
        circ.x(qr_i)
    circ.h(to_flip[0])
    cnx_o(circ, qr, to_flip[1:], to_flip[0], qr[0])
    circ.h(to_flip[0])
    for qr_i in to_flip:
        circ.x(qr_i)
        circ.h(qr_i)


def ex_1_3sat_circuit(var_nb, clauses, nb_grover=-1, is_exclusive=True):
    """
    Builds the circuit associated with the SAT problem
    :param var_nb: number of variables
    :param clauses: list of the clauses
    :param nb_grover: number of applications of grover's operator
    :param is_exclusive: should the problem be regular SAT or 1-exclusive ?
    :return: the circuit object, and the quantum registers
    """
    nb_qbits = var_nb + len(clauses) + 1
    qr = QuantumRegister(nb_qbits)
    cr = ClassicalRegister(var_nb)
    # convention : 0: lost, 1-var_nb : variables; var_nb+1-nb_qubits : clause results;
    circ = QuantumCircuit(qr, cr)

    # superposition of the states of the variables
    for i in range(1, var_nb + 1):
        circ.h(qr[i])

    if nb_grover < 0:
        # this is the theoretical optimal number of iterations
        # although, in practical cases, it can be preferable to iterate on a lesser number so the program is shorter
        nb_grover = ceil(sqrt(2**var_nb) * pi / 4)

    for _ in range(nb_grover):

        oracle(circ, qr, clauses, var_nb, is_exclusive)
        grover_flip(circ, qr, [qr[i] for i in range(1, var_nb + 1)])

    # mesure the variables at the end
    for i in range(var_nb):
        circ.measure(qr[i + 1], cr[i])

    return circ, qr


def sat_solver(var_nb, clauses, backend=None):
    """
    If the instance is satisfiable, return one of the solutions. Else return False.
    :param var_nb: Number of variables in the problem
    :param clauses: Instance of the problem, as a list of the clauses, them being list of the variables used as int
    :param backend: Qiskit backend to use to run the algorithm. Default is local simulator
    :return:
    """

    if backend is None:
        backend = BasicAer.get_backend("qasm_simulator")

    for i in range(var_nb):
        # optimal number of iterations for 2**i solutions to the instance
        # only power of two are tested, but it is sufficient to find a solution if there is one with correct probability
        # and without hurting the complexity
        nb_grover = int(pi/4 * sqrt(2**(var_nb-i)))

        circ, _ = ex_1_3sat_circuit(var_nb, clauses, nb_grover=nb_grover, is_exclusive=False)
        job = execute(circ, backend=backend, shots=1024, max_credits=3)
        res = job.result().get_counts()

        # we take as result the most returned string bit
        most_probable = max(res, key=lambda x: res[x])[::-1]
        if evaluate(most_probable, clauses, is_exclusive=False):
            print("i :", i)
            print("Solution found: " + most_probable)
            return most_probable

    print("No solution found. Instance is unlikely to be satisfiable.")
    return False


def evaluate(val, clauses, is_exclusive=True):
    """
    See if the solution val validates the instance
    :param val: a solution, as a string of 0 and 1
    :param clauses: a SAT instance
    :param is_exclusive: whether or not the problem at hand is SAT or SAT-Exclusive.
    Determine the correctness of solutions
    :return: True if the solution validates, else False
    """
    for clause in clauses:
        total = 0

        for var in clause:
            if var < 0:
                if val[-var - 1] == '0':
                    total += 1
            else:
                if val[var - 1] == '1':
                    total += 1

        if (total != 1 and is_exclusive) or (total == 0 and not is_exclusive):
            return False
    return True


def invert(res):
    """
    Negates a string, changing 0 into 1 and vice-versa. Could probably be one-lined.
    :param res: A string of 0 or 1
    :return: the negation of res
    """
    nv_res = ""
    for v in res:
        nv_res += "0" if v == "1" else "1"
    return nv_res


def ex_1_3sat(var_nb, clauses, backend=None):
    """
    Using Grover's algorithm, exhibit the solution of an Exclusive-1 SAT problem
    :param var_nb: number of variables in the problem
    :param clauses: list of clauses
    (themselves lists of numbers, indicating the variables used in the clause with their index starting from 1,
    being negative numbers if the variable is NOTed )
    :param backend: backend to use to run the algorithm.
    :return:
    """
    circ, _ = ex_1_3sat_circuit(var_nb, clauses)

    if backend is None:
        backend = BasicAer.get_backend("qasm_simulator")

    job = execute(circ, backend=backend, shots=1024, max_credits=3)
    res = job.result().get_counts()
    # res = launch(circ, backend_type=backend, verbose=True, shots=10000)
    print(res)
    return res


clauses = [[1, 2, -3], [-1, -2, -3], [1, -2, 4], [2, -3, 4]]

sat_solver(4, clauses)
