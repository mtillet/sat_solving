from IBMQuantumExperience import IBMQuantumExperience
from qiskit import IBMQ, BasicAer, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor, backend_overview

import numpy as np
from math import cos, sin, pi, log2
from matplotlib import pyplot as plt
from itertools import product
import copy
from qiskit.providers import BaseBackend
from qiskit.transpiler import PassManager

from qiskit.aqua.aqua_error import AquaError
from qiskit.aqua._discover import (local_pluggables,
                                   PluggableType,
                                   get_pluggable_class)
from qiskit.aqua.utils.json_utils import convert_json_to_dict
from qiskit.aqua.parser._inputparser import InputParser
from qiskit.aqua.parser import JSONSchema
from qiskit.aqua import QuantumInstance
from qiskit.aqua.qiskit_aqua_globals import aqua_globals
from qiskit.aqua.utils.backend_utils import (get_backend_from_provider,
                                             get_provider_from_backend,
                                             is_statevector_backend)
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter


# IBMQ.load_accounts()
print("Account loaded")


def launch(circ, shots=1024, backend_type="local", state_vectors=False, max_credits=3, verbose=False):
    if verbose:
        print("Depth : ", circ.depth())
        print("Qbits : ", circ.width())
        print("Number of gates : ", circ.count_ops())
        # print("Total : ", circ.number_atomic_gates())

    if backend_type == "local":
        if state_vectors:
            simulator = 'statevector_simulator'
        else:
            simulator = 'qasm_simulator'
        backend = BasicAer.get_backend(simulator)
    elif backend_type in ["q", "quantum"]:
        min_qbits = circ.width()
        print("Available backends:")
        IBMQ.backends()

        large_enough_devices = IBMQ.backends(
            filters=lambda x: x.configuration().n_qubits > min_qbits and not x.configuration().simulator)
        backend = least_busy(large_enough_devices)
        print("The best backend is " + backend.name())
    elif backend_type in ["hpc", "ibm_sim"]:
        backend = IBMQ.backends(filters=lambda x: x.configuration().simulator)[0]
    else:
        print("Invalid backend name. Switching to local simulator")
        backend = BasicAer.get_backend('qasm_simulator')

    job = execute(circ, backend=backend, shots=shots, max_credits=max_credits)

    if backend_type in ["q", "quantum"]:
        job_monitor(job)
    result = job.result()

    counts = result.get_counts(circ)
    # print(counts)
    return result


def backends_info():
    backend_overview()


def show_credits():
    api = IBMQuantumExperience(IBMQ.stored_accounts()[0]['token'])
    print(api.get_my_credits())


def remove_same(counts):
    to_remove = []
    keys = counts.keys()
    keys = [key for key in keys]
    for key in keys:
        if key[0] == '1':
            rev = inverted(key)
            if rev in counts.keys():
                counts[rev] += counts[key]
            else:
                counts[rev] = counts[key]
            to_remove.append(key)
    for key in to_remove:
        del counts[key]


def inverted(key):
    res = ''
    for v in key:
        if v == '1':
            res += '0'
        else:
            res += '1'
    return res


def plot_hist(data, crop=False, state_vect=False, remove_sym=False):
    if state_vect:
        t = ["".join(seq)[::-1] for seq in product("01", repeat=int(log2(len(data))))]
        d = {}
        for ind, val in zip(t, data):
            d[ind] = int(abs(val)**2 * 1024)
        data = d

    if remove_sym:
        remove_same(data)
    names = [""]*len(data)
    val = [0]*len(data)
    maxi = max(data.values())
    for ind, key in enumerate(data.keys()):
        if not crop or data[key] > maxi/10:
            names[ind] = key

    names.sort()
    for ind, key in enumerate(names):
        val[ind] = data[key]

    ind = np.arange(len(data))  # the x locations for the groups
    width = 0.9  # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(ind, val, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_xticks(ind)
    ax.set_xticklabels(names)

    def autolabel(rectangles):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rectangles:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects)

    plt.show()


def cnx_o_paquet(c, r, cont, target, losts):
    """Adds a len(cont)-qbits-Control(X) gate to c, on r, controlled by the List cont at qbit target,
    losing one qbit in the process"""
    # c.barrier(r)
    # print(len(cont), len(losts))
    if len(cont) == 1:
        c.cx(cont[0], target)
    elif len(cont) == 2:
        c.ccx(cont[0], cont[1], target)
    elif len(cont) == 3:  # and len(losts) == 1:
        c.ccx(cont[2], cont[1], losts[0])
        c.ccx(cont[0], losts[0], target)
        c.ccx(cont[2], cont[1], losts[0])
        c.ccx(cont[0], losts[0], target)
    # elif len(cont) == 4 and len(losts) == 1:
    #    cnx_o(c, r, cont, target, losts[0])
    # elif len(cont) == 5:
    #    cnx_o(c, r, cont, target, losts[0])
    else:
        m = len(cont)
        # print(m)
        # c.barrier(r)
        c.ccx(cont[0], losts[0], target)
        # c.barrier(r)
        # c.h(r)

        for i in range(1, m - 2):
            c.ccx(cont[i], losts[i], losts[i - 1])
        # c.barrier(r)
        c.ccx(cont[-1], cont[-2], losts[m - 3])
        # c.barrier(r)
        for i in range(m - 3, 0, -1):
            c.ccx(cont[i], losts[i], losts[i - 1])
        # c.barrier(r)
        c.ccx(cont[0], losts[0], target)
        # c.barrier(r)
        for i in range(1, m - 2):
            c.ccx(cont[i], losts[i], losts[i - 1])
        # c.barrier(r)
        c.ccx(cont[-1], cont[-2], losts[m - 3])
        # c.barrier(r)
        for i in range(m - 3, 0, -1):
            c.ccx(cont[i], losts[i], losts[i - 1])
    return 1


def cnx_o(c, r, cont, target, lost):
    """Adds a len(cont)-qbits-Control(X) gate to c, on r, controlled by the List cont at qbit target,
    losing one qbit in the process"""
    # c.barrier(r)
    if len(cont) == 1:
        c.cx(cont[0], target)
    elif len(cont) == 2:
        c.ccx(cont[0], cont[1], target)
    elif len(cont) == 3:
        c.ccx(cont[2], cont[1], lost)
        c.ccx(cont[0], lost, target)
        c.ccx(cont[2], cont[1], lost)
        c.ccx(cont[0], lost, target)
    else:
        m = int(np.ceil(len(cont) / 2 + 1))
        m1 = len(cont) - m
        # print(m, len(cont)+2, len(cont), m1)
        # A more efficient way to do this would be defining a new circuit
        # and just multiply it istead of doing the same thing twice
        cnx_o_paquet(c, r, cont[m1:], lost, [target] + cont[:m1])
        cnx_o_paquet(c, r, [lost] + cont[:m1], target, cont[m1:])
        cnx_o_paquet(c, r, cont[m1:], lost, [target] + cont[:m1])
        cnx_o_paquet(c, r, [lost] + cont[:m1], target, cont[m1:])
    return 1


def show_graph(graph):
    n = len(graph)
    for i, liste in enumerate(graph):
        for j, weight in enumerate(liste):
            if weight != 0:
                if weight > 0:
                    plt.plot([cos(2*pi*i/n), cos(2*pi*j/n)], [sin(2*pi*i/n), sin(2*pi*j/n)],
                             color='g', linewidth=weight/5)
                else:
                    plt.plot([cos(2*pi*i/n), cos(2*pi*j/n)], [sin(2*pi*i/n), sin(2*pi*j/n)],
                             color='r', linewidth=-weight/5)

    plt.show()


def graph_circ(n, vertex_list=False):
    res = []
    if not vertex_list:
        for i in range(n):
            l = [0]*n
            l[i-1] = 1
            l[(i+1) % n] = 1
            res.append(l)
    else:
        for i in range(n):
            res.append([(i+n-1) % n, (i+1) % n])
    return np.asarray(res)


def graph_complete(n):
    return np.ones(n) - np.eye(n)


def get_pairs(graph_mat):
    """
    Convert the matrix rep of a graph to a list of all vertices
    :param graph_mat:
    :return:
    """
    res = []
    n = len(graph_mat)
    for i in range(n):
        for j in range(i+1, n):
            if graph_mat[i][j]:
                res.append([i, j])
    return res


def get_quantum_instance(params, algo_input=None, backend=None):
    """
    TODO màj doc
    ALMOST TOTALLY COPIED FROM AQUA (_aqua->run_algorithm, beginning)

    Run algorithm as named in params, using params and algo_input as input data
    and returning a result dictionary

    Args:
        params (dict): Dictionary of params for algo and dependent objects
        algo_input (AlgorithmInput): Main input data for algorithm. Optional, an algo may run entirely from params
        backend (BaseBackend): Backend object to be used in place of backend name

    Returns:
        Result dictionary containing result of algorithm computation
    """

    inputparser = InputParser(params)
    inputparser.parse()
    # before merging defaults attempts to find a provider for the backend in case no
    # provider was passed
    if backend is None and inputparser.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER) is None:
        backend_name = inputparser.get_section_property(JSONSchema.BACKEND, JSONSchema.NAME)
        if backend_name is not None:
            inputparser.set_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER, get_provider_from_backend(backend_name))

    inputparser.validate_merge_defaults()

    algo_name = inputparser.get_section_property(PluggableType.ALGORITHM.value, JSONSchema.NAME)
    if algo_name is None:
        raise AquaError('Missing algorithm name')

    if algo_name not in local_pluggables(PluggableType.ALGORITHM):
        raise AquaError('Algorithm "{0}" missing in local algorithms'.format(algo_name))

    if algo_input is None:
        input_name = inputparser.get_section_property('input', JSONSchema.NAME)
        if input_name is not None:
            input_params = copy.deepcopy(inputparser.get_section_properties('input'))
            del input_params[JSONSchema.NAME]
            convert_json_to_dict(input_params)
            algo_input = get_pluggable_class(PluggableType.INPUT, input_name).from_params(input_params)

    algo_params = copy.deepcopy(inputparser.get_sections())
    algorithm = get_pluggable_class(PluggableType.ALGORITHM,
                                    algo_name).init_params(algo_params, algo_input)
    random_seed = inputparser.get_section_property(JSONSchema.PROBLEM, 'random_seed')
    algorithm.random_seed = random_seed
    quantum_instance = None
    # setup backend
    backend_provider = inputparser.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER)
    backend_name = inputparser.get_section_property(JSONSchema.BACKEND, JSONSchema.NAME)
    if backend_provider is not None and backend_name is not None:  # quantum algorithm
        backend_cfg = {k: v for k, v in inputparser.get_section(JSONSchema.BACKEND).items() if k not in [JSONSchema.PROVIDER, JSONSchema.NAME]}
        # TODO, how to build the noise model from a dictionary?
        # backend_cfg.pop('noise_params', None)
        backend_cfg['seed'] = random_seed
        backend_cfg['seed_mapper'] = random_seed
        pass_manager = PassManager() if backend_cfg.pop('skip_transpiler', False) else None
        if pass_manager is not None:
            backend_cfg['pass_manager'] = pass_manager

        if backend is None or not isinstance(backend, BaseBackend):
            backend = get_backend_from_provider(backend_provider, backend_name)
        backend_cfg['backend'] = backend

        # overwrite the basis_gates and coupling_map
        basis_gates = backend_cfg.pop('basis_gates', None)
        coupling_map = backend_cfg.pop('coupling_map', None)
        if backend.configuration().simulator:
            if basis_gates is not None:
                backend.configuration().basis_gates = basis_gates
            if coupling_map is not None:
                backend.configuration().coupling_map = coupling_map

        quantum_instance = QuantumInstance(**backend_cfg)

    return quantum_instance


def get_quantum_instance_2(params, algo_input=None, quantum_instance=None):
    parser = InputParser(params)
    parser.parse()
    # before merging defaults attempts to find a provider for the backend in case no
    # provider was passed
    if quantum_instance is None and parser.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER) is None:
        backend_name = parser.get_section_property(JSONSchema.BACKEND, JSONSchema.NAME)
        if backend_name is not None:
            parser.set_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER,
                                        get_provider_from_backend(backend_name))

    # check quantum_instance parameter
    backend = None
    if isinstance(quantum_instance, BaseBackend):
        backend = quantum_instance
    elif quantum_instance is not None:
        raise AquaError('Invalid QuantumInstance or BaseBackend parameter {}.'.format(quantum_instance))

    # set provider and name in input file for proper backend schema dictionary build
    if backend is not None:
        parser.add_section_properties(JSONSchema.BACKEND,
                                      {
                                        JSONSchema.PROVIDER: get_provider_from_backend(backend),
                                        JSONSchema.NAME: backend.name(),
                                      })

    parser.validate_merge_defaults()

    algo_name = parser.get_section_property(PluggableType.ALGORITHM.value, JSONSchema.NAME)
    if algo_name is None:
        raise AquaError('Missing algorithm name')

    if algo_name not in local_pluggables(PluggableType.ALGORITHM):
        raise AquaError('Algorithm "{0}" missing in local algorithms'.format(algo_name))

    if algo_input is None:
        input_name = parser.get_section_property('input', JSONSchema.NAME)
        if input_name is not None:
            input_params = copy.deepcopy(parser.get_section_properties('input'))
            del input_params[JSONSchema.NAME]
            convert_json_to_dict(input_params)
            algo_input = get_pluggable_class(PluggableType.INPUT, input_name).from_params(input_params)

    algo_params = copy.deepcopy(parser.get_sections())
    num_processes = parser.get_section_property(JSONSchema.PROBLEM, 'num_processes')
    aqua_globals.num_processes = num_processes if num_processes is not None else aqua_globals.CPU_COUNT
    random_seed = parser.get_section_property(JSONSchema.PROBLEM, 'random_seed')
    aqua_globals.random_seed = random_seed
    if quantum_instance is not None:
        return

    # setup backend
    backend_provider = parser.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER)
    backend_name = parser.get_section_property(JSONSchema.BACKEND, JSONSchema.NAME)
    if backend_provider is not None and backend_name is not None:  # quantum algorithm
        if backend is None:
            backend = get_backend_from_provider(backend_provider, backend_name)

        backend_cfg = {k: v for k, v in parser.get_section(JSONSchema.BACKEND).items() if
                       k not in [JSONSchema.PROVIDER, JSONSchema.NAME]}

        # set shots for state vector
        if is_statevector_backend(backend):
            backend_cfg['shots'] = 1

        # check coupling map
        if 'coupling_map_from_device' in backend_cfg:
            coupling_map_from_device = backend_cfg.get('coupling_map_from_device')
            del backend_cfg['coupling_map_from_device']
            if coupling_map_from_device is not None:
                names = coupling_map_from_device.split(':')
                if len(names) == 2:
                    device_backend = get_backend_from_provider(names[0], names[1])
                    device_coupling_map = device_backend.configuration().coupling_map
                    if device_coupling_map is not None:
                        coupling_map = backend_cfg.get('coupling_map')
                        if coupling_map is None:
                            backend_cfg['coupling_map'] = device_coupling_map

        # check noise model
        if 'noise_model' in backend_cfg:
            noise_model = backend_cfg.get('noise_model')
            del backend_cfg['noise_model']
            if noise_model is not None:
                names = noise_model.split(':')
                if len(names) == 2:
                    # Generate an Aer noise model for device
                    from qiskit.providers.aer import noise
                    device_backend = get_backend_from_provider(names[0], names[1])
                    noise_model = noise.device.basic_device_noise_model(device_backend.properties())
                    noise_basis_gates = None
                    if noise_model is not None and noise_model.basis_gates is not None:
                        noise_basis_gates = noise_model.basis_gates
                        noise_basis_gates = noise_basis_gates.split(',') if isinstance(noise_basis_gates,
                                                                                       str) else noise_basis_gates
                    if noise_basis_gates is not None:
                        basis_gates = backend_cfg.get('basis_gates')
                        if basis_gates is None:
                            backend_cfg['basis_gates'] = noise_basis_gates

        backend_cfg['seed_transpiler'] = random_seed
        pass_manager = PassManager() if backend_cfg.pop('skip_transpiler', False) else None
        if pass_manager is not None:
            backend_cfg['pass_manager'] = pass_manager

        backend_cfg['backend'] = backend
        if random_seed is not None:
            backend_cfg['seed'] = random_seed
        skip_qobj_validation = parser.get_section_property(JSONSchema.PROBLEM, 'skip_qobj_validation')
        if skip_qobj_validation is not None:
            backend_cfg['skip_qobj_validation'] = skip_qobj_validation

        circuit_caching = parser.get_section_property(JSONSchema.PROBLEM, 'circuit_caching')
        if circuit_caching is not None:
            backend_cfg['circuit_caching'] = circuit_caching

        skip_qobj_deepcopy = parser.get_section_property(JSONSchema.PROBLEM, 'skip_qobj_deepcopy')
        if skip_qobj_deepcopy is not None:
            backend_cfg['skip_qobj_deepcopy'] = skip_qobj_deepcopy

        cache_file = parser.get_section_property(JSONSchema.PROBLEM, 'circuit_cache_file')
        if cache_file is not None:
            backend_cfg['cache_file'] = cache_file

        measurement_error_mitigation = parser.get_section_property(JSONSchema.PROBLEM,
                                                                   'measurement_error_mitigation')
        if measurement_error_mitigation:
            backend_cfg['measurement_error_mitigation_cls'] = CompleteMeasFitter

        return QuantumInstance(**backend_cfg)


def show_tsp(tsp_ins):
    for i in range(tsp_ins.dim):
        for j in range(i):
            print(i, j)
            plt.plot([tsp_ins.coord[i][0], tsp_ins.coord[j][0]],
                     [tsp_ins.coord[i][1], tsp_ins.coord[j][1]],
                     linewidth=tsp_ins.w[i][j]/10)

    plt.show()
