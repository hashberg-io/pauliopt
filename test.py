import networkx as nx
from matplotlib import pyplot as plt

from pauliopt.pauli.anneal import anneal
from pauliopt.pauli.clifford_gates import CX, H, CY, CZ
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
import numpy as np

from pauliopt.pauli.utils import Pauli, _pauli_to_string
import stim

def two_qubit_count(count_ops):
    count = 0
    count += count_ops["cx"] if "cx" in count_ops.keys() else 0
    count += count_ops["cy"] if "cy" in count_ops.keys() else 0
    count += count_ops["cz"] if "cz" in count_ops.keys() else 0
    return count


def create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels):
    angle = np.random.choice(allowed_angels)
    nr_legs = np.random.randint(min_legs, max_legs)
    legs = np.random.choice([i for i in range(num_qubits)], size=nr_legs, replace=False)
    phase_gadget = [Pauli.I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice([Pauli.X, Pauli.Y, Pauli.Z])
    return PPhase(angle) @ phase_gadget


def generate_random_pauli_polynomial(num_qubits: int, num_gadgets: int, min_legs=None,
                                     max_legs=None, allowed_angels=None):
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.125 * np.pi]

    pp = PauliPolynomial()
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels)

    return pp


def verify_equality(qc_in, qc_out):
    """
    Verify the equality up to a global phase
    :param qc_in:
    :param qc_out:
    :return:
    """
    try:
        from qiskit.quantum_info import Statevector
    except:
        raise Exception("Please install qiskit to compare to quantum circuits")
    return Statevector.from_instruction(qc_in) \
        .equiv(Statevector.from_instruction(qc_out))


def main(num_qubits=3):
    tableau = stim.Tableau(3)
    print(tableau)
    x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy(bit_packed=False)
    print(x2x)
    print(x2z)
    x_arr = np.concatenate([x2x, x2z])
    print(x_arr)






def create_rules_graph(rules):
    rules_tuple = []
    for k, v in rules.items():
        res_key = _pauli_to_string(v[0]) + _pauli_to_string(v[1])
        rules_tuple.append((k, res_key))
    G = nx.Graph()
    print(rules_tuple)
    G.add_edges_from(rules_tuple)
    return G


if __name__ == '__main__':
    main()
