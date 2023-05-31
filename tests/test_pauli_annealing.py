import itertools
import unittest

import networkx as nx
import numpy as np
import pytket
from pytket._tket.circuit import PauliExpBox
from pytket._tket.pauli import Pauli
from pytket._tket.transform import Transform
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit
from qiskit import QuantumCircuit

from pauliopt.pauli.anneal import anneal
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import X, Y, Z, I
from pauliopt.topologies import Topology
from pauliopt.utils import pi

PAULI_TO_TKET = {
    X: Pauli.X,
    Y: Pauli.Y,
    Z: Pauli.Z,
    I: Pauli.I
}


def tket_to_qiskit(circuit: pytket.Circuit) -> QuantumCircuit:
    return tk_to_qiskit(circuit)


def pauli_poly_to_tket(pp: PauliPolynomial):
    circuit = pytket.Circuit(pp.num_qubits)
    for gadget in pp.pauli_gadgets:
        circuit.add_pauliexpbox(
            PauliExpBox([PAULI_TO_TKET[p] for p in gadget.paulis],
                        gadget.angle.to_qiskit / np.pi),
            list(range(pp.num_qubits)))
    Transform.DecomposeBoxes().apply(circuit)
    return tket_to_qiskit(circuit)


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


def generate_all_combination_pauli_polynomial(n_qubits=2):
    allowed_angels = [2 * pi, pi, pi / 2, pi / 4, pi / 8]
    pp = PauliPolynomial(n_qubits)
    for comb in itertools.product([X, Y, Z, I], repeat=n_qubits):
        pp >>= PPhase(np.random.choice(allowed_angels)) @ list(comb)
    return pp


def check_matching_architecture(qc: QuantumCircuit, G: nx.Graph):
    for gate in qc:
        if gate.operation.num_qubits == 2:
            ctrl, target = gate.qubits
            ctrl, target = ctrl._index, target._index  # TODO refactor this to a non deprecated way
            if not G.has_edge(ctrl, target):
                return False
    return True


def get_two_qubit_count(circ: QuantumCircuit):
    ops = circ.count_ops()
    two_qubit_count = 0
    two_qubit_ops = ["cx", "cy", "cz"]
    for op_key in two_qubit_ops:
        if op_key in ops.keys():
            two_qubit_count += ops[op_key]

    return two_qubit_count


class TestPauliAnnealing(unittest.TestCase):
    def test_simulated_annealing_pauli(self):
        """
        Checks in this Unit test:
        1) If one constructs the Pauli Polynomial with our libary the circuits should match the ones of tket
        2) When synthesizing onto a different architecture the circuits should match the ones of tket
        3) Check that our to_qiskit method exports the Pauli Polynomial according to an architecture
        """
        for num_qubits in [2, 3, 4]:
            for topo_creation in [Topology.line, Topology.complete]:
                pp = generate_all_combination_pauli_polynomial(n_qubits=num_qubits)

                topology = topo_creation(pp.num_qubits)
                tket_pp = pauli_poly_to_tket(pp)
                our_synth = anneal(pp, topology)

                self.assertTrue(verify_equality(tket_pp, our_synth),
                                "The annealing version returned a wrong circuit")
