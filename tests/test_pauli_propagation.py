import itertools
import unittest

import networkx as nx
import numpy as np
import pytket
from pytket._tket.circuit import PauliExpBox
from pytket._tket.transform import Transform
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit, qiskit_to_tk
from pytket._tket.pauli import Pauli
from qiskit import QuantumCircuit

from pauliopt.pauli.clifford_gates import CX, CY, CZ, H, S, V
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import X, Y, Z, I

from pauliopt.topologies import Topology


def pauli_to_tket_pauli(pauli):
    if pauli == X:
        return Pauli.X
    elif pauli == Y:
        return Pauli.Y
    elif pauli == Z:
        return Pauli.Z
    elif pauli == I:
        return Pauli.I
    else:
        raise Exception("Unknown Pauli Matrix")


def tket_to_qiskit(circuit: pytket.Circuit) -> QuantumCircuit:
    return tk_to_qiskit(circuit)


def pauli_poly_to_tket(pp: PauliPolynomial):
    circuit = pytket.Circuit(pp.num_qubits)
    for gadget in pp.pauli_gadgets:
        circuit.add_pauliexpbox(
            PauliExpBox([pauli_to_tket_pauli(p) for p in gadget.paulis],
                        gadget.angle / np.pi),
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
    allowed_angels = [2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.125 * np.pi]
    pp = PauliPolynomial()
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


class TestPauliConversion(unittest.TestCase):
    def test_circuit_construction(self):
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
                our_synth = pp.to_qiskit(topology)
                self.assertTrue(verify_equality(tket_pp, our_synth),
                                "The resulting Quantum Circuits were not equivalent")
                self.assertTrue(check_matching_architecture(our_synth, topology.to_nx),
                                "The Pauli Polynomial did not match the architecture")

    def test_gate_propagation(self):
        """
        Checks if the clifford Propagation rules are sound for 2, 3, 4 qubits
        """
        for num_qubits in [2, 3, 4]:
            pp = generate_all_combination_pauli_polynomial(n_qubits=num_qubits)
            inital_qc = pp.to_qiskit()
            for gate_class in [CX, CY, CZ, H, S, V]:
                gate = gate_class.generate_random(num_qubits)
                print(gate.to_qiskit())
                pp_ = pp.copy().propagate(gate)
                qc = QuantumCircuit(num_qubits)
                qc.compose(gate.to_qiskit().inverse(), inplace=True)
                qc.compose(pp_.to_qiskit(), inplace=True)
                qc.compose(gate.to_qiskit(), inplace=True)
                self.assertTrue(verify_equality(inital_qc, qc),
                                "The resulting Quantum Circuits were not equivalent")
