import unittest

from qiskit import QuantumCircuit
from tests.pauli.utils import (
    generate_all_combination_pauli_polynomial,
    pauli_poly_to_tket,
    verify_equality,
    check_matching_architecture,
    get_two_qubit_count,
    generate_random_depth_1_clifford,
)
from pauliopt.circuits import CX, CY, CZ, H, S, V, Circuit
from pauliopt.gates import Vdg, Sdg, Y, X, Z, SWAP
from pauliopt.topologies import Topology


class TestPauliPropagation(unittest.TestCase):
    def test_circuit_construction(self):
        for num_qubits in [2, 3, 4]:
            for topo_creation in [Topology.line, Topology.complete]:
                pp = generate_all_combination_pauli_polynomial(n_qubits=num_qubits)

                topology = topo_creation(pp.num_qubits)
                tket_pp = pauli_poly_to_tket(pp)
                our_synth = pp.to_qiskit(topology)
                self.assertTrue(
                    verify_equality(tket_pp, our_synth),
                    "The resulting Quantum Circuits were not equivalent",
                )
                self.assertTrue(
                    check_matching_architecture(our_synth, topology.to_nx),
                    "The Pauli Polynomial did not match the architecture",
                )
                self.assertEqual(
                    get_two_qubit_count(our_synth),
                    pp.two_qubit_count(topology),
                    "Two qubit count needs to be equivalent to to two qubit count of the circuit",
                )

    def test_gate_propagation(self):
        for num_qubits in [2, 3, 4]:
            pp = generate_all_combination_pauli_polynomial(n_qubits=num_qubits)
            inital_qc = pp.to_qiskit()
            for gate_class in [H, S, CX, V, Vdg, Sdg, X, Y, Z, CY, CZ, SWAP]:
                gate = generate_random_depth_1_clifford(gate_class, num_qubits)
                gate_circ = Circuit(num_qubits)
                gate_circ.add_gate(gate)
                pp_ = pp.copy().propagate(gate)
                qc = QuantumCircuit(num_qubits)
                qc.compose(gate_circ.to_qiskit().inverse(), inplace=True)
                qc.compose(pp_.to_qiskit(), inplace=True)
                qc.compose(gate_circ.to_qiskit(), inplace=True)
                self.assertTrue(
                    verify_equality(inital_qc, qc),
                    "The resulting Quantum Circuits were not equivalent",
                )
