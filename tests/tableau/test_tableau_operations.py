import unittest

import numpy as np
from parameterized import parameterized
from qiskit import QuantumCircuit, transpile

from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau
from pauliopt.topologies import Topology
from tests.utils import verify_equality, random_hscx_circuit

EXPECTED_STR = """T: 
Z/X I/I I/I I/I I/I 
I/I Y/Z I/I I/I I/I 
I/I I/I X/Z X/I I/I 
I/I I/I I/Z X/Z X/I 
I/I I/I I/Z I/Z X/Z 

"""


def tableau_from_circuit(tableau, circ: QuantumCircuit):
    for op in circ:
        if op.operation.name == "h":
            tableau.append_h(op.qubits[0].index)
        elif op.operation.name == "s":
            tableau.append_s(op.qubits[0].index)
        elif op.operation.name == "cx":
            tableau.append_cnot(op.qubits[0].index, op.qubits[1].index)
        else:
            raise TypeError(
                f"Unrecongnized Gate type: {op.operation.name} for Clifford Tableaus")
    return tableau


def tableau_from_circuit_prepend(tableau, circ: QuantumCircuit):
    for op in circ:
        if op.operation.name == "h":
            tableau.prepend_h(op.qubits[0].index)
        elif op.operation.name == "s":
            tableau.prepend_s(op.qubits[0].index)
        elif op.operation.name == "cx":
            tableau.prepend_cnot(op.qubits[0].index, op.qubits[1].index)
        else:
            raise TypeError(
                f"Unrecongnized Gate type: {op.operation.name} for Clifford Tableaus")
    return tableau


class TestTableauOperations(unittest.TestCase):

    @parameterized.expand([(5,), (10,), (15,)])
    def test_inverse(self, n_qubits):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        ct_ = ct.inverse()

        ct_ = ct_.inverse()

        self.assertTrue(np.allclose(ct.tableau, ct_.tableau),
                        "The twice inverted tableau did not match")

        self.assertTrue(np.allclose(ct.signs, ct_.signs),
                        "The twice inverted tableau did not match")

    @parameterized.expand([(5,), (10,), (15,)])
    def test_tableau_construction(self, n_qubits):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        ct_ = CliffordTableau.from_tableau(ct.tableau, ct.signs)

        self.assertTrue(np.allclose(ct.tableau, ct_.tableau),
                        "The twice inverted tableau did not match")

        self.assertTrue(np.allclose(ct.signs, ct_.signs),
                        "The twice inverted tableau did not match")

    @parameterized.expand([(5,), (10,), (15,)])
    def test_tableau_apply(self, n_qubits):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        circuit_ = transpile(circuit.inverse(), basis_gates=['h', 's', 'cx'])

        ct_ = CliffordTableau(n_qubits)
        ct_ = tableau_from_circuit(ct_, circuit_)

        ct = ct.apply(ct_)
        self.assertTrue(np.allclose(ct.tableau, np.eye(2 * n_qubits)),
                        "The inverted tableau did not match the identity")

    @parameterized.expand([(5,), (6,)])
    def test_tableau_construction_prepend(self, n_qubits):
        topo = Topology.complete(n_qubits)

        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit_prepend(ct, circuit)

        qc, perm = synthesize_tableau(ct, topo, include_swaps=False)

        self.assertTrue(verify_equality(circuit.reverse_ops(), qc),
                        "The Synthesized circuit does not equal "
                        "to the original with reversed ops")

    def test_string_representation(self):
        ct = CliffordTableau(5)
        ct.append_h(0)
        ct.append_s(1)
        ct.append_cnot(2, 3)
        ct.append_cnot(2, 4)
        ct.append_cnot(3, 4)

        self.assertEqual(str(ct), EXPECTED_STR,
                         "The string representation of the tableau is incorrect")
