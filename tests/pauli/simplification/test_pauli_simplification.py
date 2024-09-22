import unittest

from pauliopt.pauli.simplification.simple_simplify import simplify_pauli_polynomial
from tests.pauli.utils import generate_random_pauli_polynomial, verify_equality


class TestPauliSimplification(unittest.TestCase):

    def test_simplification_process(self):
        for num_qubits in [4, 6]:
            for phase_gadget in [100, 200, 300, 400]:
                pp = generate_random_pauli_polynomial(num_qubits, phase_gadget)

                pp_ = simplify_pauli_polynomial(pp)
                print()
                self.assertTrue(
                    verify_equality(pp_.to_qiskit(), pp.to_qiskit()),
                    "Resulting circuits where not equal",
                )
