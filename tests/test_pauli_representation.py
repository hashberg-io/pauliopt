import unittest

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import *

_PAULI_REPR = "(0.5) @ { I, X, Y, Z }\n(0.25) @ { X, X, Y, X }"


class TestPauliConversion(unittest.TestCase):
    def test_circuit_construction(self):
        pp = PauliPolynomial(4)

        pp >>= PPhase(0.5) @ [I, X, Y, Z]

        self.assertEqual(pp.num_qubits, 4)
        self.assertEqual(len(pp), 1)

        pp >>= PPhase(0.25) @ [X, X, Y, X]

        self.assertEqual(pp.__repr__(), _PAULI_REPR)
        self.assertEqual(len(pp), 2)

        pp_ = PauliPolynomial(num_qubits=4)
        pp_ >> pp

        self.assertEqual(pp.__repr__(), pp_.__repr__(), "Right shift resulted in different pauli Polynomials.")
