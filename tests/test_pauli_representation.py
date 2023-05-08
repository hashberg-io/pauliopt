import unittest
import numpy as np

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import *

_PAULI_REPR = "(0.5) @ { I, X, Y, Z }\n(0.25) @ { X, X, Y, X }"


class TestPauliConversion(unittest.TestCase):
    def test_circuit_construction(self):
        pp = PauliPolynomial()

        pp >>= PPhase(0.5) @ [I, X, Y, Z]

        self.assertEqual(pp.num_qubits, 4)
        self.assertEqual(pp.size, 1)

        pp >>= PPhase(0.25) @ [X, X, Y, X]

        self.assertEqual(pp.__repr__(), _PAULI_REPR)
