import os
import unittest
import random
from parameterized import parameterized

from pauliopt.circuits import Circuit
from pauliopt.gates import (
    H,
    S,
    Sdg,
    X,
    Y,
    Z,
    T,
    Tdg,
    SWAP,
    CX,
    CY,
    CZ,
    CCX,
    CCZ,
    Rx,
    Ry,
    Rz,
    CRx,
    CRy,
    CRz,
)
from pauliopt.utils import pi
from tests.utils import verify_equality, random_circuit, apply_permutation

GATES_TO_TEST = [
    H(0),
    S(0),
    Sdg(0),
    X(0),
    Y(0),
    Z(0),
    T(0),
    Tdg(0),
    SWAP(0, 1),
    CX(0, 1),
    CY(0, 1),
    CZ(0, 1),
    CCX(0, 1, 2),
    CCZ(0, 1, 2),
    Rx(pi / 2, 0),
    Ry(pi / 2, 0),
    Rz(pi / 2, 0),
    CRx(pi / 2, 0, 1),
    CRy(pi / 2, 0, 1),
    CRz(pi / 2, 0, 1),
]


class TestCircuitConstruction(unittest.TestCase):
    @parameterized.expand(
        [
            (5,),
            (6,),
            (7,),
            (8,),
        ]
    )
    def test_circuit_construction(self, nr_qubits):
        qc = random_circuit(nr_qubits=nr_qubits, nr_gates=1000)

        circ = Circuit.from_qiskit(qc)

        qc_ = circ.to_qiskit()

        self.assertTrue(
            verify_equality(qc, qc_), "The converted circuit does not equal to original"
        )

    @parameterized.expand(GATES_TO_TEST)
    def test_circuit_representation_cli(self, gate):
        circ = Circuit(3).add_gate(gate)

        with open(f"{os.getcwd()}/tests/data/circ_reps/{gate.name}_cli.txt", "r") as f:
            self.assertEqual(
                f.read(),
                circ.__str__(),
                f"The CLI representation of {gate.name} " f"is incorrect or changed!",
            )

        with open(f"{os.getcwd()}/tests/data/circ_reps/{gate.name}_svg.txt", "r") as f:
            self.assertEqual(
                f.read(),
                circ._to_svg(svg_code_only=True),
                f"The SVG representation of {gate.name} " f"is incorrect or changed!",
            )

    @parameterized.expand([(3,), (4,), (5,)])
    def test_circuit_inversion(self, nr_qubits):
        qc = random_circuit(nr_qubits=nr_qubits, nr_gates=1000)

        circ = Circuit.from_qiskit(qc)

        self.assertTrue(verify_equality(qc.inverse(), circ.inverse().to_qiskit()))

    @parameterized.expand([(3,), (4,), (5,)])
    def test_apply_permutation(self, nr_qubits):
        permutation = random.sample(list(range(nr_qubits)), nr_qubits)
        qc = random_circuit(nr_qubits=nr_qubits, nr_gates=10)

        circ = Circuit.from_qiskit(qc)
        circ.apply_permutation(permutation)
        qc = apply_permutation(qc, permutation)
        self.assertTrue(verify_equality(qc, circ.to_qiskit()))
