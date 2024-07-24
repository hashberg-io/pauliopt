import unittest

from pauliopt.pauli.synthesis.synthesis_divide_and_conquer import (
    synthesis_divide_and_conquer,
)
from pauliopt.topologies import Topology
from tests.pauli.utils import (
    generate_random_pauli_polynomial,
    verify_equality,
    apply_permutation,
)


class TestPauliSynthesis(unittest.TestCase):
    def test_pauli_annealing(self):
        for num_gadgets in [100, 200]:
            for topo in [
                Topology.complete(4),
                Topology.line(6),
                Topology.cycle(4),
                Topology.grid(2, 4),
            ]:
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                pp_ = pp.copy()

                qc_out, permutation = synthesis_divide_and_conquer(pp, topo)

                qc_out = qc_out.to_qiskit()
                qc_out = apply_permutation(qc_out, permutation)

                self.assertTrue(
                    verify_equality(qc_out, pp_.to_qiskit()), "Circuits did not match"
                )
