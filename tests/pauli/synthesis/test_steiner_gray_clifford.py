import unittest

from pauliopt.pauli.synthesis.annealing import annealing_synthesis
from pauliopt.pauli.synthesis.steiner_gray_synthesis import (
    pauli_polynomial_steiner_gray_clifford,
)
from pauliopt.topologies import Topology
from tests.pauli.utils import (
    generate_random_pauli_polynomial,
    verify_equality,
    apply_permutation,
)


class TestPauliSteinerGraySynthesis(unittest.TestCase):
    def test_steiner_gray_clifford(self):
        for num_gadgets in [100, 200]:
            for topo in [
                Topology.complete(4),
                Topology.line(4),
                Topology.line(6),
                Topology.cycle(4),
                Topology.grid(2, 4),
            ]:
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                pp_ = pp.copy()
                circ_out, gadget_perm, perm = pauli_polynomial_steiner_gray_clifford(
                    pp, topo
                )

                pp_.pauli_gadgets = [pp_[i].copy() for i in gadget_perm]
                circ_out = apply_permutation(circ_out.to_qiskit(), perm)

                self.assertTrue(verify_equality(circ_out, pp_.to_qiskit()))
