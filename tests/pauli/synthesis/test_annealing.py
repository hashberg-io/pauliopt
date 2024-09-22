import unittest

from pauliopt.pauli.synthesis.annealing import annealing_synthesis
from pauliopt.topologies import Topology
from tests.pauli.utils import generate_random_pauli_polynomial, verify_equality


class TestPauliSynthesis(unittest.TestCase):
    def test_pauli_annealing(self):
        for num_gadgets in [100]:
            for topo in [
                Topology.line(4),
                Topology.line(6),
                Topology.cycle(4),
                Topology.grid(2, 4),
            ]:
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                pp_ = pp.copy()
                qc_out = annealing_synthesis(pp, topo).to_qiskit()

                self.assertTrue(
                    verify_equality(qc_out, pp_.to_qiskit()), "Circuits did not match"
                )
