import unittest

from pauliopt.pauli.synthesis import PauliSynthesizer, SynthMethod
from pauliopt.topologies import Topology
from tests.pauli.utils import generate_random_pauli_polynomial


class TestPauliSynthesis(unittest.TestCase):
    def test_uccds(self):
        for num_gadgets in [100, 200]:
            for topo in [
                Topology.line(4),
                Topology.line(8),
                Topology.cycle(4),
                Topology.cycle(8),
                Topology.grid(2, 3),
            ]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                synthesizer = PauliSynthesizer(pp, SynthMethod.UCCDS, topo)
                synthesizer.synthesize()
                self.assertTrue(
                    synthesizer.check_circuit_equivalence(), "Circuits did not match"
                )
                self.assertTrue(
                    synthesizer.check_connectivity_predicate(),
                    "Connectivity predicate not satisfied",
                )

    def test_divide_and_conquer(self):
        for num_gadgets in [10, 30]:
            for topo in [
                Topology.line(4),
                Topology.cycle(4),
                Topology.complete(4),
                Topology.grid(2, 3),
            ]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                synthesizer = PauliSynthesizer(pp, SynthMethod.DIVIDE_AND_CONQUER, topo)
                synthesizer.synthesize()
                self.assertTrue(
                    synthesizer.check_circuit_equivalence(), "Circuits did not match"
                )
                self.assertTrue(
                    synthesizer.check_connectivity_predicate(),
                    "Connectivity predicate not satisfied",
                )

    def test_steiner_gray_nc(self):
        for num_gadgets in [5]:
            for topo in [
                Topology.line(4),
                Topology.line(6),
                Topology.cycle(4),
                Topology.grid(2, 4),
            ]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                synthesizer = PauliSynthesizer(pp, SynthMethod.STEINER_GRAY_NC, topo)
                synthesizer.synthesize()
                self.assertTrue(
                    synthesizer.check_circuit_equivalence(), "Circuits did not match"
                )
                self.assertTrue(
                    synthesizer.check_connectivity_predicate(),
                    "Connectivity predicate not satisfied",
                )

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
                synthesizer = PauliSynthesizer(
                    pp, SynthMethod.STEINER_GRAY_CLIFFORD, topo
                )
                synthesizer.synthesize()
                self.assertTrue(
                    synthesizer.check_circuit_equivalence(), "Circuits did not match"
                )
                self.assertTrue(
                    synthesizer.check_connectivity_predicate(),
                    "Connectivity predicate not satisfied",
                )

    def test_pauli_annealing(self):
        for num_gadgets in [100, 200]:
            for topo in [
                Topology.line(4),
                Topology.line(6),
                Topology.cycle(4),
                Topology.grid(2, 4),
            ]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                synthesizer = PauliSynthesizer(pp, SynthMethod.ANNEAL, topo)
                synthesizer.synthesize()
                self.assertTrue(
                    synthesizer.check_circuit_equivalence(), "Circuits did not match"
                )
                self.assertTrue(
                    synthesizer.check_connectivity_predicate(),
                    "Connectivity predicate not satisfied",
                )
