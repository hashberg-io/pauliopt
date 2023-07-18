import itertools
import unittest

import networkx as nx
import numpy as np

from qiskit.quantum_info import Statevector

from pauliopt.topologies import Topology
from pauliopt.phase.phase_circuits import PhaseCircuit, synthesis_methods
from pauliopt.phase.cx_circuits import synthesis_methods as cx_synthesis_methods

SEED = 1337


class TestPhaseCircuitSynthesis(unittest.TestCase):

    def setUp(self):
        self.n_tests = 20
        n_gadgets = 20
        self.topology = Topology.grid(3,3)
        self.circuits = [PhaseCircuit.random(self.topology.num_qubits, n_gadgets, rng_seed=SEED) for _ in range(self.n_tests)]
        self.qiskit_circuits = [self.default_to_qiskit(c) for c in self.circuits]
    
    def default_to_qiskit(self, circuit):
        return circuit.to_qiskit(self.topology, simplified=False, method="naive")

    def test_qasm_conversion(self):
        for i in range(self.n_tests):
            with self.subTest(i=i):
                circuit = PhaseCircuit.from_qasm(self.qiskit_circuits[i].qasm())
                qc = self.default_to_qiskit(circuit)
                self.assertQCEqual(qc, self.qiskit_circuits[i])

    def test_synthesis(self):
        for i in range(self.n_tests):
            for simp in [True, False]:
                for method in synthesis_methods:
                    for cx_synth in ["naive"] + cx_synthesis_methods:
                        with self.subTest(i=i, simplified=simp, method=method, cx_synth=cx_synth):
                            synthesized = self.circuits[i].to_qiskit(self.topology, simplified=simp, method=method, cx_synth=cx_synth)
                            self.assertQCEqual(synthesized, self.qiskit_circuits[i])

    def assertNdArrEqual(self, a1, a2):
        self.assertListEqual(a1.tolist(), a2.tolist())
    
    def assertPhaseCircuitEqual(self, circuit1:PhaseCircuit, circuit2:PhaseCircuit):
        qc1 = self.default_to_qiskit(circuit1)
        qc2 = self.default_to_qiskit(circuit2)
        self.assertQCEqual(qc1, qc2)
    
    def assertQCEqual(self, circuit1, circuit2):
        qc1 = Statevector.from_instruction(circuit1)
        qc2 = Statevector.from_instruction(circuit2)
        self.assertTrue(qc1.equiv(qc2))
