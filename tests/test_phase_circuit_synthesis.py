import itertools
import unittest

import networkx as nx
import numpy as np

from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit
from qiskit.circuit.library import Permutation

from pauliopt.topologies import Topology
from pauliopt.phase.phase_circuits import PhaseCircuit, synthesis_methods
from pauliopt.phase.cx_circuits import synthesis_methods as cx_synthesis_methods

SEED = 1337

def unpermute(circuit:QuantumCircuit):
    changed_circuit = False
    if circuit.metadata:
        if "initial_layout" in circuit.metadata.keys():
            circuit.compose(Permutation(circuit.num_qubits, circuit.metadata["initial_layout"]), front=True, inplace=True)
            changed_circuit = True
        if "final_layout" in circuit.metadata.keys():
            perm = circuit.metadata["final_layout"]
            circuit.compose(Permutation(circuit.num_qubits, perm), front=False, inplace=True)
            changed_circuit = True
        if not changed_circuit:
            Warning("Attempt to unpermute did not change the circuit, no initial or final layout found.")


class TestPhaseCircuitSynthesis(unittest.TestCase):

    def setUp(self):
        self.n_tests = 20
        n_gadgets = 9
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

    def test_reallocation(self):
        for i in range(self.n_tests):
            for simp in [True, False]:
                for method in synthesis_methods:
                    for cx_synth in cx_synthesis_methods:
                        with self.subTest(i=i, simplified=simp, method=method, cx_synth=cx_synth):
                            synthesized = self.circuits[i].to_qiskit(self.topology, simplified=simp, method=method, cx_synth=cx_synth, reallocate=True)
                            unpermute(synthesized)
                            qc1 = Operator(synthesized)
                            qc2 = Operator(self.qiskit_circuits[i])
                            if not qc1.equiv(qc2):
                                print("----")
                                print(synthesized)
                                print(synthesized.metadata)
                                c, cxs = self.circuits[i].to_qiskit(self.topology, simplified=simp, method=method, return_cx=True, reallocate=True)
                                print(self.circuits[i].gadgets)
                                print(c)
                                cxs1 = cxs.to_qiskit(method="naive")
                                cxs2 = cxs.to_qiskit(method="permrowcol", reallocate=True) 
                                print(cxs1)
                                print(method)
                                unpermute(cxs2)
                                print(cxs2)
                                print("Equal CNOTS", Operator(cxs1).equiv(Operator(cxs2)))
                                print("----")
                            self.assertQCEqual(synthesized, self.qiskit_circuits[i])

    def assertNdArrEqual(self, a1, a2):
        self.assertListEqual(a1.tolist(), a2.tolist())
    
    def assertPhaseCircuitEqual(self, circuit1:PhaseCircuit, circuit2:PhaseCircuit):
        qc1 = self.default_to_qiskit(circuit1)
        qc2 = self.default_to_qiskit(circuit2)
        self.assertQCEqual(qc1, qc2)
    
    def assertQCEqual(self, circuit1, circuit2):
        qc1 = Operator(circuit1)
        qc2 = Operator(circuit2)
        self.assertTrue(qc1.equiv(qc2))
