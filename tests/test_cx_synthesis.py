import itertools
import unittest

import networkx as nx
import numpy as np

from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit
from qiskit.circuit.library import Permutation

from pauliopt.topologies import Topology
from pauliopt.phase.cx_circuits import CXCircuit, CXCircuitLayer, synthesis_methods

SEED = 42


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


class TestCXSynthesis(unittest.TestCase):

    def setUp(self):
        self.n_tests = 20
        self.topology = Topology.grid(3, 3)
        edges = [c.as_pair for c in self.topology.couplings]
        edges += [(c, p) for c, p in edges]
        depth = 20
        np.random.seed(SEED)
        self.circuit = [CXCircuit(self.topology,
                                  [CXCircuitLayer(self.topology, [edges[i]]) for i in
                                   np.random.choice(len(edges), depth, replace=True)])
                        for _ in range(self.n_tests)]
        self.col_matrix = [c.parity_matrix(parities_as_columns=True) for c in
                           self.circuit]
        self.row_matrix = [c.parity_matrix(parities_as_columns=False) for c in
                           self.circuit]


    def test_transposed(self):
        for i in range(self.n_tests):
            with self.subTest(i=i):
                self.assertNdArrEqual(self.col_matrix[i], self.row_matrix[i].T)

    def test_synthesis(self):
        for i in range(self.n_tests):
            for wise in [True, False]:
                if wise:
                    matrix = self.col_matrix[i]
                else:
                    matrix = self.row_matrix[i]
                for method in synthesis_methods:
                    with self.subTest(i=i, parity_as_columns=wise, method=method):
                        synthesized_circuit = CXCircuit.from_parity_matrix(matrix,
                                                                           self.topology,
                                                                           parities_as_columns=wise,
                                                                           reallocate=False,
                                                                           method=method)
                        synth_matrix = synthesized_circuit.parity_matrix(wise)
                        self.assertNdArrEqual(synth_matrix, matrix)

    def test_synthesis_alt(self):
        for i in range(self.n_tests):
            for col_wise in [True, False]:
                for method in synthesis_methods:
                    with self.subTest(i=i, method=method, parities_as_columns=col_wise):
                        baseline = self.circuit[i].to_qiskit("naive", reallocate=False)
                        synthesized = self.circuit[i].to_qiskit(method, reallocate=False, parities_as_columns=col_wise)
                        self.assertQCEqual(baseline, synthesized)



    def test_reallocation(self):
        for i in range(self.n_tests):
            for wise in [True, False]:
                if wise:
                    matrix = self.col_matrix[i]
                else:
                    matrix = self.row_matrix[i]

                for method in synthesis_methods: # TODO when more synthesis methods are available, maybe not all are up-to-permutation.
                    with self.subTest(i=i, parity_as_columns=wise, method=method):
                        synthesized_circuit = CXCircuit.from_parity_matrix(matrix,
                                                                           self.topology,
                                                                           parities_as_columns=wise,
                                                                           reallocate=True,
                                                                           method=method)
                        synth_matrix = synthesized_circuit.parity_matrix(wise)
                        permutation = synthesized_circuit._output_mapping
                        if wise:
                            self.assertNdArrEqual(matrix, synth_matrix[:, permutation])
                        else:
                            self.assertNdArrEqual(matrix, synth_matrix[permutation,:])
                        
    def test_reallocation_alt(self):
        for i in range(self.n_tests):
            for column_wise in [True, False]:
                for method in synthesis_methods: # TODO when more synthesis methods are available, not all are up-to-permutation.
                    with self.subTest(i=i, method=method, parities_as_columns=column_wise):
                        baseline = self.circuit[i].to_qiskit("naive")
                        permuted = self.circuit[i].to_qiskit(method, True, parities_as_columns=column_wise)
                        unpermute(permuted)
                        self.assertQCEqual(baseline, permuted)

    def assertNdArrEqual(self, a1, a2):
        self.assertListEqual(a1.tolist(), a2.tolist())

    def assertQCEqual(self, circuit1, circuit2):
        qc1 = Operator(circuit1)
        qc2 = Operator(circuit2)
        self.assertTrue(qc1.equiv(qc2))