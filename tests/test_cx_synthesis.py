import itertools
import unittest

import networkx as nx
import numpy as np

from pauliopt.topologies import Topology
from pauliopt.phase.cx_circuits import CXCircuit, CXCircuitLayer, synthesis_methods

SEED = 42

class TestCXSynthesis(unittest.TestCase):

    def setUp(self):
        self.n_tests = 20
        self.topology = Topology.grid(3,3)
        edges = [c.as_pair for c in self.topology.couplings]
        edges += [(c,p) for c,p in edges]
        depth = 20
        np.random.seed(SEED)
        self.circuit = [CXCircuit(self.topology, 
                        [CXCircuitLayer(self.topology, [edges[i]]) for i in np.random.choice(len(edges), depth, replace=True)]) 
                        for _ in range(self.n_tests)]
        self.col_matrix = [c.parity_matrix(parities_as_columns=True) for c in self.circuit]
        self.row_matrix = [c.parity_matrix(parities_as_columns=False) for c in self.circuit]

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
                        synthesized_circuit = CXCircuit.from_parity_matrix(matrix, self.topology, parities_as_columns=wise, reallocate=False, method=method)
                        synth_matrix = synthesized_circuit.parity_matrix(wise)
                        self.assertNdArrEqual(synth_matrix, matrix)

    def test_reallocation(self):
        for i in range(self.n_tests):
            for wise in [True, False]:
                if wise:
                    matrix = self.col_matrix[i]
                else:
                    matrix = self.row_matrix[i]
                for method in synthesis_methods: # TODO when more synthesis methods are available, not all are up-to-permutation.
                    with self.subTest(i=i, parity_as_columns=wise, method=method):
                        synthesized_circuit = CXCircuit.from_parity_matrix(matrix, self.topology, parities_as_columns=wise, reallocate=True, method=method)
                        synth_matrix = synthesized_circuit.parity_matrix(wise)
                        permutation = synthesized_circuit._output_mapping
                        perm_matrix = matrix[:, permutation]
                        self.assertNdArrEqual(perm_matrix, synth_matrix)

    def assertNdArrEqual(self, a1, a2):
        self.assertListEqual(a1.tolist(), a2.tolist())
