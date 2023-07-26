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
from pauliopt.phase.optimized_circuits import iter_anneal, reverse_traversal, reverse_traversal_anneal

SEED = 42

def unpermute(circuit:QuantumCircuit):
    changed_circuit = False
    if circuit.metadata:
        if "initial_layout" in circuit.metadata.keys():
            perm = circuit.metadata["initial_layout"]
            circuit.compose(Permutation(circuit.num_qubits, perm), front=True, inplace=True)
            changed_circuit = True
        if "final_layout" in circuit.metadata.keys():
            circuit.compose(Permutation(circuit.num_qubits, circuit.metadata["final_layout"]), front=False, inplace=True)
            changed_circuit = True
        if not changed_circuit:
            Warning("Attempt to unpermute did not change the circuit, no initial or final layout found.")

class TestPhaseCircuitSynthesis(unittest.TestCase):

    def setUp(self):
        self.n_tests = 10
        n_gadgets = 10
        self.num_rt_iters = 10
        self.num_anneal_iters = 100
        self.cx_blocks = 5
        self.anneal_kwargs = {}
        self.opt_kwargs_list = []
        for method in synthesis_methods:
            for cx_synth in ["naive"] + cx_synthesis_methods:
                self.opt_kwargs_list.append({
                    "rng_seed": SEED,
                    "phase_method": method, 
                    "cx_method": cx_synth, 
                    "reallocate": True
                })
        self.topology = Topology.grid(2,2)
        self.circuits = [PhaseCircuit.random(self.topology.num_qubits, n_gadgets, angle_subdivision=24, rng_seed=SEED) for _ in range(self.n_tests)]
        self.qiskit_circuits = [self.default_to_qiskit(c) for c in self.circuits]

    def default_to_qiskit(self, circuit):
        return circuit.to_qiskit(self.topology, simplified=False, method="naive")
    
    def test_fixed_annealer(self):
        self.run_general_test("naive fixed anneal", {"naive naive": lambda c: iter_anneal(c, self.topology, self.cx_blocks, 1, self.num_anneal_iters, {"phase_method": "naive", "cx_method": "naive", "reallocate": False}, self.anneal_kwargs)})

    def test_reallocated_annealer(self):
        opts = {" ".join([opt_kwargs["phase_method"], opt_kwargs["cx_method"]]): lambda c: iter_anneal(c, self.topology, self.cx_blocks, 1, self.num_anneal_iters, opt_kwargs, self.anneal_kwargs) for opt_kwargs in self.opt_kwargs_list}
        self.run_general_test("anneal", opts)

    def test_iter_anneal(self):
        opts = {" ".join([opt_kwargs["phase_method"], opt_kwargs["cx_method"]]): lambda c: iter_anneal(c, self.topology, self.cx_blocks, self.num_rt_iters, self.num_anneal_iters, opt_kwargs, self.anneal_kwargs) for opt_kwargs in self.opt_kwargs_list}
        self.run_general_test("iter anneal", opts)

    def test_reverse_traversal(self):
        opts = {" ".join([opt_kwargs["phase_method"], opt_kwargs["cx_method"]]) : lambda c: reverse_traversal(c, self.topology, self.cx_blocks, self.num_rt_iters, 0, opt_kwargs, self.anneal_kwargs) for opt_kwargs in self.opt_kwargs_list}
        self.run_general_test("reverse traversal", opts)

    def test_reverse_traversal_then_anneal(self):
        opts = {" ".join([opt_kwargs["phase_method"], opt_kwargs["cx_method"]]) : lambda c: reverse_traversal(c, self.topology, self.cx_blocks, self.num_rt_iters, self.num_anneal_iters, opt_kwargs, self.anneal_kwargs) for opt_kwargs in self.opt_kwargs_list}
        self.run_general_test("reverse traversal->anneal", opts)

    def test_reverse_traversal_and_anneal(self):
        opts = {" ".join([opt_kwargs["phase_method"], opt_kwargs["cx_method"]]) : lambda c: reverse_traversal_anneal(c, self.topology, self.cx_blocks, self.num_rt_iters, self.num_anneal_iters, opt_kwargs, self.anneal_kwargs) for opt_kwargs in self.opt_kwargs_list}
        self.run_general_test("reverse traversal + anneal", opts)

    def run_general_test(self, name:str, optimizers:dict):
        for i in range(self.n_tests):
            for synth_name, optimizer in optimizers.items():
                with self.subTest(i=i, optimizer=name, synth_name=synth_name):
                    opt = optimizer(self.circuits[i])
                    synthesized = opt.to_qiskit(False)
                    unpermute(synthesized)

                    if not Operator(synthesized).equiv(Operator(self.qiskit_circuits[i])):
                        print(name)
                        print(synthesized)
                        print(opt._cx_block.to_qiskit())
                        print(self.qiskit_circuits[i])
                        print("-------")
                    self.assertQCEqual(synthesized, self.qiskit_circuits[i])
    
    def assertQCEqual(self, circuit1, circuit2):
        qc1 = Operator(circuit1)
        qc2 = Operator(circuit2)
        passed = qc1.equiv(qc2)
        self.assertTrue(passed)
