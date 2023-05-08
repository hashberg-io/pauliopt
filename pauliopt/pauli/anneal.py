import networkx as nx
import numpy as np
from qiskit import QuantumCircuit

from pauliopt.pauli.clifford_gates import CX, CY, CZ, CliffordGate
from pauliopt.pauli.clifford_region import CliffordRegion
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.phase.optimized_circuits import _validate_temp_schedule
from pauliopt.topologies import Topology


def pick_random_gate(num_qubits, G: nx.Graph, gate_set=None):
    if gate_set is None:
        gate_set = [CX, CY, CZ]

    gate = np.random.choice(gate_set)

    return gate.generate_random(num_qubits)


def compute_effect(pp: PauliPolynomial, gate: CliffordGate, topology: Topology,
                   leg_cache=None):
    pp_ = pp.copy()
    pp_.propagate(gate)

    return pp_.two_qubit_count(topology, leg_cache=leg_cache) - pp.two_qubit_count(
        topology, leg_cache=leg_cache)


def anneal(pp: PauliPolynomial, topology, schedule=("geometric", 1.0, 0.1),
           nr_iterations=100) -> QuantumCircuit:
    leg_cache = {}
    clifford_region = CliffordRegion()

    schedule = _validate_temp_schedule(schedule)
    random_nrs = np.random.uniform(0.0, 1.0, size=(nr_iterations,))
    num_qubits = pp.num_qubits
    for it in range(nr_iterations):
        t = schedule(it, nr_iterations)
        gate = pick_random_gate(num_qubits, topology.to_nx)
        effect = 2 + compute_effect(pp, gate, topology, leg_cache=leg_cache)
        accept_step = effect < 0 or random_nrs[it] < np.exp(-np.log(2) * effect / t)
        if accept_step:
            clifford_region.add_gate(gate)  # TODO optimize clifford regions
            pp.propagate(gate)

    qc = QuantumCircuit(pp.num_qubits)
    qc.compose(clifford_region.to_qiskit(), inplace=True)  # TODO route on architecture
    qc.compose(pp.to_qiskit(topology), inplace=True)
    qc.compose(clifford_region.to_qiskit().inverse(), inplace=True)
    return qc
