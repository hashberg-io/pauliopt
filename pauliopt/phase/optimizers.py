"""
    This module contains code to optimize circuits of mixed ZX phase gadgets
    using topologically-aware circuits of CNOTs.
"""

from collections import deque
from math import log
from typing import (Collection, Deque, Dict, FrozenSet, Optional, Protocol,
                    runtime_checkable, Sequence, Set, Tuple, TypedDict, Union)
import numpy as np # type: ignore
from pauliopt.phase.circuits import (PhaseGadget, PhaseCircuit, PhaseCircuitView,
                                     CXCircuitLayer, CXCircuit, CXCircuitView)
from pauliopt.topologies import Coupling, Topology
from pauliopt.utils import Number, TempSchedule

@runtime_checkable
class CostFun(Protocol):
    """
        Protocol for a cost function.
        The cost is a `float` or `int` computed from the phase block (readonly view)
        and CX block (readonly view) exposed by an instance of `PhaseCircuitOptimizer`.
    """

    def __call__(self, phase_block: PhaseCircuitView, cx_block: CXCircuitView) -> Number:
        ...

class _WeightFun(Protocol):
    def __call__(self, fro: int, to: int) -> Number:
        ...

def _prim_algorithm_debug(nodes: Collection[int], weight: _WeightFun,
                          inf: Number) -> Tuple[Sequence[Number], Sequence[Tuple[int, int]]]:
    """
        Computes the weight of the minimum spanning tree connecting
        the given nodes, using the given weight function for branches.
        The number `inf` should be larger than the maximum weight
        that can be encountered in the process.

        Returns the list of weights for the individual branches in such
        a minimum spanning tree the list of branches defining the tree itself.
    """
    if not nodes:
        return [], []
    mst_lengths = []
    mst = []
    # Initialise set of nodes to visit:
    to_visit = set(nodes)
    n0 = next(iter(to_visit))
    to_visit.remove(n0)
    # Initialise dict of distances from visited set:
    dist_from_visited = {
        n: weight(n0, n) for n in nodes
    }
    edge_from_visited: Dict[int, Tuple[int, int]] = {
        n: (n0, n) for n in nodes
    }
    while to_visit:
        # Look for the node to be visited which is nearest to the visited set:
        nearest_node = 0 # dummy value
        nearest_dist = inf # dummy value
        for n in to_visit:
            n_dist = dist_from_visited[n]
            if n_dist < nearest_dist:
                nearest_node = n
                nearest_dist = n_dist
        # Nearest node is removed and added to the MST:
        to_visit.remove(nearest_node)
        mst_lengths.append(nearest_dist)
        mst.append(edge_from_visited[nearest_node])
        # Update shortest distances to visited set:
        for n in to_visit:
            dist_nearest_n = weight(nearest_node, n)
            if dist_nearest_n < dist_from_visited[n]:
                dist_from_visited[n] = dist_nearest_n
                edge_from_visited[n] = (nearest_node, n)
    return mst_lengths, mst

def _prim_algorithm(nodes: Collection[int], weight: _WeightFun,
                    inf: Number) -> Number:
    """
        Computes the weight of the minimum spanning tree connecting
        the given nodes, using the given weight function for branches.
        The number `inf` should be larger than the maximum weight
        that can be encountered in the process.
    """
    if not nodes:
        return 0
    mst_length: Number = 0
    # Initialise set of nodes to visit:
    to_visit = set(nodes)
    n0 = next(iter(to_visit))
    to_visit.remove(n0)
    # Initialise dict of distances from visited set:
    dist_from_visited = {
        n: weight(n0, n) for n in nodes
    }
    while to_visit:
        # Look for the node to be visited which is nearest to the visited set:
        nearest_node = 0 # dummy value
        nearest_dist = inf # dummy value
        for n in to_visit:
            n_dist = dist_from_visited[n]
            if n_dist < nearest_dist:
                nearest_node = n
                nearest_dist = n_dist
        # Nearest node is removed and added to the MST:
        to_visit.remove(nearest_node)
        mst_length += nearest_dist
        # Update shortest distances to visited set:
        for n in to_visit:
            dist_nearest_n = weight(nearest_node, n)
            if dist_nearest_n < dist_from_visited[n]:
                dist_from_visited[n] = dist_nearest_n
    return mst_length

def cx_count(gadget: PhaseGadget,
             topology: Topology) -> Tuple[Sequence[Number], Sequence[Tuple[int, int]]]:
    """
        Returns detailed CX count information for the minimal spanning tree
        implementation of the given gadget on the given topology.
        Returns the list of CX counts for the individual branches of an MST and
        the MST itself (as a list of branches).
    """
    if not isinstance(gadget, PhaseGadget):
        raise TypeError(f"Expected PhaseGadget, found {type(gadget)}.")
    if not isinstance(topology, Topology):
        raise TypeError(f"Expected Topology, found {type(topology)}.")
    return _prim_algorithm_debug(gadget.qubits,
                                 lambda u, v: 4*topology.dist(u, v)-2,
                                 4*len(topology.qubits)-2)


def cx_count_cost_fun(topology: Topology, circuit_rep: int = 1) -> "CostFun":
    """
        Returns a topology-aware cost function for the optimizer, based on CX count.
        It takes an optional `circuit_rep` argument that can be used to specify the
        number of repetitions for the original circuit (e.g. for QML ansatz applications).

        The CX count for an individual phase circuit block in the optimized circuit is
        computed by finding a minimum spanning tree implementation of the phase gadgets.
        The CX count for an individual block is then multiplied by `circuit_rep` and twice
        the number of CX gates in the CX block is added (because each gate appears twice,
        once in the first block and once in the last block of the optimized circuit.).
        The resulting CX count is returned as the cost.
    """
    if not isinstance(topology, Topology):
        raise TypeError(f"Expected Topology, found {type(topology)}.")
    if not isinstance(circuit_rep, int) or circuit_rep <= 0:
        raise TypeError(f"Expected positive integer, found {circuit_rep}.")
    def cost_fun(phase_block: PhaseCircuitView, cx_block: CXCircuitView):
        phase_block_cost: Number = 0
        for gadget in phase_block.gadgets:
            phase_block_cost += _prim_algorithm(gadget.qubits,
                                                lambda u, v: 4*topology.dist(u, v)-2,
                                                4*len(topology.qubits)-2)
        return circuit_rep*phase_block_cost + 2*cx_block.num_gates
    return cost_fun


@runtime_checkable
class AnnealingCostLogger(Protocol):
    """
        Protocol for logger of initial/final cost in annealing.
    """

    def __call__(self, cost: Number, num_iters: int):
        ...


@runtime_checkable
class AnnealingIterLogger(Protocol):
    """
        Protocol for logging of iteration info in annealing.
    """

    def __call__(self, it: int, prev_cost: Number, new_cost: Number,
                 accepted: bool, flip: Tuple[int, Tuple[int, int]],
                 t: Number, num_iters: int):
        # pylint: disable = too-many-arguments
        ...


class AnnealingLoggers(TypedDict, total=False):
    """
        Typed dictionary of loggers for annealing.
    """

    log_init_cost: AnnealingCostLogger
    log_iter: AnnealingIterLogger
    log_final_cost: AnnealingCostLogger


class PhaseCircuitOptimizer:
    # pylint: disable = too-many-instance-attributes
    """
        Optimizer for phase circuits based on simulated annealing.
        The original phase circuit is passed to the constructor, together
        with a qubit topology and a fixed number of layers constraining the
        CX circuits to be used for simplification.

        To understand how this works, consider the following code snipped:

        ```py
            optimizer = PhaseCircuitOptimizer(original_circuit, topology, num_layers)
            optimizer.anneal(num_iters, temp_schedule, cost_fun)
            phase_block = optimizer.phase_block
            cx_block = optimizer.cx_block
        ```

        The optimized circuit is obtained by composing three blocks:

        1. a first block of CX gates, given by `cx_block.dag`
           (the same CX gates of `cx_block`, but in reverse order);
        2. a central block of phase gadgets, given by `phase_block`;
        3. a final block of CX gates, given by `cx_block`.

        Furthermore, if the original circuit is repeated `n` times, e.g. as part
        of a quantum machine learning ansatz, then the corresponding optimized
        circuit is obtained by repeating the central `phase_block` alone `n` times,
        keeping the first and last CX blocks unaltered (because the intermediate
        CX blocks cancel each other out when repeating the optimized circuit `n` times).

    """

    _topology: Topology
    _qubits: FrozenSet[int]
    _original_gadgets: Tuple[PhaseGadget, ...]
    _phase_block: PhaseCircuit
    _phase_block_view: PhaseCircuitView
    _cx_block: CXCircuit
    _cx_block_view: CXCircuitView
    _rng_seed: Optional[int]
    _rng: np.random.Generator

    def __init__(self, original_circuit: PhaseCircuit, topology: Topology, num_layers: int, *,
                 rng_seed: Optional[int] = None):
        if not isinstance(original_circuit, PhaseCircuit):
            raise TypeError(f"Expected PhaseCircuit, found {type(original_circuit)}.")
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise TypeError(f"Expected positive integer, found {num_layers}.")
        if rng_seed is not None and not isinstance(rng_seed, int):
            raise TypeError("RNG seed must be integer or None.")
        self._topology = topology
        self._qubits = original_circuit.qubits
        self._original_gadgets = tuple(original_circuit.gadgets)
        self._phase_block = PhaseCircuit(self._qubits, self._original_gadgets)
        self._cx_block = CXCircuit(topology,
                                   [CXCircuitLayer(topology) for _ in range(num_layers)])
        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(seed=rng_seed)
        self._phase_block_view = PhaseCircuitView(self._phase_block)
        self._cx_block_view = CXCircuitView(self._cx_block)

    @property
    def topology(self) -> Topology:
        """
            Readonly property exposing the topology constraining the circuit optimization.
        """
        return self._topology

    @property
    def qubits(self) -> FrozenSet[int]:
        """
            Readonly property exposing the qubits spanned by the circuit to be optimized.
        """
        return self._qubits

    @property
    def original_gadgets(self) -> Tuple[PhaseGadget, ...]:
        """
            Readonly property exposing the gadgets in the original circuit to be optimized.
        """
        return self._original_gadgets

    @property
    def phase_block(self) -> PhaseCircuitView:
        """
            Readonly property returning a readonly view on the optimized circuit.
        """
        return self._phase_block_view

    @property
    def cx_block(self) -> CXCircuitView:
        """
            Readonly property returning a readonly view on the CX circuit used for optimization.
        """
        return self._cx_block_view

    def anneal(self,
               num_iters: int,
               temp_schedule: TempSchedule,
               cost_fun: CostFun, *,
               loggers: AnnealingLoggers = {}):
               # pylint: disable = dangerous-default-value
        # pylint: disable = too-many-locals
        """
            Performs a cycle of simulated annealing optimization,
            using the given number of iterations, temperature schedule
            and cost function.
        """
        # Validate arguments:
        if not isinstance(num_iters, int) or num_iters <= 0:
            raise TypeError(f"Expected a positive integer, found {num_iters}.")
        if not isinstance(temp_schedule, TempSchedule):
            raise TypeError(f"Expected TempSchedule, found {type(temp_schedule)}.")
        if not isinstance(cost_fun, CostFun):
            raise TypeError(f"Expected CostFun, found {type(cost_fun)}.")
        log_init_cost = loggers.get("log_init_cost", None)
        log_iter = loggers.get("log_iter", None)
        log_final_cost = loggers.get("log_final_cost", None)
        if log_init_cost is not None and not isinstance(log_init_cost, AnnealingCostLogger):
            raise TypeError(f"Expected AnnealingCostLogger, found {type(log_init_cost)}")
        if log_iter is not None and not isinstance(log_iter, AnnealingIterLogger):
            raise TypeError(f"Expected AnnealingCostLogger, found {type(log_iter)}")
        if log_final_cost is not None and not isinstance(log_final_cost, AnnealingCostLogger):
            raise TypeError(f"Expected AnnealingCostLogger, found {type(log_final_cost)}")
        # Compute and log initial cost:
        init_cost = cost_fun(self.phase_block, self.cx_block)
        curr_cost = init_cost
        if log_init_cost is not None:
            log_init_cost(init_cost, num_iters)
        # Pre-sample random numbers to use in iterations:
        rand = self._rng.uniform(size=num_iters)
        # Run iterations:
        for it in range(num_iters):
            t = temp_schedule(it, num_iters=num_iters)
            layer_idx, (ctrl, trgt) = self.random_flip_cx()
            new_cost = cost_fun(self.phase_block, self.cx_block)
            accept_step = (new_cost < curr_cost
                           or rand[it] < np.exp(-(new_cost-curr_cost)/t))
            if log_iter is not None:
                log_iter(it, curr_cost, new_cost, accept_step,
                         (layer_idx, (ctrl, trgt)), t, num_iters)
            if accept_step:
                # Accept changes:
                curr_cost = new_cost
            else:
                # Undo changes:
                self._flip_cx(layer_idx, ctrl, trgt)
        # Log final cost:
        if log_final_cost is not None:
            log_final_cost(curr_cost, num_iters)

    def random_flip_cx(self) -> Tuple[int, Tuple[int, int]]:
        """
            Randomly flips a CX gate in the CX circuit used for the optimization,
            updating both the CX circuit and the circuit being optimized.

            Returns the layer index and gate (pair of control and target) that were
            flipped (e.g. in case the flip needs to be subsequently undone).
        """
        layer_idx = int(self._rng.integers(len(self._cx_block)))
        flippable_gates = list(self._cx_block[layer_idx]._iter_flippable_cxs()) # pylint: disable = protected-access
        gate_idx = self._rng.integers(len(flippable_gates))
        ctrl, trgt = flippable_gates[gate_idx]
        self._flip_cx(layer_idx, ctrl, trgt)
        return layer_idx, (ctrl, trgt)

    def is_cx_flippable(self, layer_idx: int, ctrl: int, trgt: int) -> bool:
        """
            Checks whether the given CX gate can be flipped in the given layer.
        """
        if not isinstance(layer_idx, int) or not 0 <= layer_idx < len(self._cx_block):
            raise TypeError(f"Invalid layer index {layer_idx} for CX circuit.")
        layer = self._cx_block[layer_idx]
        return layer.is_cx_flippable(ctrl, trgt)

    def flip_cx(self, layer_idx: int, ctrl: int, trgt: int) -> None:
        """
            Performs the actions needed to flip the given CX gate in the given layer
            of the CX circuit used for the optimization:

            - undoes all gates in layers subsequent to the given layer which are
              causally following the given gate, starting from the last layer and
              working backwards towards the gate;
            - applies the desired gate;
            - redoes all gate undone, in reverse order (starting from the gate and
              working forwards towards the last layer).
        """
        if not self.is_cx_flippable(layer_idx, ctrl, trgt):
            raise ValueError(f"Gate {(ctrl, trgt)} cannot be flipped in layer number {layer_idx}.")
        self._flip_cx(layer_idx, ctrl, trgt)

    def _flip_cx(self, layer_idx: int, ctrl: int, trgt: int) -> None:
        conj_by: Deque[Tuple[int, int]] = deque([(ctrl, trgt)])
        qubits_spanned: Set[int] = set([ctrl, trgt])
        for layer in self._cx_block[layer_idx:]:
            new_qubits_spanned: Set[int] = set()
            for q in qubits_spanned:
                incident_gate = layer.incident(q)
                if incident_gate is not None:
                    new_qubits_spanned.update({incident_gate[0], incident_gate[1]})
                    conj_by.appendleft(incident_gate) # will first undo the gate ...
                    # ... then do all gates already in conj_by ...
                    conj_by.append(incident_gate) # ... then finally redo the gate
            qubits_spanned.update(new_qubits_spanned)
        # Flip the gate in the CX circuit:
        self._cx_block[layer_idx].flip_cx(ctrl, trgt)
        # Conjugate the optimized phase gadget circuit by all necessary gates:
        for cx in conj_by:
            self._phase_block.conj_by_cx(*cx)
