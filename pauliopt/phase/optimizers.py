"""
    This module contains code to optimize circuits of mixed ZX phase gadgets
    using topologically-aware circuits of CNOTs.
"""

from collections import deque
from typing import (cast, Deque, Dict, Final, FrozenSet, Literal, Optional, overload, Protocol,
                    runtime_checkable, Sequence, Set, Tuple, Type, TypedDict)
import numpy as np # type: ignore
from pauliopt.phase.circuits import (PhaseGadget, PhaseCircuit, PhaseCircuitView,
                                     CXCircuitLayer, CXCircuit, CXCircuitView)
from pauliopt.topologies import Topology
from pauliopt.utils import Number, TempSchedule, AngleT

@runtime_checkable
class CostFun(Protocol):
    """
        Protocol for a cost function.
        The cost is a `float` or `int` computed from the phase block (readonly view)
        and CX block (readonly view) exposed by an instance of `PhaseCircuitCXBlockOptimizer`.
    """

    def __call__(self, phase_block: PhaseCircuitView, cx_block: CXCircuitView) -> Number:
        ...

def mst_impl_cost_fun(topology: Topology, circuit_rep: int = 1) -> "CostFun":
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
    cache: Dict[int, Number] = {}
    def cost_fun(phase_block: PhaseCircuitView, cx_block: CXCircuitView):
        phase_block_cost: Number = 0
        for gadget in phase_block.gadgets:
            h = hash(gadget)
            if h in cache:
                gadget_cost = cache[h]
            else:
                gadget_cost = gadget.mst_impl_cx_count(topology)
                cache[h] = gadget_cost
            phase_block_cost += gadget_cost
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


@runtime_checkable
class PhaseCircuitOptimizer(Protocol[AngleT]):
    """
        Optimizer for phase circuits based on simulated annealing.

        To understand how this works, consider the following code snippet:

        ```py
            optimizer = PhaseCircuitCXBlockOptimizer(original_circuit, topology, num_layers)
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

    @property
    def topology(self) -> Topology:
        """
            Readonly property exposing the topology constraining the circuit optimization.
        """
        ...

    @property
    def qubits(self) -> FrozenSet[int]:
        """
            Readonly property exposing the qubits spanned by the circuit to be optimized.
        """
        ...

    @property
    def original_gadgets(self) -> Tuple[PhaseGadget, ...]:
        """
            Readonly property exposing the gadgets in the original circuit to be optimized.
        """
        ...

    @property
    def phase_block(self) -> PhaseCircuitView[AngleT]:
        """
            Readonly property returning a readonly view on the optimized circuit.
        """
        ...

    @property
    def cx_block(self) -> CXCircuitView:
        """
            Readonly property returning a readonly view on the CX circuit used for optimization.
        """
        ...

    def anneal(self,
               num_iters: int,
               temp_schedule: TempSchedule,
               cost_fun: CostFun, *,
               loggers: AnnealingLoggers = {}):
               # pylint: disable = dangerous-default-value, no-self-use
        """
            Performs a cycle of simulated annealing optimization,
            using the given number of iterations, temperature schedule
            and cost function.
        """
        ...


class CXFlipOptimizer(PhaseCircuitOptimizer[AngleT]):
    # pylint: disable = too-many-instance-attributes
    """
        Optimizer for phase circuits based on simulated annealing,
        with CX block obtained by randomly flipping CX gates.
        The original phase circuit is passed to the constructor, together
        with a qubit topology and a fixed number of layers constraining the
        CX circuits to be used for simplification.

        See `PhaseCircuitOptimizer` for more details about optimizers.
    """

    _topology: Topology
    _qubits: FrozenSet[int]
    _original_gadgets: Tuple[PhaseGadget, ...]
    _phase_block: PhaseCircuit[AngleT]
    _phase_block_view: PhaseCircuitView
    _cx_block: CXCircuit
    _cx_block_view: CXCircuitView
    _rng_seed: Optional[int]
    _rng: np.random.Generator

    def __init__(self, original_circuit: PhaseCircuit[AngleT], topology: Topology, num_layers: int,
                 *, rng_seed: Optional[int] = None):
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
        while True:
            layer_idx = int(self._rng.integers(len(self._cx_block)))
            flippable_gates = list(self._cx_block[layer_idx]._iter_flippable_cxs()) # pylint: disable = protected-access
            gate_idx = self._rng.integers(len(flippable_gates))
            ctrl, trgt = flippable_gates[gate_idx]
            if layer_idx < len(self._cx_block)-1 and self._cx_block[layer_idx+1].has_cx(ctrl, trgt):
                # Try again if CX gate already present in layer above (to avoid redundancy)
                continue
            if layer_idx > 0 and self._cx_block[layer_idx-1].has_cx(ctrl, trgt):
                # Try again if CX gate already present in layer below (to avoid redundancy)
                continue
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
