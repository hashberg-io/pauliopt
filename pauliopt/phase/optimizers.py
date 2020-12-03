"""
    This module contains code to optimize circuits of mixed ZX phase gadgets
    using topologically-aware circuits of CNOTs.
"""

from collections import deque
from typing import (Deque, Dict, Generic, Optional, Protocol, runtime_checkable,
                    Set, Tuple, TypedDict, Union)
import numpy as np # type: ignore
from pauliopt.phase.circuits import (PhaseGadget, PhaseCircuit, PhaseCircuitView,
                                     CXCircuitLayer, CXCircuit, CXCircuitView)
from pauliopt.topologies import Topology
from pauliopt.utils import AngleT, TempSchedule, StandardTempSchedule, StandardTempSchedules

@runtime_checkable
class AnnealingCostLogger(Protocol):
    """
        Protocol for logger of initial/final cost in annealing.
    """

    def __call__(self, cx_count: int, num_iters: int):
        ...


@runtime_checkable
class AnnealingIterLogger(Protocol):
    """
        Protocol for logging of iteration info in annealing.
    """

    def __call__(self, it: int, prev_cx_count: int, new_cx_count: int,
                 accepted: bool, flip: Tuple[int, Tuple[int, int]],
                 t: float, num_iters: int):
        # pylint: disable = too-many-arguments
        ...


class AnnealingLoggers(TypedDict, total=False):
    """
        Typed dictionary of loggers for annealing.
    """

    log_start: AnnealingCostLogger
    log_iter: AnnealingIterLogger
    log_end: AnnealingCostLogger


def _validate_temp_schedule(schedule: Union[StandardTempSchedule, TempSchedule]) -> TempSchedule:
    if not isinstance(schedule, TempSchedule):
        if not isinstance(schedule, tuple) or len(schedule) != 3:
            raise TypeError(f"Expected triple (schedule_name, t_init, t_final), "
                            f"found {schedule}")
        schedule_name, t_init, t_final = schedule
        if schedule_name not in StandardTempSchedules:
            raise TypeError(f"Invalid standard temperature schedule name {schedule_name}, "
                            f"allowed names are: {list(StandardTempSchedules.keys())}")
        if not isinstance(t_init, (int, float)) or not isinstance(t_final, (int, float)):
            raise TypeError("Expected t_init and t_final to be int or float.")
        schedule = StandardTempSchedules[schedule_name](t_init, t_final)
    return schedule


def _validate_loggers(loggers: AnnealingLoggers) -> Tuple[Optional[AnnealingCostLogger],
                                                          Optional[AnnealingIterLogger],
                                                          Optional[AnnealingCostLogger]]:
    log_start = loggers.get("log_start", None)
    log_iter = loggers.get("log_iter", None)
    log_end = loggers.get("log_end", None)
    if log_start is not None and not isinstance(log_start, AnnealingCostLogger):
        raise TypeError(f"Expected AnnealingCostLogger, found {type(log_start)}")
    if log_iter is not None and not isinstance(log_iter, AnnealingIterLogger):
        raise TypeError(f"Expected AnnealingCostLogger, found {type(log_iter)}")
    if log_end is not None and not isinstance(log_end, AnnealingCostLogger):
        raise TypeError(f"Expected AnnealingCostLogger, found {type(log_end)}")
    return log_start, log_iter, log_end


class PhaseCircuitOptimizer(Generic[AngleT]):
    # pylint: disable = too-many-instance-attributes
    """
        Optimizer for phase circuits based on simulated annealing.
        The original phase circuit is passed to the constructor, together
        with a qubit topology and a fixed number of layers constraining the
        CX circuits to be used for simplification.

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

    _topology: Topology
    _num_qubits: int
    _original_gadgets: Tuple[PhaseGadget, ...]
    _circuit_rep: int
    _init_cx_count: int
    _gadget_cx_count_cache: Dict[int, Dict[Tuple[int, ...], int]]
    _phase_block: PhaseCircuit[AngleT]
    _phase_block_view: PhaseCircuitView
    _cx_block: CXCircuit
    _cx_block_view: CXCircuitView
    _cx_count: int
    _rng_seed: Optional[int]
    _rng: np.random.Generator

    def __init__(self, original_circuit: PhaseCircuit[AngleT], topology: Topology, num_layers: int,
                 *, circuit_rep: int = 1, rng_seed: Optional[int] = None):
        if not isinstance(original_circuit, PhaseCircuit):
            raise TypeError(f"Expected PhaseCircuit, found {type(original_circuit)}.")
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise TypeError(f"Expected positive integer, found {num_layers}.")
        if not isinstance(circuit_rep, int) or circuit_rep <= 0:
            raise TypeError(f"Expected positive integer, found {circuit_rep}.")
        if rng_seed is not None and not isinstance(rng_seed, int):
            raise TypeError("RNG seed must be integer or None.")
        self._topology = topology
        self._num_qubits = original_circuit.num_qubits
        self._original_gadgets = tuple(original_circuit.gadgets)
        self._circuit_rep = circuit_rep
        self._phase_block = PhaseCircuit(self._num_qubits, self._original_gadgets)
        self._cx_block = CXCircuit(topology,
                                   [CXCircuitLayer(topology) for _ in range(num_layers)])
        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(seed=rng_seed)
        self._phase_block_view = PhaseCircuitView(self._phase_block)
        self._cx_block_view = CXCircuitView(self._cx_block)
        self._gadget_cx_count_cache = {}
        self._init_cx_count = self._compute_cx_count()
        self._cx_count = self._init_cx_count

    @property
    def topology(self) -> Topology:
        """
            Readonly property exposing the topology constraining the circuit optimization.
        """
        return self._topology

    @property
    def num_qubits(self) -> int:
        """
            Readonly property exposing the number of qubits spanned by the circuit to be optimized.
        """
        return self._num_qubits

    @property
    def original_gadgets(self) -> Tuple[PhaseGadget, ...]:
        """
            Readonly property exposing the gadgets in the original circuit to be optimized.
        """
        return self._original_gadgets

    @property
    def circuit_rep(self) -> int:
        """
            Readonly property exposing the number of times that the original circuit is
            to be repeated, for use when computing CX counts.
        """
        return self._circuit_rep

    @property
    def init_cx_count(self) -> int:
        """
            Readonly property exposing the CX count for the original circuit.
        """
        return self._init_cx_count

    @property
    def phase_block(self) -> PhaseCircuitView:
        """
            Readonly property exposing a readonly view on the phase block of the optimized circuit.
        """
        return self._phase_block_view

    @property
    def cx_block(self) -> CXCircuitView:
        """
            Readonly property exposing a readonly view on the CX block of the optimized circuit.
        """
        return self._cx_block_view

    @property
    def cx_count(self) -> int:
        """
            Readonly property exposing the current CX count for the optimized circuit.
        """
        return self._cx_count

    def as_qiskit_circuit(self):
        """
            Returns the optimized circuit as a Qiskit circuit.

            This method relies on the `qiskit` library being available.
            Specifically, the `circuit` argument must be of type
            `qiskit.providers.BaseBackend`.
        """
        try:
            # pylint: disable = import-outside-toplevel
            from qiskit.circuit import QuantumCircuit # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'qiskit' library.")
        circuit = QuantumCircuit(self.num_qubits)
        for layer in reversed(self._cx_block):
            for ctrl, trgt in layer.gates:
                circuit.cx(ctrl, trgt)
        for __ in range(self._circuit_rep):
            for gadget in self._phase_block.gadgets:
                gadget.on_qiskit_circuit(self._topology, circuit)
        for layer in self._cx_block:
            for ctrl, trgt in layer.gates:
                circuit.cx(ctrl, trgt)
        return circuit

    def anneal(self,
               num_iters: int, *,
               schedule: Union[StandardTempSchedule, TempSchedule] = ("linear", 1.0, 0.1),
               loggers: AnnealingLoggers = {}):
               # pylint: disable = dangerous-default-value
        # pylint: disable = too-many-locals
        """
            Performs a cycle of simulated annealing optimization,
            using the given number of iterations, temperature schedule,
            initial/final temperatures.
        """
        # Validate arguments:
        if not isinstance(num_iters, int) or num_iters <= 0:
            raise TypeError(f"Expected a positive integer, found {num_iters}.")
        schedule = _validate_temp_schedule(schedule)
        log_start, log_iter, log_end = _validate_loggers(loggers)
        # Log start:
        if log_start is not None:
            log_start(self._cx_count, num_iters)
        # Pre-sample random numbers to use in iterations:
        rand = self._rng.uniform(size=num_iters)
        # Run iterations:
        for it in range(num_iters):
            t = schedule(it, num_iters=num_iters)
            layer_idx, (ctrl, trgt) = self.random_flip_cx()
            new_cx_count = self._compute_cx_count()
            cx_count_diff = new_cx_count-self._cx_count
            accept_step = cx_count_diff < 0 or rand[it] < np.exp(-cx_count_diff/t)
            if log_iter is not None:
                log_iter(it, self._cx_count, new_cx_count, accept_step,
                         (layer_idx, (ctrl, trgt)), t, num_iters)
            if accept_step:
                # Accept changes:
                self._cx_count = new_cx_count
            else:
                # Undo changes:
                self._flip_cx(layer_idx, ctrl, trgt)
        # Log end:
        if log_end is not None:
            log_end(self._cx_count, num_iters)

    def random_flip_cx(self) -> Tuple[int, Tuple[int, int]]:
        """
            Randomly flips a CX gate in the CX circuit used for the optimization,
            updating both the CX circuit and the circuit being optimized.

            Returns the layer index and gate (pair of control and target) that were
            flipped (e.g. in case the flip needs to be subsequently undone).
        """
        while True:
            layer_idx = int(self._rng.integers(len(self._cx_block)))
            ctrl, trgt = self._cx_block[layer_idx].random_flip_cx(self._rng)
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

    def _compute_cx_count(self) -> int:
        # pylint: disable = protected-access
        phase_block_cost = self._phase_block._cx_count(self._topology,
                                                       self._gadget_cx_count_cache)
        return self._circuit_rep*phase_block_cost + 2*self._cx_block.num_gates
