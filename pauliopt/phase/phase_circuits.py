"""
    This module contains code to create circuits of mixed ZX phase gadgets.
"""

from collections import deque
from itertools import islice
from math import ceil, log10
from typing import (Any, Callable, cast, Collection, Dict, FrozenSet, Iterator, List,
                    Literal, Mapping, Optional, overload, Sequence, Set, Tuple, Union)
import numpy as np
import numpy.typing as npt
from pauliopt.qasm import QASM
from pauliopt.topologies import Topology
from pauliopt.utils import Angle, AngleExpr, AngleVar, SVGBuilder, pi

def _frozenset_to_int(s: FrozenSet[int]) -> int:
    i = 0
    for x in s:
        i |= (2**x)
    return i

def _int_to_frozenset(i: int) -> FrozenSet[int]:
    s: List[int] = []
    x = 0
    while i != 0:
        if i % 2 == 1:
            s.append(x)
        i //= 2
        x += 1
    return frozenset(s)

def _int_to_iterator(i: int) -> Iterator[int]:
    x = 0
    while i != 0:
        if i % 2 == 1:
            yield x
        i //= 2
        x += 1

def _prims_algorithm_weight(nodes: Collection[int], weight: Callable[[int, int], int],
                            inf: int) -> int:
    """
        A modified version of Prim's algorithm that
        computes the weight of the minimum spanning tree connecting
        the given nodes, using the given weight function for branches.
        The number `inf` should be larger than the maximum weight
        that can be encountered in the process.
    """
    if not nodes:
        return 0
    mst_length: int = 0
    # Initialise set of nodes to visit:
    to_visit = set(nodes)
    n0 = next(iter(to_visit))
    to_visit.remove(n0)
    # Initialise dict of distances from visited set:
    dist_from_visited: Dict[int, int] = {
        n: weight(n0, n) for n in nodes
    }
    while to_visit:
        # Look for the node to be visited which is nearest to the visited set:
        nearest_node = 0 # dummy value
        nearest_dist: int = inf # dummy value
        for n in to_visit:
            n_dist: int = dist_from_visited[n]
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

def _prims_algorithm_branches(nodes: Collection[int], weight: Callable[[int, int], int],
                              inf: int) -> Sequence[Tuple[int, int]]:
    # pylint: disable = too-many-locals
    if not nodes:
        return []
    mst_branches = []
    # Initialise set of nodes to visit:
    to_visit = set(nodes)
    n0 = next(iter(to_visit))
    to_visit.remove(n0)
    # Initialise dict of distances from visited set:
    dist_from_visited: Dict[int, int] = {
        n: weight(n0, n) for n in nodes
    }
    # Initialise possible edges for the MST:
    edge_from_visited: Dict[int, Tuple[int, int]] = {
        n: (n0, n) for n in nodes
    }
    while to_visit:
        # Look for the node to be visited which is nearest to the visited set:
        nearest_node = 0 # dummy value
        nearest_dist: int = inf # dummy value
        for n in to_visit:
            n_dist: int = dist_from_visited[n]
            if n_dist < nearest_dist:
                nearest_node = n
                nearest_dist = n_dist
        # Nearest node is removed and added to the MST:
        to_visit.remove(nearest_node)
        mst_branches.append(edge_from_visited[nearest_node])
        # Update shortest distances/edges to visited set:
        for n in to_visit:
            dist_nearest_n = weight(nearest_node, n)
            if dist_nearest_n < dist_from_visited[n]:
                dist_from_visited[n] = dist_nearest_n
                edge_from_visited[n] = (nearest_node, n)
    return mst_branches

def _prims_algorithm_full(nodes: Collection[int], weight: Callable[[int, int], int],
                          inf: int) -> Tuple[int, Sequence[int], Sequence[Tuple[int, int]]]:
    # pylint: disable = too-many-locals
    if not nodes:
        return 0, [], []
    mst_length: int = 0
    mst_branch_lengths = []
    mst_branches = []
    # Initialise set of nodes to visit:
    to_visit = set(nodes)
    n0 = next(iter(to_visit))
    to_visit.remove(n0)
    # Initialise dict of distances from visited set:
    dist_from_visited: Dict[int, int] = {
        n: weight(n0, n) for n in nodes
    }
    # Initialise possible edges for the MST:
    edge_from_visited: Dict[int, Tuple[int, int]] = {
        n: (n0, n) for n in nodes
    }
    while to_visit:
        # Look for the node to be visited which is nearest to the visited set:
        nearest_node = 0 # dummy value
        nearest_dist: int = inf # dummy value
        for n in to_visit:
            n_dist: int = dist_from_visited[n]
            if n_dist < nearest_dist:
                nearest_node = n
                nearest_dist = n_dist
        # Nearest node is removed and added to the MST:
        to_visit.remove(nearest_node)
        mst_length += nearest_dist
        mst_branch_lengths.append(mst_length)
        mst_branches.append(edge_from_visited[nearest_node])
        # Update shortest distances/edges to visited set:
        for n in to_visit:
            dist_nearest_n = weight(nearest_node, n)
            if dist_nearest_n < dist_from_visited[n]:
                dist_from_visited[n] = dist_nearest_n
                edge_from_visited[n] = (nearest_node, n)
    return mst_length, mst_branch_lengths, mst_branches


class PhaseGadget:
    """
        Immutable container class for a phase gadget.
    """

    _qubits: FrozenSet[int]
    _basis: Literal["Z", "X"]
    _angle: AngleExpr

    def __init__(self, basis: Literal["Z", "X"], angle: AngleExpr, qubits: Collection[int]):
        if not isinstance(qubits, Collection) or not all(isinstance(q, int) for q in qubits):
            raise TypeError(f"Qubits should be a collection of integers, found {qubits}")
        if not qubits:
            raise ValueError("At least one qubit must be specified.")
        if basis not in ("Z", "X"):
            raise TypeError("Basis should be 'Z' or 'X'.")
        if not isinstance(angle, AngleExpr):
            raise TypeError(f"Angle should be an instance of `AngleExpr`, "
                            f"found {angle} of type {type(angle)} instead.")
        self._basis = basis
        self._angle = angle
        self._qubits = frozenset(qubits)

    @property
    def basis(self) -> Literal["Z", "X"]:
        """
            Readonly property exposing the basis for this phase gadget.
        """
        return self._basis

    @property
    def angle(self) -> AngleExpr:
        """
            Readonly property exposing the angle for this phase gadget.
        """
        return self._angle

    @property
    def qubits(self) -> FrozenSet[int]:
        """
            Readonly property exposing the qubits spanned by this phase gadget.
        """
        return self._qubits

    def cx_count(self, topology: Topology, *,
                 mapping: Optional[Union[Sequence[int], Dict[int, int]]] = None) -> int:
        """
            Returns the CX count for an implementation of this phase gadget
            on the given topology based on minimum spanning trees (MST).

            The optional `mapping` keyword argument can be used to specify a mapping of
            logical (circuit) qubits to phyisical (topology) qubits.
        """
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        if self.angle.is_zero:
            return 0
        # if self._angle.value % 2 == 0:
        #     return 0
        if mapping is not None and isinstance(mapping, Sequence):
            mapping = {
                i: mapping[i] for i in range(len(mapping))
            }
        if mapping is not None:
            # use the reverse mapping on the topology
            topology = topology.mapped_fwd({
                mapping[i]: i for i in mapping
            })
        return  _prims_algorithm_weight(self._qubits,
                                        lambda u, v: 4*topology.dist(u, v)-2,
                                        4*len(topology.qubits)-2)

    def on_qiskit_circuit(self, topology: Topology, circuit: Any) -> None:
        """
            Applies this phase gadget to a given qiskit quantum `circuit`,
            using the given `topology` to determine a minimum spanning
            tree implementation of the gadget.

            This method relies on the `qiskit` library being available.
            Specifically, the `circuit` argument must be of type
            `qiskit.providers.BaseBackend`.
        """
        # pylint: disable = too-many-branches, too-many-locals
        # TODO: currently uses CX ladder, must change into balanced tree! (same CX count)
        try:
            # pylint: disable = import-outside-toplevel
            from qiskit.circuit import QuantumCircuit # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("You must install the 'qiskit' library.") from e
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Argument 'circuit' must be of type "
                            "`qiskit.circuit.QuantumCircuit`.")
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        # Build MST data structure:
        mst_branches = _prims_algorithm_branches(self._qubits,
                                                 lambda u, v: 4*topology.dist(u, v)-2,
                                                 4*len(topology.qubits)-2)
        upper_ladder: List[Tuple[int, int]] = []
        if len(self._qubits) == 1:
            q0 = next(iter(self._qubits))
        else:
            q0 = min(*self._qubits)
        if mst_branches:
            incident: Dict[int, Set[Tuple[int, int]]] = {
                q: set() for q in self._qubits
            }
            for fst, snd in mst_branches:
                incident[fst].add((fst, snd))
                incident[snd].add((snd, fst))
            # Create ladder of CX gates:
            visited: Set[int] = set()
            queue = deque([q0])
            while queue:
                q = queue.popleft()
                visited.add(q)
                for tail, head in incident[q]:
                    if head not in visited:
                        if self.basis == "Z":
                            upper_ladder.append((head, tail))
                        else:
                            upper_ladder.append((tail, head))
                        queue.append(head)
        for ctrl, trgt in reversed(upper_ladder):
            circuit.cx(ctrl, trgt)
        if self.basis == "Z":
            circuit.rz(self.angle.to_qiskit, q0)
        else:
            circuit.rx(self.angle.to_qiskit, q0)
        for ctrl, trgt in upper_ladder:
            circuit.cx(ctrl, trgt)

    def print_impl_info(self, topology: Topology) -> None:
        """
            Prints information about an implementation of this phase gadget
            on the given topology based on minimum spanning trees (MST).
        """
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        mst_length, mst_branch_lengths, mst_branches = \
            _prims_algorithm_full(self._qubits,
                                  lambda u, v: 4*topology.dist(u, v)-2,
                                  4*len(topology.qubits)-2)
        print(f"MST implementation info for {str(self)}:")
        print(f"  - Overall CX count for gadget: {mst_length}")
        print(f"  - MST branches: {mst_branches}")
        print(f"  - CX counts for MST branches: {mst_branch_lengths}")
        print("")

    def __str__(self) -> str:
        return f"{self.basis}({self.angle}) @ {set(self.qubits)}"

    def __repr__(self) -> str:
        return f"PhaseGadget({repr(self.basis)}, {self.angle}, {set(self.qubits)})"

    def __hash__(self) -> int:
        return hash((self.basis, self.angle, self.qubits))

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if not isinstance(other, PhaseGadget):
            return NotImplemented
        return (self.basis == other.basis
                and self.angle == other.angle
                and self.qubits == other.qubits)


class Z:
    """
        Constructs a Z phase gadget with the idiomatic syntax:

        ```py
            Z(angle) @ qubits
        ```
    """

    _angle: AngleExpr

    def __init__(self, angle: AngleExpr):
        if not isinstance(angle, AngleExpr):
            raise TypeError(f"angle should be `AngleExpr`, "
                            f"found {angle} of type {type(angle)} instead.")
        self._angle = angle

    def __matmul__(self, qubits: Collection[int]) -> PhaseGadget:
        return PhaseGadget("Z", self._angle, qubits)


class X:
    """
        Constructs an X phase gadget with the idiomatic syntax:

        ```py
            X(angle) @ qubits
        ```
    """

    _angle: AngleExpr

    def __init__(self, angle: AngleExpr):
        if not isinstance(angle, AngleExpr):
            raise TypeError(f"angle should be `AngleExpr`, "
                            f"found {angle} of type {type(angle)} instead.")
        self._angle = angle

    def __matmul__(self, qubits: Collection[int]) -> PhaseGadget:
        return PhaseGadget("X", self._angle, qubits)


def _rx(qubit: int, angle: Angle) -> List[PhaseGadget]:
    return [X(angle) @ {qubit}]

def _rz(qubit: int, angle: Angle) -> List[PhaseGadget]:
    return [Z(angle) @ {qubit}]

def _ry(qubit: int, angle: Angle) -> List[PhaseGadget]:
    """ Phase gadget implementation of single-qubit Y rotation. """
    return _rx(qubit, +pi/2) + _rz(qubit, angle) + _rx(qubit, -pi/2)

def _i(qubit: int) -> List[PhaseGadget]:
    return []

def _x(qubit: int) -> List[PhaseGadget]:
    return _rx(qubit, pi)

def _z(qubit: int) -> List[PhaseGadget]:
    """ Phase gadget implementation of single-qubit X gate. """
    return _rz(qubit, pi)

def _y(qubit: int) -> List[PhaseGadget]:
    """ Phase gadget implementation of single-qubit Y gate. """
    return _z(qubit) + _x(qubit)

def _s(qubit: int) -> List[PhaseGadget]:
    """ Phase gadget implementation of single-qubit S gate. """
    return _rz(qubit, pi/2)

def _sdg(qubit: int) -> List[PhaseGadget]:
    """ Phase gadget implementation of single-qubit S gate. """
    return _rz(qubit, -pi/2)

def _v(qubit: int) -> List[PhaseGadget]:
    """ Phase gadget implementation of single-qubit S gate. """
    return _rx(qubit, pi/2)

def _vdg(qubit: int) -> List[PhaseGadget]:
    """ Phase gadget implementation of single-qubit S gate. """
    return _rx(qubit, -pi/2)

def _t(qubit: int) -> List[PhaseGadget]:
    """ Phase gadget implementation of single-qubit T gate. """
    return _rz(qubit, pi/4)

def _h(qubit: int,
       basis: Literal["Z", "X"] = "Z",
       sign: Literal[1, -1]=1) -> List[PhaseGadget]:
    """ Phase gadget implementation of single-qubit Hadamard gate. """
    if basis not in ("Z", "X"):
        raise TypeError(f"Invalid basis {basis}.")
    if sign not in (1, -1):
        raise TypeError(f"Invalid sign {sign}.")
    if basis == "Z":
        return _rx(qubit, sign*pi/2) + _rz(qubit, sign*pi/2) + _rx(qubit, sign*pi/2)
    return _rz(qubit, sign*pi/2) + _rx(qubit, sign*pi/2) + _rz(qubit, sign*pi/2)

def _cu1(ctrl: int, tgt: int, angle: Angle) -> List[PhaseGadget]:
    """ Phase gadget implementation of CU1 gate. """
    return [Z(-angle) @ {ctrl, tgt}] + _rz(ctrl, angle) + _rz(tgt, angle)

def _crz(ctrl: int, tgt: int, angle: Angle) -> List[PhaseGadget]:
    """ Phase gadget implementation of CRZ gate. """
    return [Z(-angle / 2) @ {ctrl, tgt}] + _rz(tgt, angle / 2)

def _cry(ctrl: int, tgt: int, angle: Angle) -> List[PhaseGadget]:
    """ Phase gadget implementation of CRY gate. """
    return _v(tgt) + _crz(ctrl, tgt, angle) + _vdg(tgt)

def _crx(ctrl: int, tgt: int, angle: Angle) -> List[PhaseGadget]:
    """ Phase gadget implementation of CRX gate. """
    return _h(tgt) + _crz(ctrl, tgt, angle) + _h(tgt, sign=-1)

def _cz(leg1: int, leg2: int) -> List[PhaseGadget]:
    """ Phase gadget implementation of CZ gate. """
    return _cu1(leg1, leg2, pi / 2)

def _cy(leg1: int, leg2: int) -> List[PhaseGadget]:
    """ Phase gadget implementation of CY gate. """
    return _v(leg2) + _cz(leg1, leg2) + _vdg(leg2)

def _cx(ctrl: int, tgt: int) -> List[PhaseGadget]:
    """ Phase gadget implementation of CX gate. """
    return _h(tgt) + _cz(ctrl, tgt) + _h(tgt, sign=-1)

def _u3(qubit: int, theta: Angle, phi: Angle, lam: Angle) -> List[PhaseGadget]:
    """ Phase gadget implementation of U3 gate. """
    return _rz(qubit, lam) + _ry(qubit, theta) + _rz(qubit, phi)


class PhaseCircuit(Sequence[PhaseGadget]):
    """
        Container class for a circuit of mixed ZX phase gadgets.
    """
    # pylint: disable = too-many-public-methods

    _matrix: Dict[Literal["Z", "X"], npt.NDArray[np.uint8]]
    """
        For `basis in ("Z", "X")`, the matrix `self._matrix[basis]`
        is the binary matrix encoding the qubits spanned by the
        `basis` gadgets.
    """

    _gadget_idxs: Dict[Literal["Z", "X"], List[int]]
    """
        For `basis in ("Z", "X")`, the list `self._gadget_idxs[basis]`
        maps each column index `c` for `self._matrix[basis]` to the
        index `self._gadget_idxs[c]` in the global list of gadgets for
        this circuit for the `basis` gadget corresponding to column `c`.
    """

    _gadget_legs_cache: Dict[Literal["Z", "X"], List[Optional[Tuple[int, ...]]]]
    """
        For `basis in ("Z", "X")`, the matrix `self._matrix[basis]`
        is the binary matrix encoding the qubits spanned by the
        `basis` gadgets.
    """

    _num_qubits: int
    """
        The number of qubits spanned by this circuit.
    """

    _angles: List[AngleExpr]
    """
        The global list of angles for the gadgets.
        The angle for the `basis` gadget corresponding to column index `c`
        of matrix `self._matrix[basis]` is given by:

        ```py
            self._angles[self._gadget_idxs[basis][c]]
        ```
    """

    _rev_gadget_idxs: List[Tuple[Literal["Z", "X"], int]]
    """
        Reverse gadget index list: to each global gadget index `idx` it returns
        a pair with the gadget basis `basis` and the column index `c` for the gadget
        in `self._gadget_idxs[basis]` and `self._matrix[basis]`.
    """

    def __init__(self, num_qubits: int, gadgets: Sequence[PhaseGadget] = tuple()):
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise TypeError("Number of qubits must be a positive integer.")
        if (not isinstance(gadgets, Sequence)
            or not all(isinstance(g, PhaseGadget) for g in gadgets)): # pylint: disable = C0330
            raise TypeError("Gadgets should be a sequence of PhaseGadget.")
        self._num_qubits = num_qubits
        # Fills the lists of original indices and angles for the gadgets:
        self._gadget_idxs = {"Z": [], "X": []}
        self._angles = []
        self._rev_gadget_idxs = []
        for i, gadget in enumerate(gadgets):
            self._rev_gadget_idxs.append((gadget.basis, len(self._gadget_idxs[gadget.basis])))
            self._gadget_idxs[gadget.basis].append(i)
            angle = gadget.angle
            if isinstance(angle, Angle):
                angle %= 2*pi
            self._angles.append(angle)
        self._matrix = {}
        self._gadget_legs_cache = {}
        for basis in cast(Sequence[Literal["Z", "X"]], ("Z", "X")):
            # Create a zero matrix for the basis:
            self._matrix[basis] = np.zeros(shape=(num_qubits, len(self._gadget_idxs[basis])),
                                           dtype=np.uint8)
            # Set matrix elements to 1 for all qubits spanned by the gadgets for the basis:
            legs_cache: List[Optional[Tuple[int, ...]]] = []
            self._gadget_legs_cache[basis] = legs_cache
            for i, idx in enumerate(self._gadget_idxs[basis]):
                for q in gadgets[idx].qubits:
                    if not 0 <= q < num_qubits:
                        msg = f"Qubit {q} for gadget #{idx} out of bounds ({num_qubits = })"
                        raise IndexError(msg)
                    self._matrix[basis][q, i] = 1
                legs_cache.append(tuple(sorted(gadgets[idx].qubits)))

    @property
    def num_qubits(self) -> int:
        """
            Readonly property exposing the number of qubits spanned by this phase circuit.
        """
        return self._num_qubits

    @property
    def num_gadgets(self) -> int:
        """
            Readonly property exposing the number of phase gadgets in the circuit.
        """
        return len(self._angles)

    @property
    def gadgets(self) -> Sequence[PhaseGadget]:
        """
            Readonly property returning the sequence of phase gadgets in this
            phase circuit, in order from first to last.

            This collection is freshly generated at every call.
        """
        return tuple(self._iter_gadgets())

    @property
    def as_readonly(self) -> "PhaseCircuitView":
        """
            Returns a readonly view on this circuit.
        """
        return PhaseCircuitView(self)

    def set_angles(self, angles: Sequence[AngleExpr]) -> None:
        """
            Sets all angles for this circuit.
        """
        if not isinstance(angles, Sequence):
            raise TypeError(f"Expected Sequence[AngleExpr], found {type(angles)}.")
        for angle in angles:
            if not isinstance(angle, AngleExpr):
                raise TypeError(f"Expected AngleExpr, found {type(angle)}")
        if len(angles) != len(self._angles):
            raise ValueError(f"Expected {len(self._angles)} angles, "
                             f"found {len(angles)} instead.")
        pi2 = 2*pi
        self._angles = [angle%pi2 if isinstance(angle, Angle) else angle
                        for angle in angles]

    def refresh_angle_vars(self, params: Union[str, Callable[[int], AngleVar]]) -> None:
        if isinstance(params, str):
            params = lambda i: AngleVar(f"{params}[{i}]", f"{params}_{i}")
        new_angles = [
            angle if isinstance(angle, Angle) else params(i)
            for i, angle in enumerate(self._angles)
        ]
        self.set_angles(new_angles)

    def rx(self, qubit: int, angle: AngleExpr) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit X rotation. """
        if not isinstance(qubit, int) or not 0 <= qubit < self.num_qubits:
            raise TypeError(f"Invalid qubit {qubit}")
        self >>= X(angle) @ {qubit}
        return self

    def rz(self, qubit: int, angle: AngleExpr) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit Z rotation. """
        if not isinstance(qubit, int) or not 0 <= qubit < self.num_qubits:
            raise TypeError(f"Invalid qubit {qubit}")
        self >>= Z(angle) @ {qubit}
        return self

    def ry(self, qubit: int, angle: AngleExpr) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit Y rotation. """
        self.rx(qubit, +pi/2)
        self.rz(qubit, angle)
        self.rx(qubit, -pi/2)
        return self

    def i(self, qubit: int) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit I gate. """
        return self

    def x(self, qubit: int) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit X gate. """
        self.rx(qubit, pi)
        return self

    def z(self, qubit: int) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit X gate. """
        self.rz(qubit, pi)
        return self

    def y(self, qubit: int) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit Y gate. """
        self.z(qubit)
        self.x(qubit)
        return self

    def s(self, qubit: int) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit S gate. """
        self.rz(qubit, pi/2)
        return self

    def sdg(self, qubit: int) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit Sdg gate. """
        self.rz(qubit, -pi/2)
        return self

    def v(self, qubit: int) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit S gate. """
        self.rx(qubit, pi/2)
        return self

    def vdg(self, qubit: int) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit Sdg gate. """
        self.rx(qubit, -pi/2)
        return self

    def t(self, qubit: int) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit T gate. """
        self.rz(qubit, pi/4)
        return self

    def h(self, qubit: int,
          basis: Literal["Z", "X"] = "Z",
          sign: Literal[1, -1]=1) -> "PhaseCircuit":
        """ Phase gadget implementation of single-qubit Hadamard gate. """
        if basis not in ("Z", "X"):
            raise TypeError(f"Invalid basis {basis}.")
        if sign not in (1, -1):
            raise TypeError(f"Invalid sign {sign}.")
        if basis == "Z":
            self.rx(qubit, sign*pi/2)
            self.rz(qubit, sign*pi/2)
            self.rx(qubit, sign*pi/2)
        else:
            self.rz(qubit, sign*pi/2)
            self.rx(qubit, sign*pi/2)
            self.rz(qubit, sign*pi/2)
        return self

    def cu1(self, ctrl: int, tgt: int, angle: AngleExpr) -> "PhaseCircuit":
        """ Phase gadget implementation of CU1 gate. """
        self.add_gadget(Z(-angle) @ {ctrl, tgt})
        self.rz(ctrl, angle)
        self.rz(tgt, angle)
        return self

    def crz(self, ctrl: int, tgt: int, angle: AngleExpr) -> "PhaseCircuit":
        """ Phase gadget implementation of CRZ gate. """
        self.add_gadget(Z(-angle / 2) @ {ctrl, tgt})
        self.rz(tgt, angle / 2)
        return self

    def cry(self, ctrl: int, tgt: int, angle: AngleExpr) -> "PhaseCircuit":
        """ Phase gadget implementation of CRY gate. """
        self.v(tgt)
        self.crz(ctrl, tgt, angle)
        self.vdg(tgt)
        return self

    def crx(self, ctrl: int, tgt: int, angle: Angle) -> "PhaseCircuit":
        """ Phase gadget implementation of CRX gate. """
        self.h(tgt)
        self.crz(ctrl, tgt, angle)
        self.h(tgt, sign=-1)
        return self

    def cz(self, leg1: int, leg2: int) -> "PhaseCircuit":
        """ Phase gadget implementation of CZ gate. """
        self.cu1(leg1, leg2, pi / 2)
        return self

    def cy(self, leg1: int, leg2: int) -> "PhaseCircuit":
        """ Phase gadget implementation of CY gate. """
        self.v(leg1)
        self.cz(leg1, leg2)
        self.vdg(leg2)
        return self

    def cx(self, ctrl: int, tgt: int) -> "PhaseCircuit":
        """ Phase gadget implementation of CX gate. """
        self.h(tgt)
        self.cz(ctrl, tgt)
        self.h(tgt, sign=-1)
        return self

    def u3(self, qubit: int, theta: AngleExpr, phi: AngleExpr, lam: AngleExpr) -> "PhaseCircuit":
        """ Phase gadget implementation of U3 gate. """
        self.rz(qubit, lam)
        self.ry(qubit, theta)
        self.rz(qubit, phi)
        return self

    def ccz(self, leg1: int, leg2: int, leg3: int) -> "PhaseCircuit":
        """ Phase gadget implementation of CCZ gate. """
        for bit in (leg1, leg2, leg3):
            self.rz(bit, pi / 4)
        for src, tgt in ((leg1, leg2), (leg2, leg3), (leg3, leg1)):
            self.add_gadget(Z(-pi / 4) @ {src, tgt})
        self.add_gadget(Z(pi / 4) @ (leg1, leg2, leg3))
        return self

    def ccy(self, leg1: int, leg2: int, leg3: int) -> "PhaseCircuit":
        """ Phase gadget implementation of CCX gate. """
        self.v(leg3)
        self.ccz(leg1, leg2, leg3)
        self.vdg(leg3)
        return self

    def ccx(self, leg1: int, leg2: int, leg3: int) -> "PhaseCircuit":
        """ Phase gadget implementation of CCX gate. """
        self.h(leg3)
        self.ccz(leg1, leg2, leg3)
        self.h(leg3, sign=-1)
        return self

    def add_gadget(self, gadget: PhaseGadget) -> "PhaseCircuit":
        """
            Adds a phase gadget to the circuit.
            This is rather less efficient than passing the gadgets in the constructor,
            because the internal numpy arrays have to be copied in the process.

            The circuit is modified in-place and then returned, as per the
            [fluent interface pattern](https://en.wikipedia.org/wiki/Fluent_interface).
        """
        if not isinstance(gadget, PhaseGadget):
            raise TypeError(f"Expected PhaseGadget, found {type(gadget)}.")
        basis = gadget.basis
        gadget_idx = len(self._angles)
        new_col: npt.NDArray[np.uint64] = np.zeros(shape=(self._num_qubits, 1),
                                                   dtype=np.uint64)
        for q in gadget.qubits:
            new_col[q] = 1
        self._matrix[basis] = np.append(self._matrix[basis], new_col, axis=1)
        self._rev_gadget_idxs.append((basis, len(self._gadget_idxs[gadget.basis])))
        self._gadget_idxs[basis].append(gadget_idx)
        self._angles.append(gadget.angle)
        self._gadget_legs_cache[basis].append(tuple(sorted(gadget.qubits)))
        return self

    def cx_count(self, topology: Topology, *,
                 mapping: Optional[Union[Sequence[int], Dict[int, int]]] = None) -> int:
        """
            Returns the CX count for an implementation of this phase gadget
            on the given topology based on minimum spanning trees (MST).

            The optional `mapping` keyword argument can be used to specify a mapping of
            logical (circuit) qubits to phyisical (topology) qubits.
        """
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        if mapping is not None and len(mapping) != self.num_qubits:
            raise TypeError(f"Expected {self.num_qubits} mapping entries, "
                            f"found {len(mapping)}")
        if mapping is not None and isinstance(mapping, Sequence):
            mapping = {
                i: mapping[i] for i in range(len(mapping))
            }
        if mapping is not None and set(mapping.values()) != set(range(self.num_qubits)):
            raise TypeError(f"Expected mapping images [0, ..., {self.num_qubits-1}], "
                            f"found {sorted(set(mapping.values()))}")
        if mapping is not None:
            # use the reverse mapping on the topology
            topology = topology.mapped_fwd({
                mapping[i]: i for i in mapping
            })
        return self._cx_count(topology, {})

    def mapped(self, mapping: Union[Sequence[int], Mapping[int, int]]) -> "PhaseCircuit":
        """
            Returns a new phase circuit with the same gadgets but having
            qubits remapped according to the given mapping.
        """
        if isinstance(mapping, Sequence):
            if len(mapping) < self.num_qubits:
                raise ValueError(f"Expected mapping keys [0,...,{self._num_qubits}], "
                                 f"found {sorted(mapping)} instead.")
            _mapping = list(mapping)
        elif isinstance(mapping, Mapping):
            _mapping = []
            for i in range(self._num_qubits):
                if i not in mapping:
                    raise ValueError(f"Expected mapping keys [0,...,{self._num_qubits}], "
                                     f"found {sorted(mapping.keys())} instead.")
                _mapping.append(mapping[i])
        else:
            raise TypeError(f"Expected Sequence[int] or Mapping[int, int], "
                            f"found {type(mapping)} instead.")
        if set(_mapping) != set(range(self._num_qubits)):
            raise ValueError(f"Expected mapping values [0,...,{self._num_qubits}], "
                             f"found {sorted(_mapping)} instead.")
        remapped_gadgets = [
            PhaseGadget(g.basis, g.angle, [_mapping[q] for q in g.qubits])
            for g in self.gadgets
        ]
        return PhaseCircuit(self._num_qubits, remapped_gadgets)

    def to_qiskit(self, topology: Topology) -> Any:
        """
            Returns this circuit as a Qiskit circuit.

            This method relies on the `qiskit` library being available.
            Specifically, the `circuit` argument must be of type
            `qiskit.providers.BaseBackend`.
        """
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        try:
            # pylint: disable = import-outside-toplevel
            from qiskit.circuit import QuantumCircuit
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("You must install the 'qiskit' library.") from e
        circuit = QuantumCircuit(self.num_qubits)
        for gadget in self.gadgets:
            gadget.on_qiskit_circuit(topology, circuit)
        return circuit

    @overload
    def to_svg(self, *,
               zcolor: str = "#CCFFCC",
               xcolor: str = "#FF8888",
               hscale: float = 1.0, vscale: float = 1.0,
               scale: float = 1.0,
               svg_code_only: Literal[True]
               ) -> str:
        ...

    @overload
    def to_svg(self, *,
               zcolor: str = "#CCFFCC",
               xcolor: str = "#FF8888",
               hscale: float = 1.0, vscale: float = 1.0,
               scale: float = 1.0,
               svg_code_only: Literal[False] = False
               ) -> Any:
        ...

    def to_svg(self, *,
               zcolor: str = "#CCFFCC",
               xcolor: str = "#FF8888",
               hscale: float = 1.0, vscale: float = 1.0,
               scale: float = 1.0,
               svg_code_only: bool = False
               ) -> Any:
        """
            Returns an SVG representation of this circuit, using
            the ZX calculus to express phase gadgets.

            The keyword arguments `zcolor` and `xcolor` can be used to
            specify a colour for the Z and X basis spiders in the circuit.
            The keyword arguments `hscale` and `vscale` can be used to
            scale the circuit representation horizontally and vertically.
            The keyword argument `scale` can be used to scale the circuit
            representation isotropically.
            The keyword argument `svg_code_only` (default `False`) can be used
            to specify that the SVG code itself be returned, rather than the
            IPython `SVG` object.
        """
        if not isinstance(zcolor, str):
            raise TypeError("Keyword argument 'zcolor' must be string.")
        if not isinstance(xcolor, str):
            raise TypeError("Keyword argument 'xcolor' must be string.")
        if not isinstance(hscale, (int, float)) or hscale <= 0.0:
            raise TypeError("Keyword argument 'hscale' must be positive float.")
        if not isinstance(vscale, (int, float)) or vscale <= 0.0:
            raise TypeError("Keyword argument 'vscale' must be positive float.")
        if not isinstance(scale, (int, float)) or scale <= 0.0:
            raise TypeError("Keyword argument 'scale' must be positive float.")
        return self._to_svg(zcolor=zcolor, xcolor=xcolor,
                            hscale=hscale, vscale=vscale, scale=scale,
                            svg_code_only=svg_code_only) # type: ignore[call-overload]

    @overload
    def _to_svg(self, *,
                zcolor: str = "#CCFFCC",
                xcolor: str = "#FF8888",
                hscale: float = 1.0, vscale: float = 1.0,
                scale: float = 1.0,
                svg_code_only: Literal[True]
                ) -> str:
        ...

    @overload
    def _to_svg(self, *,
                zcolor: str = "#CCFFCC",
                xcolor: str = "#FF8888",
                hscale: float = 1.0, vscale: float = 1.0,
                scale: float = 1.0,
                svg_code_only: Literal[False] = False
                ) -> Any:
        ...

    def _to_svg(self, *,
                zcolor: str = "#CCFFCC",
                xcolor: str = "#FF8888",
                hscale: float = 1.0, vscale: float = 1.0,
                scale: float = 1.0,
                svg_code_only: bool = False
                ) -> Any:
        # pylint: disable = too-many-locals, too-many-statements
        # TODO: clean this up, restructure into a separate function, reuse for opt circuit
        num_qubits = self._num_qubits
        vscale *= scale
        hscale *= scale
        gadgets = self.gadgets
        num_digits = int(ceil(log10(num_qubits)))
        line_height = int(ceil(30*vscale))
        row_width = int(ceil(120*hscale))
        pad_x = int(ceil(10*hscale))
        margin_x = int(ceil(40*hscale))
        pad_y = int(ceil(20*vscale))
        r = pad_y//2-2
        font_size = 2*r
        pad_x += font_size*(num_digits+1)
        delta_fst = row_width//4
        delta_snd = 2*row_width//4
        width = 2*pad_x + 2*margin_x + row_width*len(gadgets)
        height = pad_y + line_height*(num_qubits+1)
        builder = SVGBuilder(width, height)
        levels: List[int] = [0 for _ in range(num_qubits)]
        max_lvl = 0
        for gadget in gadgets:
            fill = zcolor if gadget.basis == "Z" else xcolor
            other_fill = xcolor if gadget.basis == "Z" else zcolor
            qubit_span = range(min(gadget.qubits), max(gadget.qubits)+1)
            lvl = max(levels[q] for q in qubit_span)
            max_lvl = max(max_lvl, lvl)
            x = pad_x + margin_x + lvl * row_width
            for q in qubit_span:
                levels[q] = lvl+1
            if len(gadget.qubits) > 1:
                text_y = pad_y+min(gadget.qubits)*line_height+line_height//2
                for q in gadget.qubits:
                    y = pad_y + (q+1)*line_height
                    builder.line((x, y), (x+delta_fst, text_y))
                for q in gadget.qubits:
                    y = pad_y + (q+1)*line_height
                    builder.circle((x, y), r, fill)
                builder.line((x+delta_fst, text_y), (x+delta_snd, text_y))
                builder.circle((x+delta_fst, text_y), r, other_fill)
                builder.circle((x+delta_snd, text_y), r, fill)
                builder.text((x+delta_snd+2*r, text_y), str(gadget.angle), font_size=font_size)
            else:
                for q in gadget.qubits:
                    y = pad_y + (q+1)*line_height
                    builder.circle((x, y), r, fill)
                builder.text((x+r, y-line_height//3), str(gadget.angle), font_size=font_size)
        width = 2*pad_x + 2*margin_x + row_width*(2*max_lvl+1)//2
        _builder = SVGBuilder(width, height)
        for q in range(num_qubits):
            y = pad_y + (q+1) * line_height
            _builder.line((pad_x, y), (width-pad_x, y))
            _builder.text((0, y), f"{str(q):>{num_digits}}", font_size=font_size)
            _builder.text((width-pad_x+r, y), f"{str(q):>{num_digits}}", font_size=font_size)
        _builder >>= builder
        svg_code = repr(_builder)
        if svg_code_only:
            return svg_code
        try:
            # pylint: disable = import-outside-toplevel
            from IPython.core.display import SVG # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("You must install the 'IPython' library.") from e
        return SVG(svg_code)

    def cloned(self) -> "PhaseCircuit":
        """
            Produces an exact copy of this phase circuit.
        """
        return PhaseCircuit(self._num_qubits, tuple(self._iter_gadgets()))

    def conj_by_cx(self, ctrl: int, trgt: int) -> "PhaseCircuit":
        """
            Conjugates this circuit by a CX gate with given control/target.
            The circuit is modified in-place and then returned, as per the
            [fluent interface pattern](https://en.wikipedia.org/wiki/Fluent_interface).
        """
        if not 0 <= ctrl < self._num_qubits:
            raise ValueError(f"Invalid control qubit {ctrl}.")
        if not 0 <= trgt < self._num_qubits:
            raise ValueError(f"Invalid target qubit {trgt}.")
        self._matrix["Z"][ctrl, :] = (self._matrix["Z"][ctrl, :] + self._matrix["Z"][trgt, :]) % 2
        self._matrix["X"][trgt, :] = (self._matrix["X"][trgt, :] + self._matrix["X"][ctrl, :]) % 2
        # Update legs caches:
        z_gadget_legs_cache = self._gadget_legs_cache["Z"]
        for z_gadget_idx in np.where(self._matrix["Z"][trgt, :] == 1)[0]:
            z_gadget_legs_cache[z_gadget_idx] = None
        x_gadget_legs_cache = self._gadget_legs_cache["X"]
        for x_gadget_idx in np.where(self._matrix["X"][ctrl, :] == 1)[0]:
            x_gadget_legs_cache[x_gadget_idx] = None
        return self

    # def simplified(self) -> "PhaseCircuit":
    #     """
    #         Returns a new phase circuit which has been simplified using the
    #         commutation and fusion rules for gadgets.
    #     """
    #     # pylint: disable = too-many-locals, too-many-branches, too-many-statements
    #     num_qubits = self.num_qubits
    #     gadgets = [g for g in self.gadgets if not g.angle.is_zero]
    #     # Groups of gadgets of the same basis, fused together where possible
    #     GadgetGroup = Tuple[Literal["Z", "X"], Dict[FrozenSet[int], Angle]]
    #     groups: List[GadgetGroup] = [("Z", {})]
    #     # Perform the grouping and fusion
    #     for g in gadgets:
    #         basis, angles = groups[-1]
    #         g_basis = g.basis
    #         g_qubits = g.qubits
    #         g_angle = g.angle
    #         # Add the gadget to the current group, or create a new group.
    #         if g_basis == basis:
    #             # Add gadget to current group (fuse if possible)
    #             if g_qubits in angles:
    #                 angles[g_qubits] += g_angle
    #             else:
    #                 angles[g_qubits] = g_angle
    #         else:
    #             # Create a new group (basis has changed)
    #             groups.append((g_basis, {g_qubits: g_angle}))
    #     # The pi gates will be collected separately here
    #     pi_gates = {
    #         "Z": [0 for _ in range(num_qubits)],
    #         "X": [0 for _ in range(num_qubits)]
    #     }
    #     # Perform all commutations, fusions and pi gadget simplifications
    #     for i, (basis, angles) in enumerate(groups): # pylint: disable = too-many-nested-blocks
    #         # Try commuting all gadgets to the left as much as possible
    #         for qubits, angle in angles.items():
    #             if angle == 0:
    #                 # Skip zeroed gadgets
    #                 continue
    #             # Try to commute the gadget to the left as much as possible
    #             j = i # j is the current group to which the gadget has been commuted
    #             obstacle_found = False # this records whether we found an obstacle
    #             while not obstacle_found and j >= 2:
    #                 _, angles_commute = groups[j-1] # angles to commute through
    #                 for qubits_commute, angle_commute in angles_commute.items():
    #                     if angle_commute.is_zero:
    #                         # Zero angle gadget, not an obstable
    #                         continue
    #                     if len(qubits&qubits_commute) % 2 != 0:
    #                         # Odd number of shared legs, obstacle found
    #                         obstacle_found = True
    #                         break
    #                 if not obstacle_found:
    #                     # Go to the next group of same basis down the list
    #                     j -= 2
    #             # Fuse the gadget into the group, and apply pi gate simplification
    #             pi_gadget = False
    #             if j < i:
    #                 # We managed to perform some non-trivial commutation
    #                 angles[qubits] = Angle.zero
    #                 _, angles_fuse = groups[j]
    #                 if qubits in angles_fuse:
    #                     # Fuse with existing gadget on same qubits and same basis
    #                     angles_fuse[qubits] += angle
    #                 else:
    #                     angles_fuse[qubits] = angle
    #                     # Add gadget to group
    #                 if angles_fuse[qubits].is_pi:
    #                     # This is a pi gadget, further simplification to be performed
    #                     angles_fuse[qubits] = Angle.zero # Remove gadget from this group
    #                     pi_gadget = True
    #             elif angle.is_pi:
    #                 # We didn't manage to commute the gadget, but it is a pi gadget
    #                 angles[qubits] = Angle.zero # Remove gadget from this group
    #                 pi_gadget = True
    #             if pi_gadget:
    #                 # pi gadget
    #                 for k in range(0, j)[::-2]:
    #                     # Commute through gadgets below of other basis, flipping sign if necessary
    #                     _, angles_k = groups[k]
    #                     for qubits_k in angles_k:
    #                         if len(qubits_k&qubits)%2 == 1:
    #                             # Odd number of legs in comon: flip sign
    #                             angles_k[qubits_k] *= -1
    #                 for q in qubits:
    #                     # Break into single-qubit pi gates, recorded separately (at start of circ)
    #                     pi_gates[basis][q] += 1
    #     # Create the new list of gadgets
    #     new_gadgets: List[PhaseGadget] = []
    #     for q in range(num_qubits):
    #         if pi_gates["Z"][q]%2 == 1:
    #             # Single-qubit pi Z gate
    #             new_gadgets.append(PhaseGadget("Z", pi, {q}))
    #     for q in range(num_qubits):
    #         if pi_gates["X"][q]%2 == 1:
    #             # Single-qubit pi X gate
    #             new_gadgets.append(PhaseGadget("X", pi, {q}))
    #     for basis, angles in groups:
    #         for qubits, angle in angles.items():
    #             angle = angle % (2*pi)
    #             if angle != 0: # skip zero angle gadgets
    #                 new_gadgets.append(PhaseGadget(basis, angle, qubits))
    #     # Return a new phase circuit.
    #     return PhaseCircuit(num_qubits, new_gadgets)

    def simplified(self) -> "PhaseCircuit":
        """
            Returns a new phase circuit which has been simplified using the
            commutation and fusion rules for gadgets.
        """
        # pylint: disable = too-many-locals, too-many-branches, too-many-statements
        num_qubits = self.num_qubits
        gadgets = [g for g in self.gadgets if not g.angle.is_zero]
        # Groups of gadgets of the same basis, fused together where possible
        GadgetGroup = Tuple[Literal["Z", "X"], Dict[int, AngleExpr]]
        groups: List[GadgetGroup] = [("Z", {})]
        # Perform the grouping and fusion
        for g in gadgets:
            basis, angles = groups[-1]
            g_basis = g.basis
            g_qubits = _frozenset_to_int(g.qubits)
            g_angle = g.angle
            # Add the gadget to the current group, or create a new group.
            if g_basis == basis:
                # Add gadget to current group (fuse if possible)
                if g_qubits in angles:
                    angles[g_qubits] += g_angle
                else:
                    angles[g_qubits] = g_angle
            else:
                # Create a new group (basis has changed)
                groups.append((g_basis, {g_qubits: g_angle}))
        # The pi gates will be collected separately here
        pi_gates = {
            "Z": [0 for _ in range(num_qubits)],
            "X": [0 for _ in range(num_qubits)]
        }
        # Perform all commutations, fusions and pi gadget simplifications
        # TODO: explain with comments how the jumplist works
        # jumplist: Dict[int, Optional[Dict[int, int]]] = {}
        for i, (basis, angles) in enumerate(groups): # pylint: disable = too-many-nested-blocks
            # Try commuting all gadgets to the left as much as possible
            for qubits, angle in angles.items():
                if angle == 0:
                    # Skip zeroed gadgets
                    continue
                # Try to commute the gadget to the left as much as possible
                j = i # j is the current group to which the gadget has been commuted
                obstacle_found = False # this records whether we found an obstacle
                while not obstacle_found and j >= 2:
                    # if j in jumplist:
                    #     j_jump = jumplist[j]
                    #     if j_jump is not None and qubits in j_jump:
                    #         print("Jump", i, j, j_jump[qubits], bin(qubits))
                    #         j = j_jump[qubits]
                    #         break
                    _, angles_commute = groups[j-1] # angles to commute through
                    for qubits_commute, angle_commute in angles_commute.items():
                        if angle_commute.is_zero:
                            # Zero angle gadget, not an obstable
                            continue
                        # https://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-positive-integer
                        if bin(qubits&qubits_commute).count("1") % 2 != 0:
                            # Odd number of shared legs, obstacle found
                            obstacle_found = True
                            break
                    if not obstacle_found:
                        # Go to the next group of same basis down the list
                        j -= 2
                # if i in jumplist and jumplist[i] is not None:
                #     jumplist[i][qubits] = j # type: ignore # TODO: do this right
                # else:
                #     jumplist[i] = {
                #         qubits: j
                #     }
                # Fuse the gadget into the group, and apply pi gate simplification
                pi_gadget = False
                if j < i:
                    # for k in range(j+1, i, 2):
                    #     jumplist[k] = None
                    # We managed to perform some non-trivial commutation
                    angles[qubits] = Angle.zero
                    _, angles_fuse = groups[j]
                    if qubits in angles_fuse:
                        # Fuse with existing gadget on same qubits and same basis
                        angles_fuse[qubits] += angle
                    else:
                        # Add gadget to group
                        angles_fuse[qubits] = angle
                    if angles_fuse[qubits].is_pi:
                        # This is a pi gadget, further simplification to be performed
                        angles_fuse[qubits] = Angle.zero # Remove gadget from this group
                        pi_gadget = True
                elif angle.is_pi:
                    # We didn't manage to commute the gadget, but it is a pi gadget
                    angles[qubits] = Angle.zero # Remove gadget from this group
                    pi_gadget = True
                if pi_gadget:
                    # pi gadget
                    for k in range(0, j)[::-2]:
                        # Commute through gadgets below of other basis, flipping sign if necessary
                        _, angles_k = groups[k]
                        for qubits_k in angles_k:
                            if bin(qubits_k&qubits).count("1")%2 == 1:
                                # Odd number of legs in comon: flip sign
                                angles_k[qubits_k] *= -1
                    for q in _int_to_iterator(qubits):
                        # Break into single-qubit pi gates, recorded separately (at start of circ)
                        pi_gates[basis][q] += 1
        # Create the new list of gadgets
        new_gadgets: List[PhaseGadget] = []
        for q in range(num_qubits):
            if pi_gates["Z"][q]%2 == 1:
                # Single-qubit pi Z gate
                new_gadgets.append(PhaseGadget("Z", pi, {q}))
        for q in range(num_qubits):
            if pi_gates["X"][q]%2 == 1:
                # Single-qubit pi X gate
                new_gadgets.append(PhaseGadget("X", pi, {q}))
        for basis, angles in groups:
            for qubits, angle in angles.items():
                if isinstance(angle, Angle):
                    angle = angle % (2*pi)
                if angle != 0: # skip zero angle gadgets
                    new_gadgets.append(PhaseGadget(basis, angle, _int_to_frozenset(qubits)))
        # Return a new phase circuit.
        return PhaseCircuit(num_qubits, new_gadgets)

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if not isinstance(other, PhaseCircuit):
            return NotImplemented
        if self.num_gadgets != other.num_gadgets:
            return NotImplemented
        if self.num_qubits != other.num_qubits:
            return False
        return all(g == h for g, h in zip(self._iter_gadgets(), other._iter_gadgets()))

    def __irshift__(self, gadgets: Union[PhaseGadget, "PhaseCircuit",
                                         Sequence[PhaseGadget]]) -> "PhaseCircuit":
        if isinstance(gadgets, PhaseGadget):
            gadgets = [gadgets]
        elif isinstance(gadgets, PhaseCircuit):
            gadgets = gadgets.gadgets
        if (not isinstance(gadgets, Sequence)
                or not all(isinstance(gadget, PhaseGadget) for gadget in gadgets)):
            raise TypeError(f"Expected phase gadget or sequence of phase gadgets, found {gadgets}.")
        for gadget in gadgets:
            self.add_gadget(gadget)
        return self

    def __rshift__(self, gadgets: Union[PhaseGadget, "PhaseCircuit",
                                        Sequence[PhaseGadget]]) -> "PhaseCircuit":
        circ: PhaseCircuit = PhaseCircuit(self.num_qubits, [])
        circ >>= self
        circ >>= gadgets
        return circ

    def _reindex(self, idx: int) -> int:
        if 0 <= idx < self.num_gadgets:
            return idx
        if -self.num_gadgets <= idx < 0:
            return idx + self.num_gadgets
        raise IndexError(f"Invalid gadget index {idx}")

    @overload
    def __getitem__(self, idx: int) -> PhaseGadget:
        ...

    @overload
    def __getitem__(self, idx: slice) -> "PhaseCircuit":
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[PhaseGadget, "PhaseCircuit"]:
        if isinstance(idx, int):
            idx = self._reindex(idx)
            basis, col_idx = self._rev_gadget_idxs[idx]
            col = self._matrix[basis][:, col_idx]
            angle = self._angles[idx]
            return PhaseGadget(basis, angle, {i for i, b in enumerate(col) if b % 2 == 1})
        if isinstance(idx, slice):
            start, stop, step = (idx.start, idx.stop, idx.step)
            return PhaseCircuit(self.num_qubits, list(self._iter_gadgets(start, stop, step)))
        raise TypeError(f"Expected int or slice, found {type(idx)}")

    def __len__(self) -> int:
        return self.num_gadgets

    def _iter_gadgets(self,
                      start: Optional[int] = None,
                      stop: Optional[int] = None,
                      step: Optional[int] = None) -> Iterator[PhaseGadget]:
        if start is None:
            start = 0
        else:
            start = self._reindex(start)
        if stop is not None:
            stop = self._reindex(stop)
        for idx, angle in islice(enumerate(self._angles), start, stop, step):
            basis, col_idx = self._rev_gadget_idxs[idx]
            col = self._matrix[basis][:, col_idx]
            yield PhaseGadget(basis, angle, {i for i, b in enumerate(col) if b % 2 == 1})

    def _cx_count(self, topology: Topology, cache: Dict[int, Dict[Tuple[int, ...], int]]) -> int:
        """
            Returns the CX count for an implementation of this phase circuit
            on the given topology based on minimum spanning trees (MST).
        """
        # pylint: disable = too-many-locals
        num_qubits = self._num_qubits
        weight = lambda u, v: 4*topology.dist(u, v)-2
        inf = 4*num_qubits-2
        count = 0
        for basis in ("Z", "X"):
            basis = cast(Literal["Z", "X"], basis)
            gadget_legs_cache = self._gadget_legs_cache[basis]
            for j, col in enumerate(self._matrix[basis].T):
                angle = self._angles[self._gadget_idxs[basis][j]]
                if angle.is_zero_or_pi:
                    # Skip zero and pi gadgets
                    continue
                legs = gadget_legs_cache[j]
                if legs is None:
                    legs = tuple(int(i) for i in np.where(col == 1)[0])
                    gadget_legs_cache[j] = legs
                num_legs = len(legs)
                if num_legs <= 1:
                    # Skip single-qubit gates
                    continue
                _cache = cache.get(num_legs, None)
                if _cache is None:
                    _cache = {}
                    cache[num_legs] = _cache
                legs_count = _cache.get(legs, None)
                if legs_count is None:
                    legs_count = _prims_algorithm_weight(legs, weight, inf)
                    _cache[legs] = legs_count
                count += legs_count
        return count

    def _repr_svg_(self) -> Any:
        """
            Magic method for IPython/Jupyter pretty-printing.
            See https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
        """
        return self._to_svg(svg_code_only=True)

    @staticmethod
    def random(num_qubits: int, num_gadgets: int, *,
               parametric: Union[None, str, Callable[[int], AngleVar]] = None,
               angle_subdivision: int = 4,
               min_legs: int = 1,
               max_legs: Optional[int] = None,
               rng_seed: Optional[int] = None) -> "PhaseCircuit":
        """
            Generates a random circuit of mixed ZX phase gadgets on the given number of qubits,
            with the given number of gadgets.

            The optional argument `angle_subdivision` (default: 4) can be used to specify the
            denominator in the random fractional multiples of pi used as values for the angles.

            The optional arguments `min_legs` (default: 1, minimum: 1) and `max_legs`
            (default: `None`, minimum `min_legs`) can be used to specify the minimum and maximum
            number of legs for the phase gadgets. If `None`, `max_legs` is set to `len(qubits)`.

            The optional argument `rng_seed` (default: `None`) is used as seed for the RNG.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise TypeError("Number of qubits must be a positive integer.")
        if not isinstance(num_gadgets, int) or num_gadgets < 0:
            raise TypeError("Number of gadgets must be non-negative integer.")
        if not isinstance(angle_subdivision, int) or angle_subdivision <= 0:
            raise TypeError("Angle subdivision must be positive integer.")
        if not isinstance(min_legs, int) or min_legs <= 0:
            raise TypeError("Minimum legs must be positive integer or 'None'.")
        if min_legs is None:
            min_legs = 1
        if max_legs is not None and (not isinstance(max_legs, int) or max_legs < min_legs):
            raise TypeError("Maximum legs must be positive integer or 'None'.")
        if max_legs is None:
            max_legs = num_qubits
        if rng_seed is not None and not isinstance(rng_seed, int):
            raise TypeError("RNG seed must be integer or 'None'.")
        if parametric is not None:
            if isinstance(parametric, str):
                s = parametric
                parametric = lambda i: AngleVar(f"{s}[{i}]", f"{s}_{i}")
        rng = np.random.default_rng(seed=rng_seed)
        angle_rng_seed = int(rng.integers(65536)) # type: ignore[attr-defined]
        basis_idxs = rng.integers(2, size=num_gadgets) # type: ignore[attr-defined]
        num_legs = rng.integers(min_legs, max_legs+1, size=num_gadgets) # type: ignore[attr-defined]
        legs_list: List[npt.NDArray[int]] = [
            rng.choice(num_qubits, num_legs[i], replace=False) for i in range(num_gadgets)
        ]
        angle_rng = np.random.default_rng(seed=angle_rng_seed)
        angles: List[Union[Angle, AngleVar]]
        angles = [int(x)*pi/angle_subdivision
                  for x in angle_rng.integers(1, 2*angle_subdivision, # type: ignore[attr-defined]
                                              size=num_gadgets)]
        if parametric is not None:
            angles = [parametric(i) for i in range(num_gadgets)]
        bases = cast(Sequence[Literal["Z", "X"]], ("Z", "X"))
        gadgets: List[PhaseGadget] = [
            PhaseGadget(bases[(basis_idx+i)%2],
                        angle,
                        [int(x) for x in legs])
            for i, (basis_idx, angle, legs) in enumerate(zip(basis_idxs,
                                                             angles,
                                                             legs_list))
        ]
        return PhaseCircuit(num_qubits, gadgets)

    @staticmethod
    def from_qasm(qasm: Union[str, QASM], *,
                  mapping: Union[Sequence[int], Mapping[int, int], None] = None,
                  allow_classical: bool = True) -> "PhaseCircuit":
        """
            Constructs a phase circuit from a QASM program.

            An optional mapping from QASM qubits to circuit qubits can be supplied.
        """
        # pylint: disable = too-many-locals, too-many-branches, too-many-statements
        if isinstance(qasm, str):
            qasm = QASM.parse(qasm)
        if not isinstance(qasm, QASM):
            raise TypeError(f"Expected QASM object, found {qasm}")
        if not isinstance(allow_classical, bool):
            raise TypeError()
        qasm_num_qubits = qasm.num_qubits
        num_bits = qasm.num_bits
        if not allow_classical and num_bits > 0:
            raise ValueError("Cannot construct from quantum circuits with classical registers.")
        if isinstance(mapping, Sequence):
            if len(mapping) < qasm_num_qubits:
                raise ValueError(f"Expected mapping keys [0,...,{qasm_num_qubits}], "
                                 f"found {sorted(mapping)} instead.")
            _mapping: Optional[List[int]] = list(mapping)
        elif isinstance(mapping, Mapping):
            _mapping = []
            for i in range(qasm_num_qubits):
                if i not in mapping:
                    raise ValueError(f"Expected mapping keys [0,...,{qasm_num_qubits}], "
                                     f"found {sorted(mapping.keys())} instead.")
                _mapping.append(mapping[i])
        elif mapping is None:
            _mapping = None
        else:
            raise TypeError(f"Expected Sequence[int] or Mapping[int, int] or None, "
                            f"found {type(mapping)} instead.")
        if _mapping is not None and set(_mapping) != set(range(qasm_num_qubits)):
            raise ValueError(f"Expected mapping values [0,...,{qasm_num_qubits}], "
                             f"found {sorted(_mapping)} instead.")
        gadgets = []
        qubits: Dict[str, Tuple[int, ...]] = {}
        qubit_idx = 0
        for reg in qasm.registers:
            if isinstance(reg, QASM.QReg):
                qubits[reg.name] = tuple(qubit_idx + i for i in range(reg.size))
                qubit_idx += reg.size
        if _mapping is not None:
            qubits = {
                reg: tuple(_mapping[i] for i in reg_qs)
                for reg, reg_qs in qubits.items()
            }
        for statement in qasm:
            if isinstance(statement, QASM.Version):
                continue
            if isinstance(statement, QASM.Comment):
                continue
            if isinstance(statement, QASM.Include):
                continue
            if isinstance(statement, QASM.QReg):
                continue
            if isinstance(statement, QASM.CReg):
                continue
            if isinstance(statement, QASM.UGate):
                params = (statement.theta, statement.phi, statement.lam)
                reg_name = statement.qubit.register.name
                if statement.qubit.pos is not None:
                    ugate_qubits = [qubits[reg_name][statement.qubit.pos]]
                else:
                    ugate_qubits = list(qubits[reg_name])
                for q in ugate_qubits:
                    gadgets += _u3(q, *params)
                continue
            if isinstance(statement, QASM.CXGate):
                ctrl_reg_name = statement.control.register.name
                trgt_reg_name = statement.target.register.name
                ctrl_size = statement.control.size
                trgt_size = statement.target.size
                ctrl_pos = statement.control.pos
                trgt_pos = statement.target.pos
                if ctrl_size == 1:
                    ctrl_pos = 0
                if trgt_size == 1:
                    trgt_pos = 0
                if ctrl_pos is None and trgt_pos is None:
                    qubit_pairs = [(qubits[ctrl_reg_name][i], qubits[trgt_reg_name][i])
                                   for i in range(ctrl_size)]
                elif ctrl_pos is not None and trgt_pos is None:
                    qubit_pairs = [(qubits[ctrl_reg_name][ctrl_pos], qubits[trgt_reg_name][i])
                                   for i in range(trgt_size)]
                elif ctrl_pos is None and trgt_pos is not None:
                    qubit_pairs = [(qubits[ctrl_reg_name][i], qubits[trgt_reg_name][trgt_pos])
                                   for i in range(ctrl_size)]
                elif ctrl_pos is not None and trgt_pos is not None:
                    qubit_pairs = [(qubits[ctrl_reg_name][ctrl_pos],
                                    qubits[trgt_reg_name][trgt_pos])]
                for c, t in qubit_pairs:
                    gadgets += _cx(c, t)
                continue
            if isinstance(statement, QASM.Gate):
                reg_name = [t.register.name for t in statement.targets]
                reg_size = [t.register.size for t in statement.targets]
                reg_pos = [t.pos for t in statement.targets]
                for i in range(len(statement.targets)):
                    if reg_size[i] == 1:
                        reg_pos[i] = 0
                if all(p is not None for p in reg_pos):
                    size = 1
                    gate_qubits = [tuple(qubits[name][pos]
                                         for name, pos in zip(reg_name,
                                                              cast(Sequence[int], reg_pos)))]
                elif all(p is None for p in reg_pos):
                    size = reg_size[0]
                    gate_qubits = [tuple(qubits[name][i] for name in reg_name)
                                   for i in range(size)]
                else:
                    raise Exception("This should not happen. Please open a bug report.")
                gate_params = statement.params
                gate_methods = {
                    "i": (_i, 1, 0),
                    "x": (_x, 1, 0),
                    "y": (_y, 1, 0),
                    "z": (_z, 1, 0),
                    "h": (_h, 1, 0),
                    "s": (_s, 1, 0),
                    "t": (_t, 1, 0),
                    "rx": (_rx, 1, 1),
                    "ry": (_ry, 1, 1),
                    "rz": (_rz, 1, 1),
                    "u3": (_u3, 1, 3),
                    "cx": (_cx, 2, 0),
                    "cy": (_cy, 2, 0),
                    "cz": (_cz, 2, 0),
                    "crx": (_crx, 2, 1),
                    "cry": (_cry, 2, 1),
                    "crz": (_crz, 2, 1),
                    "cu1": (_cu1, 2, 1),
                }
                for gate_name, gate_args in gate_methods.items():
                    if statement.name == gate_name:
                        m, num_qubits, num_params = gate_args
                        if len(gate_qubits[0]) != num_qubits:
                            raise ValueError(f"Expected {num_qubits} qubits for {gate_name}, "
                                             f"found {len(gate_qubits[0])}")
                        if len(gate_params) != num_params:
                            raise ValueError(f"Expected {num_params} angles for {gate_name}, "
                                             f"found {len(gate_params)}")
                        for qs in gate_qubits:
                            gadgets += m(*qs, *gate_params) # type: ignore # TODO: fix this!
                        break
                continue
            raise ValueError(f"Unsupported QASM statement: {statement}")
        circ = PhaseCircuit(qasm_num_qubits, gadgets)
        return circ


class PhaseCircuitView:
    """
        Readonly view on a phase circuit.
    """

    _circuit: PhaseCircuit

    def __init__(self, circuit: PhaseCircuit):
        if not isinstance(circuit, PhaseCircuit):
            raise TypeError(f"Expected PhaseCircuit, found {type(circuit)}.")
        self._circuit = circuit

    @property
    def num_qubits(self) -> int:
        """
            Readonly property exposing the number of qubits spanned by the phase circuit.
        """
        return self._circuit.num_qubits

    @property
    def num_gadgets(self) -> int:
        """
            Readonly property exposing the number of phase gadgets in the circuit.
        """
        return self._circuit.num_gadgets

    @property
    def gadgets(self) -> Sequence[PhaseGadget]:
        """
            Readonly property returning the sequence of phase gadgets in the
            phase circuit, in order from first to last.

            This collection is freshly generated at every call.
        """
        return self._circuit.gadgets

    @overload
    def to_svg(self, *,
               zcolor: str = "#CCFFCC",
               xcolor: str = "#FF8888",
               hscale: float = 1.0, vscale: float = 1.0,
               scale: float = 1.0,
               svg_code_only: Literal[True]
               ) -> str:
        ...

    @overload
    def to_svg(self, *,
               zcolor: str = "#CCFFCC",
               xcolor: str = "#FF8888",
               hscale: float = 1.0, vscale: float = 1.0,
               scale: float = 1.0,
               svg_code_only: Literal[False] = False
               ) -> Any:
        ...

    def to_svg(self, *,
               zcolor: str = "#CCFFCC",
               xcolor: str = "#FF8888",
               hscale: float = 1.0, vscale: float = 1.0,
               scale: float = 1.0,
               svg_code_only: bool = False
               ) -> Any:
        # pylint: disable = too-many-locals
        """
            Returns an SVG representation of this circuit, using
            the ZX calculus to express phase gadgets.

            The keyword arguments `zcolor` and `xcolor` can be used to
            specify a colour for the Z and X basis spiders in the circuit.
            The keyword arguments `hscale` and `vscale` can be used to
            scale the circuit representation horizontally and vertically.
            The keyword argument `svg_code_only` (default `False`) can be used
            to specify that the SVG code itself be returned, rather than the
            IPython `SVG` object.
        """
        return self._circuit.to_svg(zcolor=zcolor, xcolor=xcolor,
                                    hscale=hscale, vscale=vscale,
                                    scale=scale,
                                    svg_code_only=svg_code_only) # type: ignore[call-overload]

    def cloned(self) -> PhaseCircuit:
        """
            Produces an exact copy of the phase circuit.
        """
        return self._circuit.cloned()

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if isinstance(other, PhaseCircuit):
            return self._circuit == other
        if isinstance(other, PhaseCircuitView):
            return self._circuit == other._circuit
        return NotImplemented

    def _repr_svg_(self) -> Any:
        """
            Magic method for IPython/Jupyter pretty-printing.
            See https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
        """
        return self._circuit._repr_svg_() # pylint: disable = protected-access
