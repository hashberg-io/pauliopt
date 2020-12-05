"""
    This module contains code to create circuits of mixed ZX phase gadgets.
"""

from collections import deque
from math import ceil, log10
from typing import (Callable, cast, Collection, Dict, FrozenSet, Generic, Iterator, List,
                    Literal, Optional, Sequence, Set, Tuple, Union)
import numpy as np # type: ignore
from pauliopt.topologies import Topology
from pauliopt.utils import AngleT, AngleProtocol, Angle, SVGBuilder, pi

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


class PhaseGadget(Generic[AngleT]):
    """
        Immutable container class for a phase gadget.
    """

    _qubits: FrozenSet[int]
    _basis: Literal["Z", "X"]
    _angle: AngleT

    def __init__(self, basis: Literal["Z", "X"], angle: AngleT, qubits: Collection[int]):
        if not isinstance(qubits, Collection) or not all(isinstance(q, int) for q in qubits):
            raise TypeError(f"Qubits should be a collection of integers, found {qubits}")
        if not qubits:
            raise ValueError("At least one qubit must be specified.")
        if basis not in ("Z", "X"):
            raise TypeError("Basis should be 'Z' or 'X'.")
        if not isinstance(angle, AngleProtocol):
            raise TypeError(f"Angle should respect the `AngleProtocol` Protocol, "
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
    def angle(self) -> AngleT:
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

    def cx_count(self, topology: Topology) -> int:
        """
            Returns the CX count for an implementation of this phase gadget
            on the given topology based on minimum spanning trees (MST).
        """
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        return  _prims_algorithm_weight(self._qubits,
                                        lambda u, v: 4*topology.dist(u, v)-2,
                                        4*len(topology.qubits)-2)

    def on_qiskit_circuit(self, topology: Topology, circuit) -> None:
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
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'qiskit' library.")
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
            circuit.rz(float(self.angle), q0)
        else:
            circuit.rx(float(self.angle), q0)
        for ctrl, trgt in upper_ladder:
            circuit.cx(ctrl, trgt)

    def print_impl_info(self, topology: Topology):
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

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, PhaseGadget):
            return NotImplemented
        return (self.basis == other.basis
                and self.angle == other.angle
                and self.qubits == other.qubits)


class Z(Generic[AngleT]):
    """
        Constructs a Z phase gadget with the idiomatic syntax:

        ```py
            Z(angle) @ qubits
        ```
    """

    _angle: AngleT

    def __init__(self, angle: AngleT):
        if not isinstance(angle, AngleProtocol):
            raise TypeError(f"Angle should respect the `AngleProtocol` Protocol, "
                            f"found {angle} of type {type(angle)} instead.")
        self._angle = angle

    def __matmul__(self, qubits: Collection[int]) -> PhaseGadget[AngleT]:
        return PhaseGadget("Z", self._angle, qubits)


class X(Generic[AngleT]):
    """
        Constructs an X phase gadget with the idiomatic syntax:

        ```py
            X(angle) @ qubits
        ```
    """

    _angle: AngleT

    def __init__(self, angle: AngleT):
        if not isinstance(angle, AngleProtocol):
            raise TypeError(f"Angle should respect the `AngleProtocol` Protocol, "
                            f"found {angle} of type {type(angle)} instead.")
        self._angle = angle

    def __matmul__(self, qubits: Collection[int]) -> PhaseGadget[AngleT]:
        return PhaseGadget("X", self._angle, qubits)


class PhaseCircuit(Generic[AngleT]):
    """
        Container class for a circuit of mixed ZX phase gadgets.
    """

    _matrix: Dict[Literal["Z", "X"], np.ndarray]
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

    _angles: List[AngleT]
    """
        The global list of angles for the gadgets.
        The angle for the `basis` gadget corresponding to column index `c`
        of matrix `self._matrix[basis]` is given by:

        ```py
            self._angles[self._gadget_idxs[basis][c]]
        ```
    """

    def __init__(self, num_qubits: int, gadgets: Sequence[PhaseGadget[AngleT]] = tuple()):
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise TypeError("Number of qubits must be a positive integer.")
        if (not isinstance(gadgets, Sequence)
            or not all(isinstance(g, PhaseGadget) for g in gadgets)): # pylint: disable = C0330
            raise TypeError("Gadgets should be a sequence of PhaseGadget.")
        self._num_qubits = num_qubits
        # Fills the lists of original indices and angles for the gadgets:
        self._gadget_idxs = {"Z": [], "X": []}
        self._angles = []
        for i, gadget in enumerate(gadgets):
            self._gadget_idxs[gadget.basis].append(i)
            self._angles.append(gadget.angle)
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
    def as_readonly(self) -> "PhaseCircuitView[AngleT]":
        """
            Returns a readonly view on this circuit.
        """
        return PhaseCircuitView(self)

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
        new_col = np.zeros(shape=(self._num_qubits, 1), dtype=np.uint64)
        for q in gadget.qubits:
            new_col[q] = 1
        self._matrix[basis] = np.append(self._matrix[basis], new_col, axis=1)
        self._gadget_idxs[basis].append(gadget_idx)
        self._angles.append(gadget.angle)
        return self

    def cx_count(self, topology: Topology) -> int:
        """
            Returns the CX count for an implementation of this phase gadget
            on the given topology based on minimum spanning trees (MST).
        """
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        return self._cx_count(topology, {})

    def to_qiskit(self, topology: Topology):
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
            from qiskit.circuit import QuantumCircuit # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'qiskit' library.")
        circuit = QuantumCircuit(self.num_qubits)
        for gadget in self.gadgets:
            gadget.on_qiskit_circuit(topology, circuit)
        return circuit

    def to_svg(self, *,
               zcolor: str = "#CCFFCC",
               xcolor: str = "#FF8888",
               hscale: float = 1.0, vscale: float = 1.0,
               scale: float = 1.0,
               svg_code_only: bool = False
               ):
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
                            svg_code_only=svg_code_only)
    def _to_svg(self, *,
                zcolor: str = "#CCFFCC",
                xcolor: str = "#FF8888",
                hscale: float = 1.0, vscale: float = 1.0,
                scale: float = 1.0,
                svg_code_only: bool = False
                ):
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
        delta_fst = row_width//2
        delta_snd = 3*row_width//4
        width = 2*pad_x + 2*margin_x + row_width*len(gadgets)
        height = pad_y + line_height*(num_qubits+1)
        builder = SVGBuilder(width, height)
        for q in range(num_qubits):
            y = pad_y + (q+1) * line_height
            builder.line((pad_x, y), (width-pad_x, y))
            builder.text((0, y), f"{str(q):>{num_digits}}", font_size=font_size)
            builder.text((width-pad_x+r, y), f"{str(q):>{num_digits}}", font_size=font_size)
        for row, gadget in enumerate(gadgets):
            fill = zcolor if gadget.basis == "Z" else xcolor
            other_fill = xcolor if gadget.basis == "Z" else zcolor
            x = pad_x + margin_x + row * row_width
            for q in gadget.qubits:
                y = pad_y + (q+1)*line_height
                builder.line((x, y), (x+delta_fst, pad_y))
            for q in gadget.qubits:
                y = pad_y + (q+1)*line_height
                builder.circle((x, y), r, fill)
            builder.line((x+delta_fst, pad_y), (x+delta_snd, pad_y))
            builder.circle((x+delta_fst, pad_y), r, other_fill)
            builder.circle((x+delta_snd, pad_y), r, fill)
            builder.text((x+delta_snd+2*r, pad_y), str(gadget.angle), font_size=font_size)
        svg_code = repr(builder)
        if svg_code_only:
            return svg_code
        try:
            # pylint: disable = import-outside-toplevel
            from IPython.core.display import SVG # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'IPython' library.")
        return SVG(svg_code)

    def clone(self) -> "PhaseCircuit[AngleT]":
        """
            Produces an exact copy of this phase circuit.
        """
        return PhaseCircuit(self._num_qubits, tuple(self._iter_gadgets()))

    def conj_by_cx(self, ctrl: int, trgt: int) -> "PhaseCircuit[AngleT]":
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

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, PhaseCircuit):
            return NotImplemented
        if self.num_gadgets != other.num_gadgets:
            return NotImplemented
        if self.num_qubits != other.num_qubits:
            return False
        return all(g == h for g, h in zip(self._iter_gadgets(), other._iter_gadgets()))

    def __irshift__(self, gadgets: Union[PhaseGadget[AngleT],
                                         PhaseCircuit[AngleT],
                                         Sequence[PhaseGadget[AngleT]]]) -> "PhaseCircuit":
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

    def __rshift__(self, gadgets: Union[PhaseGadget[AngleT],
                                        PhaseCircuit[AngleT],
                                        Sequence[PhaseGadget[AngleT]]]) -> "PhaseCircuit":
        circ: PhaseCircuit[AngleT] = PhaseCircuit(self.num_qubits, [])
        circ >>= gadgets
        return circ

    def _iter_gadgets(self) -> Iterator[PhaseGadget]:
        next_idx = {"Z": 0, "X": 0}
        for i, angle in enumerate(self._angles):
            Z_next = next_idx["Z"]
            X_next = next_idx["X"]
            if Z_next < len(self._gadget_idxs["Z"]) and i == self._gadget_idxs["Z"][Z_next]:
                basis: Literal["Z", "X"] = "Z"
            elif X_next < len(self._gadget_idxs["X"]) and i == self._gadget_idxs["X"][X_next]:
                basis = "X"
            else:
                raise Exception("This should never happen. Please open an issue on GitHub.")
            col_idx = next_idx[basis]
            next_idx[basis] += 1
            col = self._matrix[basis][:, col_idx]
            yield PhaseGadget(basis, angle, {i for i, b in enumerate(col) if b % 2 == 1})

    def _cx_count(self, topology: Topology, cache: Dict[int, Dict[Tuple[int, ...], int]]) -> int:
        """
            Returns the CX count for an implementation of this phase gadget
            on the given topology based on minimum spanning trees (MST).
        """
        num_qubits = self._num_qubits
        weight = lambda u, v: 4*topology.dist(u, v)-2
        inf = 4*num_qubits-2
        count = 0
        for basis in ("Z", "X"):
            basis = cast(Literal["Z", "X"], basis)
            gadget_legs_cache = self._gadget_legs_cache[basis]
            for j, col in enumerate(self._matrix[basis].T):
                legs = gadget_legs_cache[j]
                if legs is None:
                    legs = tuple(int(i) for i in np.where(col == 1)[0])
                    gadget_legs_cache[j] = legs
                # legs = tuple(int(i) for i in np.where(col == 1)[0])
                num_legs = len(legs)
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

    def _repr_svg_(self):
        """
            Magic method for IPython/Jupyter pretty-printing.
            See https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
        """
        return self._to_svg(svg_code_only=True)

    @staticmethod
    def random(num_qubits: int, num_gadgets: int, *,
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
        rng = np.random.default_rng(seed=rng_seed)
        angle_rng_seed = int(rng.integers(65536))
        basis_idxs = rng.integers(2, size=num_gadgets)
        num_legs = rng.integers(min_legs, max_legs+1, size=num_gadgets)
        legs_list: list = [
            rng.choice(num_qubits, num_legs[i], replace=False) for i in range(num_gadgets)
        ]
        angle_rng = np.random.default_rng(seed=angle_rng_seed)
        angles = [int(x)*pi/angle_subdivision
                  for x in angle_rng.integers(1, 2*angle_subdivision, size=num_gadgets)]
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


class PhaseCircuitView(Generic[AngleT]):
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

    def to_svg(self, *,
               zcolor: str = "#CCFFCC",
               xcolor: str = "#FF8888",
               hscale: float = 1.0, vscale: float = 1.0,
               scale: float = 1.0,
               svg_code_only: bool = False
               ):
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
                                    svg_code_only=svg_code_only)

    def clone(self) -> PhaseCircuit[AngleT]:
        """
            Produces an exact copy of the phase circuit.
        """
        return self._circuit.clone()

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, PhaseCircuit):
            return self._circuit == other
        if isinstance(other, PhaseCircuitView):
            return self._circuit == other._circuit
        return NotImplemented

    def _repr_svg_(self):
        """
            Magic method for IPython/Jupyter pretty-printing.
            See https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
        """
        return self._circuit._repr_svg_() # pylint: disable = protected-access
