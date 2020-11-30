"""
    This module contains code to create circuits of mixed ZX phase gadgets.
"""

from math import ceil, log10
from typing import (Callable, cast, Collection, Dict, Final, FrozenSet, Generic, Iterator, List,
                    Literal, Optional, overload, Protocol, runtime_checkable, Sequence, Tuple,TypeVar,  Union)
import numpy as np # type: ignore
from pauliopt.topologies import Coupling, Topology, Matching
from pauliopt.utils import AngleT, AngleProtocol, Angle, SVGBuilder, Number





_WeightT = TypeVar("_WeightT", bound="_Weight")

class _Weight(Protocol[_WeightT]):

    def __add__(self, other: _WeightT) -> _WeightT:
        ...

    def __lt__(self, other: _WeightT) -> bool:
        ...

def _prims_algorithm(nodes: Collection[int], weight: Callable[[int, int], _WeightT],
                     zero: _WeightT, inf: _WeightT) -> _WeightT:
    """
        A modified version of Prim's algorithm that
        computes the weight of the minimum spanning tree connecting
        the given nodes, using the given weight function for branches.
        The number `inf` should be larger than the maximum weight
        that can be encountered in the process.
        The number `zero` should be `zero` for the given number type.
    """
    if not nodes:
        return zero
    mst_length: _WeightT = zero
    # Initialise set of nodes to visit:
    to_visit = set(nodes)
    n0 = next(iter(to_visit))
    to_visit.remove(n0)
    # Initialise dict of distances from visited set:
    dist_from_visited: Dict[int, _WeightT] = {
        n: weight(n0, n) for n in nodes
    }
    while to_visit:
        # Look for the node to be visited which is nearest to the visited set:
        nearest_node = 0 # dummy value
        nearest_dist: _WeightT = inf # dummy value
        for n in to_visit:
            n_dist: _WeightT = dist_from_visited[n]
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

def _prims_algorithm_debug(nodes: Collection[int], weight: Callable[[int, int], _WeightT],
                           zero: _WeightT, inf: _WeightT) -> Tuple[_Weight,
                                                                   Sequence[_Weight],
                                                                   Sequence[Tuple[int, int]]]:
    if not nodes:
        return zero, [], []
    mst_length: _WeightT = zero
    mst_branch_lengths = []
    mst_branches = []
    # Initialise set of nodes to visit:
    to_visit = set(nodes)
    n0 = next(iter(to_visit))
    to_visit.remove(n0)
    # Initialise dict of distances from visited set:
    dist_from_visited: Dict[int, _WeightT] = {
        n: weight(n0, n) for n in nodes
    }
    # Initialise possible edges for the MST:
    edge_from_visited: Dict[int, Tuple[int, int]] = {
        n: (n0, n) for n in nodes
    }
    while to_visit:
        # Look for the node to be visited which is nearest to the visited set:
        nearest_node = 0 # dummy value
        nearest_dist: _WeightT = inf # dummy value
        for n in to_visit:
            n_dist: _WeightT = dist_from_visited[n]
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
            raise TypeError("Qubits should be a collection of integers.")
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

    def mst_impl_cx_count(self, topology: Topology) -> int:
        """
            Returns the CX count for an implementation of this phase gadget
            on the given topology based on minimum spanning trees (MST).
        """
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        return  _prims_algorithm(self._qubits,
                                 lambda u, v: 4*topology.dist(u, v)-2,
                                 0, 4*len(topology.qubits)-2)

    def mst_impl_cx_circuit(self, topology: Topology) -> "CXCircuit":
        """
            CX circuit used to implement this gadget on the given topology
            using a minimum spanning tree (MST) technique.
            The implementation consists of this CX circuit, followed by
            a phase gate on any qubit, followed by the dagger of this
            CX circuit.
        """
        raise NotImplementedError()

    def print_mst_impl_info(self, topology: Topology):
        """
            Prints information about an implementation of this phase gadget
            on the given topology based on minimum spanning trees (MST).
        """
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        mst_length, mst_branch_lengths, mst_branches = \
            _prims_algorithm_debug(self._qubits,
                                   lambda u, v: 4*topology.dist(u, v)-2,
                                   0, 4*len(topology.qubits)-2)
        print(f"MST implementation info for {str(self)}:")
        print(f"  - Overall CX count for gadget: {mst_length}")
        print(f"  - MST branches: {mst_branches}")
        print(f"  - CX counts for MST branches: {mst_branch_lengths}")
        # TODO: add CX circuit info when `mst_impl` is implemented.
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

    _qubits: Tuple[int, ...]
    """
        The qubits spanned by this circuit, in increasing order.
    """

    _qubits_idxs: Dict[int, int]
    """
        Reverse directory, mapping qubits `q in self._qubits` to the
        corresponding row indices in the matrices `self._matrix[basis]`
        (same row index in both matrices).
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

    def __init__(self, qubits: Collection[int], gadgets: Sequence[PhaseGadget[AngleT]] = tuple()):
        if not isinstance(qubits, Collection) or not all(isinstance(q, int) for q in qubits):
            raise TypeError("Qubits should be a collection of integers.")
        if (not isinstance(gadgets, Sequence)
            or not all(isinstance(g, PhaseGadget) for g in gadgets)): # pylint: disable = C0330
            raise TypeError("Gadgets should be a sequence of PhaseGadget.")
        self._qubits = tuple(sorted(set(qubits)))
        # Create a reverse directory, mapping qubits to row indices in the matrices:
        self._qubits_idxs = {q: i for i, q in enumerate(self._qubits)}
        # Fills the lists of original indices and angles for the gadgets:
        self._gadget_idxs = {"Z": [], "X": []}
        self._angles = []
        for i, gadget in enumerate(gadgets):
            self._gadget_idxs[gadget.basis].append(i)
            self._angles.append(gadget.angle)
        num_qubits = len(self.qubits)
        self._matrix = {}
        for basis in cast(Sequence[Literal["Z", "X"]], ("Z", "X")):
            # Create a zero matrix for the basis:
            self._matrix[basis] = np.zeros(shape=(num_qubits, len(self._gadget_idxs[basis])),
                                           dtype=np.uint64)
            # Set matrix elements to 1 for all qubits spanned by the gadgets for the basis:
            for i, idx in enumerate(self._gadget_idxs[basis]):
                for q in gadgets[idx].qubits:
                    q_idx = self._qubits_idxs[q]
                    self._matrix[basis][q_idx, i] = 1

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
        new_col = np.zeros(shape=(len(self._qubits), 1), dtype=np.uint64)
        for q in gadget.qubits:
            new_col[self._qubits_idxs[q]] = 1
        self._matrix[basis] = np.append(self._matrix[basis], new_col, axis=1)
        self._gadget_idxs[basis].append(gadget_idx)
        self._angles.append(gadget.angle)
        return self

    @property
    def qubits(self) -> FrozenSet[int]:
        """
            Readonly property exposing the qubits spanned by this phase circuit.
        """
        return frozenset(self._qubits)

    @property
    def num_qubits(self) -> int:
        """
            Readonly property exposing the number of qubits spanned by this phase circuit.
        """
        return len(self._qubits)

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
        vscale *= scale
        hscale *= scale
        gadgets = self.gadgets
        qubits = sorted(self.qubits)
        num_digits = int(ceil(log10(len(qubits))))
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
        height = 2*pad_y + line_height*(len(qubits)+1)
        builder = SVGBuilder(width, height)
        for i, q in enumerate(qubits):
            y = pad_y + (i+1) * line_height
            builder.line((pad_x, y), (width-pad_x, y))
            builder.text((0, y), f"{str(q):>{num_digits}}", font_size=font_size)
            builder.text((width-pad_x+r, y), f"{str(q):>{num_digits}}", font_size=font_size)
        for row, gadget in enumerate(gadgets):
            fill = zcolor if gadget.basis == "Z" else xcolor
            other_fill = xcolor if gadget.basis == "Z" else zcolor
            x = pad_x + margin_x + row * row_width
            for q in gadget.qubits:
                y = pad_y + (qubits.index(q)+1)*line_height
                builder.line((x, y), (x+delta_fst, pad_y))
            for q in gadget.qubits:
                y = pad_y + (qubits.index(q)+1)*line_height
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
        return PhaseCircuit(self.qubits, tuple(self._iter_gadgets()))

    def conj_by_cx(self, ctrl: int, trgt: int) -> "PhaseCircuit[AngleT]":
        """
            Conjugates this circuit by a CX gate with given control/target.
            The circuit is modified in-place and then returned, as per the
            [fluent interface pattern](https://en.wikipedia.org/wiki/Fluent_interface).
        """
        qubits = self.qubits
        if ctrl not in qubits:
            raise ValueError(f"Invalid control qubit {ctrl}.")
        if trgt not in qubits:
            raise ValueError(f"Invalid target qubit {trgt}.")
        self._matrix["Z"][ctrl, :] = (self._matrix["Z"][ctrl, :] + self._matrix["Z"][trgt, :]) % 2
        self._matrix["X"][trgt, :] = (self._matrix["X"][trgt, :] + self._matrix["X"][ctrl, :]) % 2
        return self

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, PhaseCircuit):
            return NotImplemented
        if self.num_gadgets != other.num_gadgets:
            return NotImplemented
        if self.qubits != other.qubits:
            return False
        return all(g == h for g, h in zip(self._iter_gadgets(), other._iter_gadgets()))

    def __irshift__(self, gadgets: Union[PhaseGadget, Sequence[PhaseGadget]]) -> "PhaseCircuit":
        if isinstance(gadgets, PhaseGadget):
            gadgets = [gadgets]
        if (not isinstance(gadgets, Sequence)
                or not all(isinstance(gadget, PhaseGadget) for gadget in gadgets)):
            raise TypeError(f"Expected phase gadget or sequence of phase gadgets, found {gadgets}.")
        for gadget in gadgets:
            self.add_gadget(gadget)
        return self

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

    def _repr_svg_(self):
        """
            Magic method for IPython/Jupyter pretty-printing.
            See https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
        """
        return self.to_svg(svg_code_only=True)

    @staticmethod
    def random(qubits: Collection[int], num_gadgets: int, *,
               angle_subdivision: int = 4,
               min_legs: int = 1,
               max_legs: Optional[int] = None,
               rng_seed: Optional[int] = None) -> "PhaseCircuit":
        """
            Generates a random circuit of mixed ZX phase gadgets on the given collection of qubits,
            with the given number of gadgets.

            The optional argument `angle_subdivision` (default: 4) can be used to specify the
            denominator in the random fractional multiples of pi used as values for the angles.

            The optional arguments `min_legs` (default: 1, minimum: 1) and `max_legs`
            (default: `None`, minimum `min_legs`) can be used to specify the minimum and maximum
            number of legs for the phase gadgets. If `None`, `max_legs` is set to `len(qubits)`.

            The optional argument `rng_seed` (default: `None`) is used as seed for the RNG.
        """
        if not isinstance(qubits, Collection) or not all(isinstance(q, int) for q in qubits):
            raise TypeError("Qubits should be a collection of integers.")
        qubits = list(qubits)
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
            max_legs = len(qubits)
        if rng_seed is not None and not isinstance(rng_seed, int):
            raise TypeError("RNG seed must be integer or 'None'.")
        rng = np.random.default_rng(seed=rng_seed)
        angle_rng_seed = int(rng.integers(65536))
        basis_idxs = rng.integers(2, size=num_gadgets)
        num_legs = rng.integers(min_legs, max_legs+1, size=num_gadgets)
        legs_idxs_list: list = [
            rng.choice(len(qubits), num_legs[i], replace=False) for i in range(num_gadgets)
        ]
        _angles = Angle.random(angle_subdivision, size=num_gadgets, rng_seed=angle_rng_seed)
        if isinstance(_angles, Angle):
            angles: Sequence[Angle] = [_angles]
        else:
            angles = _angles
        bases = cast(Sequence[Literal["Z", "X"]], ("Z", "X"))
        gadgets: List[PhaseGadget] = [
            PhaseGadget(bases[(basis_idx+i)%2],
                        angle,
                        [qubits[leg] for leg in legs_idxs])
            for i, (basis_idx, angle, legs_idxs) in enumerate(zip(basis_idxs,
                                                                  angles,
                                                                  legs_idxs_list))
        ]
        return PhaseCircuit(qubits, gadgets)


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
    def qubits(self) -> FrozenSet[int]:
        """
            Readonly property exposing the qubits spanned by the phase circuit.
        """
        return self._circuit.qubits

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

GateLike = Union[List[int], Tuple[int, int]]

class CXCircuitLayer:
    """
        Container for a layer of CX gates constrained
        by a given qubit topology.

        It uses `pauliopt.topologies.Matching` to keep track of which
        couplings in the qubit topology are currently occupied by a CX gate,
        and to efficiently determine whether a CX gate can be added to the layer.
    """

    _topology: Topology
    _gates: Dict[Coupling, Tuple[int, int]]
    _matching: Matching

    def __init__(self, topology: Topology,
                 gates: Sequence[GateLike] = tuple()):
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        if not isinstance(gates, Sequence):
            raise TypeError(f"Expected sequence of ordered pairs, found {gates}")
        self._topology = topology
        self._gates = {}
        self._matching = Matching(topology)
        for gate in gates:
            if not isinstance(gate, (list, tuple)) or len(gate) != 2 or len(set(gate)) != 2:
                raise TypeError(f"Expected gates to be pairs of distinct integers, found {gate}.")
            ctrl, trgt = gate
            self.flip_cx(ctrl, trgt)

    @property
    def topology(self) -> Topology:
        """
            Readonly property exposing the qubit topology
            constraining this CX circuit layer.
        """
        return self._topology

    @property
    def num_gates(self) -> int:
        """
            Readonly property returning the number of gates in this
            CX circuit layer.
        """
        return len(self._gates)

    @property
    def gates(self) -> FrozenSet[Tuple[int, int]]:
        """
            Readonly property returning the collection of gates in this
            CX circuit layer.

            This collection is freshly generated at every call.
        """
        return frozenset(self._gates.values())

    @property
    def flippable_cxs(self) -> FrozenSet[Tuple[int, int]]:
        """
            Readonly property returning the collection of CX gates that
            that can be currently flipped in this layer, namely:

            - all gates currently in the layer (will be removed by flip);
            - all gates with both qubits currently not covered by a gate
              already in the layer (will be added by flip).

            This collection is freshly generated at every call.
        """
        return frozenset(self._iter_flippable_cxs())

    def incident(self, qubit: int) -> Optional[Tuple[int, int]]:
        """
            Returns the CX gate incident to the given qubit in this layer,
            or `None` if there is no gate incident to the qubit.
        """
        incident_coupling = self._matching.incident(qubit)
        if incident_coupling is None:
            return None
        return self._gates[incident_coupling]

    def has_cx(self, ctrl: int, trgt: int) -> bool:
        """
            Checks whether the given CX gate is in the layer:
        """
        gate = (ctrl, trgt)
        coupling = Coupling(ctrl, trgt)
        return self._gates.get(coupling, None) == gate

    def is_cx_flippable(self, ctrl: int, trgt: int) -> bool:
        """
            Checks whether the given CX gate can be flipped in this layer.
            This is true if:

            - the gate is present (gate can be removed);
            - the gate is not present, and neither the control nor the
              target are already covered by some other gate (gate can be added).
        """
        gate = (ctrl, trgt)
        coupling = Coupling(ctrl, trgt)
        if coupling in self._gates:
            return self._gates[coupling] == gate
        if self.incident(ctrl) is not None:
            return False
        if self.incident(trgt) is not None:
            return False
        return True

    def flip_cx(self, ctrl: int, trgt: int) -> "CXCircuitLayer":
        """
            Adds/removes a CX gate with given control and target to/from the layer.
            Raises `ValueError` if the gate cannot be added/removed.

            The layer is modified in-place and then returned, as per the
            [fluent API pattern](https://en.wikipedia.org/wiki/Fluent_interface).
        """
        if not isinstance(ctrl, int):
            raise TypeError(f"Expected integer, found {ctrl}.")
        if not isinstance(trgt, int):
            raise TypeError(f"Expected integer, found {trgt}.")
        gate = (ctrl, trgt)
        if not self._matching.is_flippable(gate):
            raise ValueError(f"Cannot add CX gate {gate} to the layer: "
                             f"gate is not present, but one of control or target "
                             f"is already involved in some other gate.")
        coupling = Coupling(ctrl, trgt)
        if coupling in self._gates:
            if self._gates[coupling] == gate:
                # CX gate already there, gets removed from the layer:
                del self._gates[coupling]
                self._matching.flip(coupling)
                return self
            raise ValueError(f"Invalid CX gate {gate} for given topology: another gate "
                             f"{self._gates[coupling]} already exists for this qubit pair.")
        # CX gate gets added to the layer:
        self._gates[coupling] = (ctrl, trgt)
        self._matching.flip(coupling)
        return self

    def draw(self, layout: str = "kamada_kawai", *,
             figsize: Optional[Tuple[int, int]] = None,
             zcolor: str = "#CCFFCC",
             xcolor: str = "#FF8888",
             **kwargs):
        # pylint: disable = too-many-locals
        """
            Draws this CX circuit layer using NetworkX and Matplotlib.

            The `layout` keyword argument can be used to select a NetworkX layout
            from the available ones (exposed by `Topology.available_nx_layouts`).
            The `figsize` keyword argument is passed to `matplotlib.pyplot.figure`:
            if specified, it determines the width and height of the figure being drawn.
            The `zcolor` and `xcolor` keyword arguments are used to determine the colour
            of the Z and X dots in a CX gate (analogous to `PhaseCircuit.to_svg`).
            Keyword arguments `kwargs` are those of `networkx.draw_networkx`.
        """
        try:
            # pylint: disable = import-outside-toplevel
            import networkx as nx # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'networkx' library.")
        try:
            # pylint: disable = import-outside-toplevel
            import matplotlib.pyplot as plt # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'matplotlib' library.")
        G = self.topology.to_nx
        nodes = sorted(self.topology.qubits)
        edges = sorted(self.topology.couplings, key=lambda c: c.as_pair)
        kwargs = {**kwargs}
        layouts = self.topology.available_nx_layouts
        if "pos" not in kwargs:
            if layout not in layouts:
                raise ValueError(f"Invalid layout found: {layout}. "
                                 f"Valid layouts: {', '.join(repr(l) for l in layouts)}")
            kwargs["pos"] = getattr(nx, layout+"_layout")(G)
        if "node_color" not in kwargs:
            kwargs["node_color"] = ["#dddddd" for _ in nodes]
        for node_idx, node in enumerate(nodes):
            gate = self.incident(node)
            if gate is not None:
                ctrl, trgt = gate
                if node == ctrl:
                    kwargs["node_color"][node_idx] = zcolor
                else:
                    kwargs["node_color"][node_idx] = xcolor
        if "edge_color" not in kwargs:
            kwargs["edge_color"] = ["#dddddd" for _ in edges]
        for ctrl, trgt in self.gates:
            edge_idx = edges.index(Coupling(ctrl, trgt))
            kwargs["edge_color"][edge_idx] = "#000000"
        plt.figure(figsize=figsize)
        nx.draw_networkx(G, **kwargs)
        plt.show()

    def __irshift__(self, gates: Union[GateLike, Sequence[GateLike]]) -> "CXCircuitLayer":
        if (isinstance(gates, (list, tuple))
                and all(isinstance(x, int) for x in gates) and len(gates) == 2):
            gates = [cast(Union[List[int], Tuple[int, int]], gates)]
        if not isinstance(gates, Sequence):
            raise TypeError(f"Expected sequence of gates, found {gates}")
        for gate in gates:
            if not isinstance(gate, Sequence) or len(gate) != 2:
                raise TypeError(f"Expected ordered pair, found {gate}")
            ctrl, trgt = gate
            self.flip_cx(ctrl, trgt)
        return self

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, CXCircuitLayer):
            return NotImplemented
        if self.topology != other.topology:
            return False
        return self.gates == other.gates

    def _iter_flippable_cxs(self) -> Iterator[Tuple[int, int]]:
        """
            Iterates over all CX gates that can be flipped.
            The layer must not be changed while the gates are being iterated,
            reason why this is a private method.

            The private method `pauliopt.topology.Matching._iter_flippable_couplings`
            can be called safely because `self._matching` is only accessible internally
            to the layer and cannot be accidentally modified (subject to the layer not
            being modified).
        """
        # pylint: disable = protected-access
        for coupling in self._matching._iter_flippable_couplings():
            if coupling in self._gates:
                yield self._gates[coupling]
            else:
                fst, snd = coupling
                yield (fst, snd)
                yield (snd, fst)

CXCircuitLayerLike = Union[CXCircuitLayer, Sequence[GateLike]]

class CXCircuit(Sequence[CXCircuitLayer]):
    """
        Container for a circuit of CX gates, consisting of a given number of layers
        and constrained by a given qubit topology.
    """

    _topology: Topology
    _layers: List[CXCircuitLayer]

    def __init__(self, topology: Topology, layers: Sequence[CXCircuitLayer] = tuple()):
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        if not isinstance(layers, Sequence):
            raise TypeError(f"Expected sequence of CXCircuitLayer, found {layers}")
        for layer in layers:
            if not isinstance(layer, CXCircuitLayer):
                raise TypeError(f"Expected CXCircuitLayer, found {type(layer)}")
            if layer.topology != topology:
                raise ValueError("Layer topology different from circuit topology.")
        self._topology = topology
        self._layers = list(layers)

    @property
    def topology(self) -> Topology:
        """
            Readonly property exposing the qubit topology
            constraining this CX circuit.
        """
        return self._topology

    @property
    def num_gates(self) -> int:
        """
            Readonly property returning the total number of gates in this
            CX circuit.
        """
        return sum((layer.num_gates for layer in self), 0)

    def draw(self, layout: str = "kamada_kawai", *,
             figsize: Optional[Tuple[int, int]] = None,
             zcolor: str = "#CCFFCC",
             xcolor: str = "#FF8888",
             **kwargs):
        """
            Draws this CX circuit using NetworkX and Matplotlib.
            Keyword arguments `kwargs` are those of `networkx.draw_networkx`.
        """
        for layer_idx, layer in enumerate(self):
            print(f"Layer {layer_idx}:")
            layer.draw(layout=layout, figsize=figsize,
                       zcolor=zcolor, xcolor=xcolor, **kwargs)

    @overload
    def __getitem__(self, layer_idx: int) -> CXCircuitLayer:
        ...

    @overload
    def __getitem__(self, layer_idx: slice) -> Sequence[CXCircuitLayer]:
        ...

    def __getitem__(self, layer_idx):
        return self._layers[layer_idx]

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self) -> Iterator[CXCircuitLayer]:
        return iter(self._layers)

    def __irshift__(self, layers: Union[CXCircuitLayerLike,
                                        Sequence[CXCircuitLayerLike]]) -> "CXCircuit":
        if isinstance(layers, CXCircuitLayer):
            layers = [layers]
        elif (isinstance(layers, Sequence)
              and all(isinstance(g, (list, tuple))
                      and all(isinstance(x, int) for x in g) # pylint: disable = C0330
                      and len(g) == 2 for g in layers)): # pylint: disable = C0330
            layers = [cast(Sequence[GateLike], layers)]
        if not isinstance(layers, Sequence):
            raise TypeError(f"Expected sequence of layers, found {layers}")
        for layer in layers:
            if not isinstance(layer, CXCircuitLayer):
                if not isinstance(layer, Sequence):
                    raise TypeError(f"Expected a sequence of pairs of ints, found {layer}")
                layer = CXCircuitLayer(self.topology, cast(Sequence[GateLike], layer))
            self._layers.append(layer)
        return self

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, CXCircuit):
            return NotImplemented
        if self.topology != other.topology:
            return False
        if len(self) != len(other):
            return False
        return all(sl == ol for sl, ol in zip(self, other))


class CXCircuitLayerView():
    """
        Readonly view on a CX circuit layer.
    """

    _layer: CXCircuitLayer

    def __init__(self, layer: CXCircuitLayer):
        if not isinstance(layer, CXCircuitLayer):
            raise TypeError(f"Expected CXCircuitLayer, found {type(layer)}.")
        self._layer = layer

    @property
    def topology(self) -> Topology:
        """
            Readonly property exposing the qubit topology
            constraining this CX circuit layer.
        """
        return self._layer.topology

    @property
    def gates(self) -> FrozenSet[Tuple[int, int]]:
        """
            Readonly property returning the collection of gates in this
            CX circuit layer.

            This collection is freshly generated at every call.
        """
        return self._layer.gates

    @property
    def num_gates(self) -> int:
        """
            Readonly property returning the number of gates in this
            CX circuit layer.
        """
        return self._layer.num_gates

    @property
    def flippable_cxs(self) -> FrozenSet[Tuple[int, int]]:
        """
            Readonly property returning the collection of CX gates that
            that can be currently flipped in this layer, namely:

            - all gates currently in the layer (will be removed by flip);
            - all gates with both qubits currently not covered by a gate
              already in the layer (will be added by flip).

            This collection is freshly generated at every call.
        """
        return self._layer.flippable_cxs

    def incident(self, qubit: int) -> Optional[Tuple[int, int]]:
        """
            Returns the CX gate incident to the given qubit in this layer,
            or `None` if there is no gate incident to the qubit.
        """
        return self._layer.incident(qubit)

    def has_cx(self, ctrl: int, trgt: int) -> bool:
        """
            Checks whether the given CX gate is in the layer:
        """
        return self._layer.has_cx(ctrl, trgt)

    def is_cx_flippable(self, ctrl: int, trgt: int) -> bool:
        """
            Checks whether the given CX gate can be flipped in this layer.
            This is true if:

            - the gate is present (gate can be removed);
            - the gate is not present, and neither the control nor the
              target are already covered by some other gate (gate can be added).
        """
        return self._layer.is_cx_flippable(ctrl, trgt)

    def draw(self, layout: str = "kamada_kawai", *,
             figsize: Optional[Tuple[int, int]] = None,
             zcolor: str = "#CCFFCC",
             xcolor: str = "#FF8888",
             **kwargs):
        """
            Draws this CX circuit layer using NetworkX and Matplotlib.
            Keyword arguments `kwargs` are those of `networkx.draw_networkx`.
        """
        return self._layer.draw(layout=layout, figsize=figsize,
                                zcolor=zcolor, xcolor=xcolor, **kwargs)

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, CXCircuitLayer):
            return self._layer == other
        if isinstance(other, CXCircuitLayerView):
            return self._layer == other._layer
        return NotImplemented


class CXCircuitView(Sequence[CXCircuitLayerView]):
    """
        Readonly view on a CX circuit.
    """

    _circuit: CXCircuit

    def __init__(self, circuit: CXCircuit):
        if not isinstance(circuit, CXCircuit):
            raise TypeError(f"Expected CXCircuit, found {type(circuit)}.")
        self._circuit = circuit

    @property
    def topology(self) -> Topology:
        """
            Readonly property exposing the qubit topology
            constraining this CX circuit.
        """
        return self._circuit.topology


    @property
    def num_gates(self) -> int:
        """
            Readonly property returning the number of gates in this
            CX circuit.
        """
        return self._circuit.num_gates

    def draw(self, layout: str = "kamada_kawai", *,
             figsize: Optional[Tuple[int, int]] = None,
             zcolor: str = "#CCFFCC",
             xcolor: str = "#FF8888",
             **kwargs):
        """
            Draws this CX circuit using NetworkX and Matplotlib.
            Keyword arguments `kwargs` are those of `networkx.draw_networkx`.
        """
        return self._circuit.draw(layout=layout, figsize=figsize,
                                  zcolor=zcolor, xcolor=xcolor, **kwargs)

    @overload
    def __getitem__(self, layer_idx: int) -> CXCircuitLayerView:
        ...

    @overload
    def __getitem__(self, layer_idx: slice) -> Sequence[CXCircuitLayerView]:
        ...

    def __getitem__(self, layer_idx):
        return CXCircuitLayerView(self._circuit[layer_idx])

    def __len__(self) -> int:
        return len(self._circuit)

    def __iter__(self) -> Iterator[CXCircuitLayerView]:
        return (CXCircuitLayerView(l) for l in self._circuit)

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if isinstance(other, CXCircuit):
            return self._circuit == other
        if isinstance(other, CXCircuitView):
            return self._circuit == other._circuit
        return NotImplemented
