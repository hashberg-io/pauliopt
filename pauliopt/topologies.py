"""
    This module contains utility code to deal with qubit topologies.
"""

from typing import (Collection, Dict, Final, FrozenSet, Iterator,
                    List, Mapping, Optional, Set, Tuple, TypedDict)


class Coupling(FrozenSet[int]):
    """
        Type for couplings in a qubit topology, i.e. unordered
        pairs of adjacent qubits.
    """
    def __new__(cls, fst: int, snd: int):
        if not isinstance(fst, int):
            raise TypeError(f"Expected integer, found {fst}")
        if not isinstance(snd, int):
            raise TypeError(f"Expected integer, found {snd}")
        if len({fst, snd}) != 2:
            raise ValueError("Expected a pair of distinct qubits.")
        # see https://github.com/python/mypy/issues/6061
        return super(Coupling, cls).__new__(cls, {fst, snd}) # type: ignore

    @property
    def as_pair(self) -> Tuple[int, int]:
        """
            Returns the coupling as a (increasingly) ordered pair.
        """
        return (min(*self), max(*self)) # pylint: disable = not-an-iterable

    def __str__(self) -> str:
        fst, snd = sorted(self)
        return f"{{{fst}, {snd}}}"

    def __repr__(self) -> str:
        fst, snd = sorted(self)
        return f"Coupling({fst}, {snd})"


CouplingLike = Collection[int]
"""
    Type alias for things that could be used to specify couplings,
    namely any collection of `int` (subject to additional restrictions).

    In an ideal world, this should be "int collections of len 2",
    but static typing does not yet allow for such a constraint.
"""


def _validate_coupling(qubits: FrozenSet[int], coupling_like: CouplingLike) -> Coupling:
    # Assumes `qubits` was already validated.
    if not isinstance(coupling_like, Collection) or len(coupling_like) != 2:
        raise TypeError(f"Expected a pair, found {coupling_like}.")
    fst, snd = coupling_like
    if fst not in qubits:
        raise TypeError(f"Invalid qubit {fst}.")
    if snd not in qubits:
        raise TypeError(f"Invalid qubit {snd}.")
    return Coupling(fst, snd)


class TopologyDict(TypedDict, total=True):
    """
        The type of the dictionary returned by `Topology.as_dict`,
        suitable for JSON serialization.
    """

    qubits: List[int]
    """
        Property exposing the qubits in the topology.
    """

    couplings: List[List[int]]
    """
        Property exposing the couplings between qubits in the topology.
    """


Layouts: Final[Tuple[str, ...]] = ("circular", "kamada_kawai", "random",
                                   "shell", "spring", "spectral", "spiral")



def floyd_warshall(topology: "Topology", *,
                   enforce_connected: bool = False) -> Mapping[Tuple[int, int], Optional[int]]:
    """
        Runs the Floyd–Warshall to compute a dictionary of distances between qubits in a given
        qubit topology. If two qubits are not connected, distance is `None`.
        If two qubits are not connected and `enforce_connected` is `True`, `ValueError` is raised.
    """
    if not isinstance(topology, Topology):
        raise TypeError(f"Expected Topology, found {type(topology)}.")
    if not isinstance(enforce_connected, bool):
        raise TypeError(f"Expected boolean, found {enforce_connected}.")
    inf = len(topology.qubits) # a number surely larger than max dist in topology
    def init_dist(u, v):
        if u == v:
            return 0
        coupling = Coupling(u, v)
        if coupling in topology.couplings:
            return 1
        return inf
    dist = {
        (u, v): init_dist(u, v)
        for u in topology.qubits for v in topology.qubits
    }
    for w in topology.qubits:
        for u in topology.qubits:
            for v in topology.qubits:
                upper_bound = dist[(u, w)] + dist[(w, v)]
                if dist[(u, v)] > upper_bound:
                    dist[(u, v)] = upper_bound
    for u in topology.qubits:
        for v in topology.qubits:
            if dist[(u, v)] == inf:
                if enforce_connected:
                    raise ValueError("Topology is not connected.")
                dist[(u, v)] = None
    return dist

class Topology:
    """
        Container class for a qubit topology.
    """
    _qubits: FrozenSet[int]
    _couplings: FrozenSet[Coupling]
    _adjacent: Mapping[int, FrozenSet[int]]
    _dist: Mapping[Tuple[int, int], Optional[int]]

    def __init__(self, qubits: Collection[int], couplings: Collection[CouplingLike]):
        if not isinstance(qubits, Collection) or not all(isinstance(q, int) for q in qubits):
            raise TypeError("Qubits must be a collection of integers.")
        if not qubits:
            raise ValueError("Qubits must be a non-empty collection.")
        if not isinstance(couplings, Collection):
            raise TypeError("Couplings must be a collection of pairs of qubits.")
        self._qubits = frozenset(qubits)
        self._couplings = frozenset(_validate_coupling(self._qubits, c) for c in couplings)
        _adjacent: Dict[int, Set[int]] = {}
        for q in self.qubits:
            _adjacent[q] = set()
        for fst, snd in self.couplings:
            _adjacent[fst].add(snd)
            _adjacent[snd].add(fst)
        self._adjacent = {q: frozenset(n) for q, n in _adjacent.items()}
        self._dist = floyd_warshall(self)

    @property
    def qubits(self) -> FrozenSet[int]:
        """
            Readonly property exposing the qubits in this topology.
        """
        return self._qubits

    @property
    def couplings(self) -> FrozenSet[Coupling]:
        """
            Readonly property exposing the couplings between qubits in this topology.
        """
        return self._couplings

    @property
    def as_dict(self) -> TopologyDict:
        """
            Readonly property returning this topology as
            a dictionary, for serialization purposes.
        """
        return {
            "qubits": sorted(self.qubits),
            "couplings": sorted(list(c.as_pair) for c in self.couplings)
        }

    @property
    def is_planar(self) -> bool:
        """
            Whether this qubit topology is a planar graph.
        """
        try:
            # pylint: disable = import-outside-toplevel
            import networkx as nx # type: ignore
        except ModuleNotFoundError as e: # pylint: disable = unused-variable
            raise ModuleNotFoundError("You must install the 'networkx' library.")
        G = self.to_nx
        is_planar, _ = nx.check_planarity(G)
        return is_planar

    @property
    def available_nx_layouts(self) -> Tuple[str, ...]:
        """
            Readonly property returning the available layouts for this qubit topology.
        """
        if self.is_planar:
            return Layouts+("planar",)
        return Layouts

    @property
    def to_nx(self):
        """
            Readonly property returning a NetworkX graph version of this topology.
            Requires the 'networkx' library to work.
        """
        try:
            # pylint: disable = import-outside-toplevel
            import networkx as nx # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'networkx' library.")
        g = nx.Graph()
        g.add_nodes_from(sorted(self.qubits))
        g.add_edges_from(sorted(self.couplings, key=lambda c: c.as_pair))
        return g

    def draw(self, layout: str = "kamada_kawai", *,
             figsize: Optional[Tuple[int, int]] = None,
             **kwargs):
        """
            Draws this qubit topology using NetworkX and Matplotlib.

            The `layout` keyword argument can be used to select a NetworkX layout
            from the available ones (exposed by `Topology.available_nx_layouts`).
            The `figsize` keyword argument is passed to `matplotlib.pyplot.figure`:
            if specified, it determines the width and height of the figure being drawn.
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
        G = self.to_nx
        kwargs = {**kwargs}
        layouts = self.available_nx_layouts
        if "pos" not in kwargs:
            if layout not in layouts:
                raise ValueError(f"Invalid layout found: {layout}. "
                                 f"Valid layouts: {', '.join(repr(l) for l in layouts)}")
            kwargs["pos"] = getattr(nx, layout+"_layout")(G)
        if "node_color" not in kwargs:
            kwargs["node_color"] = "#dddddd"
        plt.figure(figsize=figsize)
        nx.draw_networkx(G, **kwargs)
        plt.show()

    def adjacent(self, qubit: int) -> FrozenSet[int]:
        """
            Readonly property exposing the (frozen) set of qubits adjacent
            to (i.e. couple with) the given qubit.
        """
        if not isinstance(qubit, int):
            raise TypeError("Qubit should be an integer.")
        if qubit not in self._adjacent:
            raise ValueError(f"Invalid qubit {qubit}.")
        return self._adjacent[qubit]

    def incident(self, qubit: int) -> Iterator[Coupling]:
        """
            Readonly property returning an iterator running over all couplings
            incident onto the given qubit.

            This is returned as an iterator, rather than a collection,
            because the couplings are generated on the fly (i.e. this is not
            merely exposing some internal collection).
        """
        adjacent = self.adjacent(qubit)
        return (Coupling(qubit, q) for q in adjacent)

    def dist(self, fro: int, to):
        """
            Readonly property returning an iterator running over all couplings
            incident onto the given qubit.

            This is returned as an iterator, rather than a collection,
            because the couplings are generated on the fly (i.e. this is not
            merely exposing some internal collection).
        """
        if not isinstance(fro, int) or fro not in self.qubits:
            raise TypeError(f"Expected a valid qubit, found {fro}.")
        if not isinstance(to, int) or to not in self.qubits:
            raise TypeError(f"Expected a valid qubit, found {to}.")
        return self._dist[(fro, to)]

    def __repr__(self) -> str:
        if self.couplings:
            return (f"Topology({set(self.qubits)}, "
                    f"[{', '.join(str(c) for c in self.couplings)}])")
        return f"Topology({set(self.qubits)})"

    def __hash__(self) -> int:
        return hash((self.qubits, self.couplings))

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, Topology):
            return NotImplemented
        if self.qubits != other.qubits:
            return False
        if self.couplings != other.couplings:
            return False
        return True

    @staticmethod
    def from_dict(topology: TopologyDict) -> "Topology":
        """
            Creates a `Topology` instance from a dictionary in the
            format obtained from `Topology.as_dict`,
            for de-serialization purposes.
        """
        if "qubits" not in topology:
            raise TypeError("Expected key 'qubits'.")
        if "couplings" not in topology:
            raise TypeError("Expected key 'couplings'.")
        return Topology(topology["qubits"], topology["couplings"])

    @staticmethod
    def line(num_qubits: int) -> "Topology":
        """
            Creates a line topology on the given number of qubits.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise TypeError("Number of qubits must be positive integer.")
        qubits = range(num_qubits)
        couplings = [[i, i+1] for i in range(num_qubits-1)]
        return Topology(qubits, couplings)

    @staticmethod
    def cycle(num_qubits: int) -> "Topology":
        """
            Creates a cycle topology on the given number of qubits.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise TypeError("Number of qubits must be positive integer.")
        qubits = range(num_qubits)
        couplings = [[i, (i+1)%num_qubits] for i in range(num_qubits)]
        return Topology(qubits, couplings)

    @staticmethod
    def complete(num_qubits: int) -> "Topology":
        """
            Creates a complete topology on the given number of qubits.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise TypeError("Number of qubits must be positive integer.")
        qubits = range(num_qubits)
        couplings = [[i, j] for i in range(num_qubits) for j in range(i+1, num_qubits)]
        return Topology(qubits, couplings)

    @staticmethod
    def grid(num_rows: int, num_cols: int) -> "Topology":
        """
            Creates a grid topology with the given number of rows and cols.
            Qubits are indexed by rows.
        """
        if not isinstance(num_rows, int) or num_rows <= 0:
            raise TypeError("Number of rows must be positive integer.")
        if not isinstance(num_cols, int) or num_cols <= 0:
            raise TypeError("Number of cols must be positive integer.")
        qubits = range(num_rows * num_cols)
        def qubit(r, c):
            return num_cols*r + c
        couplings: List[List[int]] = []
        for r in range(num_rows):
            for c in range(num_cols):
                if r < num_rows-1:
                    couplings.append([qubit(r, c), qubit(r+1, c)])
                if c < num_cols-1:
                    couplings.append([qubit(r, c), qubit(r, c+1)])
        return Topology(qubits, couplings)

    @staticmethod
    def periodic_grid(num_rows: int, num_cols: int) -> "Topology":
        """
            Creates a periodic grid topology with the given number of rows and cols.
            Qubits are indexed by rows.
        """
        if not isinstance(num_rows, int) or num_rows <= 0:
            raise TypeError("Number of rows must be positive integer.")
        if not isinstance(num_cols, int) or num_cols <= 0:
            raise TypeError("Number of cols must be positive integer.")
        qubits = range(num_rows * num_cols)
        def qubit(r, c):
            return num_cols*r + c
        couplings: List[List[int]] = []
        for r in range(num_rows):
            for c in range(num_cols):
                couplings.append([qubit(r, c), qubit((r+1)%num_rows, c)])
                couplings.append([qubit(r, c), qubit(r, (c+1)%num_cols)])
        return Topology(qubits, couplings)

    @staticmethod
    def from_qiskit_config(config) -> "Topology":
        """
            Static method to construct the topology from a
            Qiskit backend configuration.

            This method relies on the `qiskit` library being available.
            Specifically, the `config` argument must be of type
            `qiskit.providers.models.QasmBackendConfiguration`.
        """
        try:
            # pylint: disable = import-outside-toplevel
            from qiskit.providers.models import QasmBackendConfiguration # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'qiskit' library.")
        if not isinstance(config, QasmBackendConfiguration):
            raise TypeError("Argument backend must be of type "
                            "`qiskit.providers.models.QasmBackendConfiguration`, "
                            f"found {type(config)}.")
        config_dict = config.to_dict()
        n_qubits: int = config_dict["n_qubits"]
        coupling_map: List[List[int]] = config_dict["coupling_map"]
        return Topology(range(n_qubits), coupling_map)

    @staticmethod
    def from_qiskit_backend(backend) -> "Topology":
        """
            Static method to construct the topology from a Qiskit backend.

            This method relies on the `qiskit` library being available.
            Specifically, the `backend` argument must be of type
            `qiskit.providers.BaseBackend`.
        """
        try:
            # pylint: disable = import-outside-toplevel, unused-import
            from qiskit.providers import BaseBackend # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'qiskit' library.")
        if not isinstance(backend, BaseBackend):
            raise TypeError("Argument backend must be of type "
                            "`qiskit.providers.BaseBackend`.")
        return Topology.from_qiskit_config(backend.configuration())


class Matching:
    """
        Mutable container class for a matching on a qubit topology.
    """

    _topology: Topology
    _matched_couplings: Set[Coupling]
    _matched_qubits: Set[int]
    _incident_coupling: Dict[int, Coupling]

    def __init__(self, topology: Topology):
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {topology}.")
        self._topology = topology
        self._matched_couplings = set()
        self._matched_qubits = set()
        self._incident_coupling = {}

    @property
    def topology(self) -> Topology:
        """
            Readonly property exposing the qubit topology
            underlying this matching.
        """
        return self._topology

    @property
    def matched_couplings(self) -> FrozenSet[Coupling]:
        """
            Readonly property returning the collection of couplings
            currently in this matching.

            This collection is freshly generated at every call.
        """
        return frozenset(self._matched_couplings)

    @property
    def matched_qubits(self) -> FrozenSet[int]:
        """
            Readonly property returning the collection of qubits
            currently matched in this matching.

            This collection is freshly generated at every call.
        """
        return frozenset(self._matched_qubits)

    @property
    def flippable_couplings(self) -> FrozenSet[Coupling]:
        """
            Readonly property returning the collection of couplings
            that can be currently flipped in this matching, namely:

            - all couplings currently in the matching (will be removed by flip);
            - all couplings with both qubits currently not matched by the matching
              (will be added by flip).

            This collection is freshly generated at every call.
        """
        return frozenset(self._iter_flippable_couplings())

    def incident(self, qubit: int) -> Optional[Coupling]:
        """
            Returns the coupling incident to the given qubit in this matching,
            or `None` if the qubit is not matched.
        """
        return self._incident_coupling.get(qubit, None)

    def is_flippable(self, coupling: CouplingLike) -> bool:
        """
            Checks whether the coupling can be flipped:

            - always true if the coupling is already present in the matching;
            - otherwise true only if neither qubit in the coupling is currently matched.
        """
        coupling = _validate_coupling(self.topology.qubits, coupling)
        if coupling not in self.topology.couplings:
            raise ValueError(f"Invalid coupling {coupling} for the given topology.")
        return self._is_flippable(coupling)

    def flip(self, coupling: CouplingLike) -> "Matching":
        """
            Flips the given coupling in the matching (removes it if it is already present,
            adds it if it is not yed present and can be added).
            Raises `ValueError` if the coupling is not flippable.

            The matching is modified in-place and then returned, as per the
            [fluent API pattern](https://en.wikipedia.org/wiki/Fluent_interface).
        """
        coupling = _validate_coupling(self.topology.qubits, coupling)
        return self._flip(coupling)

    def _is_flippable(self, coupling: Coupling) -> bool:
        if coupling in self._matched_couplings:
            return True
        fst, snd = coupling
        if fst in self._matched_qubits:
            return False
        if snd in self._matched_qubits:
            return False
        return True

    def _flip(self, coupling: Coupling) -> "Matching":
        if not self._is_flippable(coupling):
            raise ValueError(f"Cannot add coupling {coupling} to matching:"
                             f"one of the qubits is already matched. ")
        fst, snd = coupling
        if coupling in self._matched_couplings:
            self._matched_couplings.remove(coupling)
            self._matched_qubits.remove(fst)
            self._matched_qubits.remove(snd)
            del self._incident_coupling[fst]
            del self._incident_coupling[snd]
        else:
            self._matched_couplings.add(coupling)
            self._matched_qubits.add(fst)
            self._matched_qubits.add(snd)
            self._incident_coupling[fst] = coupling
            self._incident_coupling[snd] = coupling
        return self

    def _iter_flippable_couplings(self) -> Iterator[Coupling]:
        for coupling in self._topology.couplings:
            fst, snd = coupling
            if coupling in self._matched_couplings:
                yield coupling
            elif fst not in self._matched_qubits and snd not in self._matched_qubits:
                yield coupling
            else:
                # coupling is not flippable
                pass
