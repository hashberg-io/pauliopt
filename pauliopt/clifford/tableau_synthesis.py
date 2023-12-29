import networkx as nx
from networkx.algorithms.approximation import steiner_tree

from pauliopt.circuits import Circuit
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.gates import CX, H, S
from pauliopt.topologies import Topology


def heurisitc_fkt(row, G, remaining: CliffordTableau):
    """
    The heuristic function for picking the pivot in the tableau synthesis algorithm.

    :param row: The row to consider
    :param G: The graph of the topology
    :param remaining: The remaining tableau
    """
    row_x = [nx.shortest_path_length(G, source=row, target=col) for col in G.nodes
             if remaining.x_out(row, col) != 0]
    row_z = [nx.shortest_path_length(G, source=row, target=col) for col in G.nodes
             if remaining.z_out(row, col) != 0]
    dist_x = sum(row_x)
    dist_z = sum(row_z)
    return dist_x + dist_z


def pick_pivot(G, remaining: "CliffordTableau", possible_swaps, include_swaps):
    """
    Pick the pivot to eliminate the next column in the tableau synthesis algorithm.

    We currently use the heuristic function h to pick the pivot,
    i.e choose the row with the smallest `heurisitc_fkt` value.

    :param G: The graph of the topology
    :param remaining: The remaining tableau
    :param possible_swaps: The columns that can be swapped
    :param include_swaps: Whether to include the columns that can be swapped
    """
    scores = []
    has_cutting_swappable = any([not is_cutting(i, G) for i in possible_swaps])
    for col in G.nodes:
        if not is_cutting(col, G) or \
                (include_swaps and has_cutting_swappable and col in possible_swaps):
            scores.append((col, col, heurisitc_fkt(col, G, remaining)))
    assert len(scores) > 0
    return min(scores, key=lambda x: x[2])[:2]


def update_dfs(dfs, parent, child):
    """
    Helper function to update the dfs list in place. (See compute_steiner_tree)

    :param dfs: The dfs list to update
    :param parent: The parent node
    :param child: The child node
    """
    for i, (p, c) in enumerate(dfs):
        if c == parent:
            dfs[i] = (p, child)
        if p == parent:
            dfs[i] = (child, c)
        if c == child:
            dfs[i] = (p, parent)
        if p == child:
            dfs[i] = (parent, c)
    return dfs


def relabel_graph_inplace(G, parent, child):
    """
    Helper function to relabel the graph in place. (See compute_steiner_tree)

    :param G: The graph to relabel
    :param parent: The parent node
    :param child: The child node
    """
    swap = {parent: -1}
    nx.relabel_nodes(G, swap, copy=False)
    swap = {child: parent}
    nx.relabel_nodes(G, swap, copy=False)
    swap = {-1: child}
    nx.relabel_nodes(G, swap, copy=False)


def compute_steiner_tree(root: int, nodes: [int], sub_graph: nx.Graph,
                         include_swaps=False, lookup=None,
                         swappable_nodes=None, permutation=None, n_qubits=None):
    """
    Compute the steiner tree of the sub_graph with the given nodes.
    This function is a wrapper around the networkx steiner tree function.

    It will additionally swap the columns of the remaining tableau to further reduce
    the amount of CNOTs if include_swaps is True.
    Include_swaps requires lookup, swappable_nodes, permutation and n_qubits to be set.

    :param root: The root node of the steiner tree
    :param nodes: The nodes to include in the steiner tree
    :param sub_graph: The graph of the topology

    :param include_swaps: Whether to include swaps in the steiner tree
    :param lookup: The lookup table of the topology
    :param swappable_nodes: The nodes that can be swapped
    :param permutation: The permutation of the topology
    :param n_qubits: The number of qubits
    """
    steiner_stree = steiner_tree(sub_graph, nodes)
    steiner_stree = nx.Graph(steiner_stree)
    if len(steiner_stree.nodes()) < 1:
        return []
    if include_swaps:
        if lookup is None:
            raise Exception("Lookup table is required to include swaps")
        if swappable_nodes is None:
            raise Exception("Swappable nodes are required to include swaps")
        if permutation is None:
            raise Exception("Permutation is required to include swaps")
        if n_qubits is None:
            raise Exception("Number of qubits is required to include swaps")

        for _ in range(n_qubits):
            dfs = list(reversed(list(nx.dfs_edges(steiner_stree, source=root))))
            swapped = []
            while dfs:
                parent, child = dfs.pop(0)
                if parent == root:
                    continue

                # if the parent is zero and the child is one and both are swappable
                # then swap them
                if lookup[parent] == 0 and lookup[child] == 1 and \
                        child in swappable_nodes and \
                        parent in swappable_nodes:
                    relabel_graph_inplace(steiner_stree, parent, child)
                    relabel_graph_inplace(sub_graph, parent, child)
                    dfs = update_dfs(dfs, parent, child)
                    # remaining.swap_cols(parent, child)
                    permutation[parent], permutation[child] = \
                        permutation[child], permutation[parent]

                    swapped.append(parent)
                    swapped.append(child)

        steiner_stree = steiner_tree(sub_graph, nodes)
    traversal = nx.bfs_edges(steiner_stree, source=root)
    return list(reversed(list(traversal)))


def is_cutting(vertex, g):
    """
    Check if the given vertex is a cutting vertex in the given graph.

    :param vertex: The vertex to check
    :param g: The graph to check
    """
    return vertex in nx.articulation_points(g)


def sanitize_z(row, row_z, remaining, apply):
    """
    Sanitization process for the stabilizer part.

    Essentially:
    - If the z_out is Y (=3), then apply S
    - If the z_out is X (=1), then apply H


    :param row: The row of the tableau
    :param row_z: The row of the tableau for the stabilizer part
    :param remaining: The remaining tableau
    :param apply: The function to apply a gate
    """
    for column in row_z:
        if remaining.z_out(row, column) == 3:
            apply("S", (column,))

        if remaining.z_out(row, column) == 1:
            apply("H", (column,))

    # caveat for the pivot
    if remaining.x_out(row, row) == 3:
        apply("S", (row,))


def sanitize_field_x(row, row_x, remaining, apply):
    """
    Sanitization process for the destabilizer part.

    Essentially:
    - If the x_out is Y (=3), then apply S
    - If the x_out is X (=2), then apply H

    :param row: The row of the tableau
    :param row_x: The row of the tableau for the destabilizer part
    :param remaining: The remaining tableau
    :param apply: The function to apply a gate
    """
    for column in row_x:
        if remaining.x_out(row, column) == 3:
            apply("S", (column,))

        if remaining.x_out(row, column) == 2:
            apply("H", (column,))


def remove_interactions(pivot, row, sub_graph, remaining, apply, basis,
                        include_swaps=False, swappable_nodes=None, permutation=None):
    """
    Remove the interactions of the destabilizer/stabilizer part.
    This function assumed that all elements are Z or I.

    Include swaps requires swappable_nodes, permutation and include_swaps to be set.

    :param pivot: The pivot of the tableau
    :param row: The specific row of the tableau
    :param sub_graph: The graph of the topology
    :param remaining: The remaining tableau
    :param apply: The function to apply a gate
    :param basis: The basis of the tableau (x for destabilizer or z for stabilizer)
    :param swappable_nodes: The nodes that can be swapped
    :param permutation: The permutation of the topology
    :param include_swaps: Whether to include swaps in the steiner tree

    """
    row = list(set([pivot] + row))
    lookup = {node: int(remaining.x_out(pivot, node) != 0) for node in
              sub_graph.nodes}
    traversal = compute_steiner_tree(pivot, row, sub_graph,
                                     include_swaps=include_swaps,
                                     lookup=lookup,
                                     swappable_nodes=swappable_nodes,
                                     permutation=permutation,
                                     n_qubits=remaining.n_qubits)
    if basis == "x":
        for parent, child in traversal:
            if remaining.x_out(pivot, parent) == 0:
                apply("CNOT", (child, parent))

        for parent, child in traversal:
            apply("CNOT", (parent, child))
    elif basis == "z":
        for parent, child in traversal:
            if remaining.z_out(pivot, parent) == 0:
                apply("CNOT", (parent, child))

        for parent, child in traversal:
            apply("CNOT", (child, parent))


def steiner_reduce_column(pivot, sub_graph, remaining,
                          apply, swappable_nodes=None, permutation=None,
                          include_swaps=False):
    """
    Steiner reduce a column of the tableau.

    :param pivot: The pivot of the tableau
    :param sub_graph: The graph of the topology
    :param remaining: The remaining tableau
    :param apply: The function to apply a gate
    :param swappable_nodes: The nodes that can be swapped
    :param permutation: The permutation of the topology
    :param include_swaps: Whether to include swaps in the steiner tree
    """
    # 2. Sanitize the destabilizer row
    row_x = [col for col in sub_graph.nodes if remaining.x_out(pivot, col) != 0]
    sanitize_field_x(pivot, row_x, remaining, apply)

    # 3. Remove the interactions from the destabilizer row
    remove_interactions(pivot, row_x, sub_graph, remaining, apply, "x",
                        include_swaps=include_swaps,
                        swappable_nodes=swappable_nodes,
                        permutation=permutation)

    # 4. Sanitize the stabilizer row
    row_z = [row for row in sub_graph.nodes if remaining.z_out(pivot, row) != 0]
    sanitize_z(pivot, row_z, remaining, apply)

    # 5. Remove the interactions from the stabilizer row
    remove_interactions(pivot, row_z, sub_graph, remaining, apply, "z",
                        include_swaps=include_swaps,
                        swappable_nodes=swappable_nodes,
                        permutation=permutation)

    # ensure that the pivots are in ZX basis
    # (this is provided by the construction of a tableau)
    assert remaining.x_out(pivot, pivot) == 1
    assert remaining.z_out(pivot, pivot) == 2


def get_non_cutting_vertex(G, pivot_col, swappable_nodes):
    non_cutting_vertices = []
    for node in G.nodes:
        if not is_cutting(node, G) and node in swappable_nodes:
            shortest_path_len = nx.shortest_path_length(G, source=node,
                                                        target=pivot_col,
                                                        weight="weight")
            non_cutting_vertices.append((node, shortest_path_len))
    non_cutting = min(non_cutting_vertices, key=lambda x: x[1])[0]
    return non_cutting


def synthesize_tableau(tableau: CliffordTableau, topo: Topology, include_swaps=True):
    """
    Architecture aware synthesis of a Clifford tableau.
    This is the implementation of the algorithm described in Winderl et. al. [1]

    :param tableau: The Clifford tableau
    :param topo: The topology
    :param include_swaps: Whether to allow initial and final measurement permutations

    :return (qc, perm): The synthesized circuit and a inital/final permutation


    References

    [1] Winderl, Huang, et al. "Architecture-Aware Synthesis of Stabilizer Circuits from Clifford Tableaus." arXiv preprint arXiv:2309.08972 (2023).

    """

    qc = Circuit(tableau.n_qubits)

    remaining = tableau.inverse()
    permutation = {v: v for v in range(tableau.n_qubits)}
    swappable_nodes = list(range(tableau.n_qubits))

    G = topo.to_nx
    for e1, e2 in G.edges:
        G[e1][e2]["weight"] = 0

    def apply(gate_name: str, gate_data: tuple):
        if gate_name == "CNOT":
            remaining.append_cnot(gate_data[0], gate_data[1])
            qc.add_gate(CX(gate_data[0], gate_data[1]))
            if gate_data[0] in swappable_nodes:
                swappable_nodes.remove(gate_data[0])
            if gate_data[1] in swappable_nodes:
                swappable_nodes.remove(gate_data[1])
            G[gate_data[0]][gate_data[1]]["weight"] = 2
        elif gate_name == "H":
            remaining.append_h(gate_data[0])
            qc.add_gate(H(gate_data[0]))
        elif gate_name == "S":
            remaining.append_s(gate_data[0])
            qc.add_gate(S(gate_data[0]))
        else:
            raise Exception("Unknown Gate")

    while G.nodes:
        # 1. Pick a pivot
        pivot_col, pivot_row = pick_pivot(G, remaining, swappable_nodes,
                                          include_swaps)

        if is_cutting(pivot_col, G) and include_swaps:
            non_cutting = get_non_cutting_vertex(G, pivot_col, swappable_nodes)

            relabel_graph_inplace(G, non_cutting, pivot_col)
            permutation[pivot_col], permutation[non_cutting] = \
                permutation[non_cutting], permutation[pivot_col]

        steiner_reduce_column(pivot_col, G, remaining,
                              apply, swappable_nodes, permutation, include_swaps)

        if pivot_col in swappable_nodes:
            swappable_nodes.remove(pivot_col)
        G.remove_node(pivot_col)

    signs_copy_z = remaining.signs[tableau.n_qubits:2 * tableau.n_qubits].copy()
    for col in range(tableau.n_qubits):
        if signs_copy_z[col] != 0:
            apply("H", (col,))

    for col in range(tableau.n_qubits):
        if signs_copy_z[col] != 0:
            apply("S", (col,))
            apply("S", (col,))

    for col in range(tableau.n_qubits):
        if signs_copy_z[col] != 0:
            apply("H", (col,))

    for col in range(tableau.n_qubits):
        if remaining.signs[col] != 0:
            apply("S", (col,))
            apply("S", (col,))

    return qc, permutation
