import itertools

import networkx as nx
import numpy as np
import pytket
from pytket._tket.circuit import PauliExpBox
from pytket._tket.pauli import I as TketI
from pytket._tket.pauli import X as TketX
from pytket._tket.pauli import Y as TketY
from pytket._tket.pauli import Z as TketZ
from pytket._tket.transform import Transform
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit
from qiskit import QuantumCircuit

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli_strings import X, Y, Z, I

PAULI_TO_TKET = {X: TketX, Y: TketY, Z: TketZ, I: TketI}


def apply_permutation(qc: QuantumCircuit, permutation: list) -> QuantumCircuit:
    """
    Apply a permutation to a qiskit quantum circuit.
    :param qc:
    :param permutation:
    :return:
    """
    register = qc.qregs[0]
    qc_out = QuantumCircuit(register)
    for instruction in qc:
        op_qubits = [
            register[permutation[register.index(q)]] for q in instruction.qubits
        ]
        qc_out.append(instruction.operation, op_qubits)
    return qc_out


def tket_to_qiskit(circuit: pytket.Circuit) -> QuantumCircuit:
    """
    Convert a tket circuit to qiskit circuit.
    :param circuit:
    :return:
    """
    return tk_to_qiskit(circuit)


def create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels):
    """
    Generate a random phase gadget.
    :param num_qubits:
    :param min_legs:
    :param max_legs:
    :param allowed_angels:
    :return:
    """
    angle = np.random.choice(allowed_angels)
    nr_legs = np.random.randint(min_legs, max_legs)
    legs = np.random.choice([i for i in range(num_qubits)], size=nr_legs, replace=False)
    phase_gadget = [I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice([X, Y, Z])
    return PPhase(angle) @ phase_gadget


def generate_random_pauli_polynomial(
    num_qubits: int, num_gadgets: int, min_legs=None, max_legs=None, allowed_angels=None
):
    """
    Generate a random pauli polynomial.
    :param num_qubits:
    :param num_gadgets:
    :param min_legs:
    :param max_legs:
    :param allowed_angels:
    :return:
    """
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.125 * np.pi]

    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(
            num_qubits, min_legs, max_legs, allowed_angels
        )

    return pp


def pauli_poly_to_tket(pp: PauliPolynomial):
    """
    Convert a PauliPolynomial to a tket boxes.
    :param pp:
    :return:
    """
    circuit = pytket.Circuit(pp.num_qubits)
    for gadget in pp.pauli_gadgets:
        circuit.add_pauliexpbox(
            PauliExpBox(
                [PAULI_TO_TKET[p] for p in gadget.paulis],
                gadget.angle / np.pi,
            ),
            list(range(pp.num_qubits)),
        )
    Transform.DecomposeBoxes().apply(circuit)
    return tket_to_qiskit(circuit)


def verify_equality(qc_in: QuantumCircuit, qc_out: QuantumCircuit) -> bool:
    """
    Verify the equality up to a global phase
    :param qc_in:
    :param qc_out:
    :return:
    """
    try:
        from qiskit.quantum_info import Operator
    except:
        raise Exception("Please install qiskit to compare to quantum circuits")
    return Operator.from_circuit(qc_in).equiv(Operator.from_circuit(qc_out))


def check_matching_architecture(qc: QuantumCircuit, G: nx.Graph):
    """
    Check if a circuit matches the architecture graph.

    :param qc:
    :param G:
    :return:
    """
    for gate in qc:
        if gate.operation.num_qubits == 2:
            ctrl, target = gate.qubits
            ctrl, target = (
                ctrl._index,
                target._index,
            )  # TODO refactor this to a non deprecated way
            if not G.has_edge(ctrl, target):
                return False
    return True


def generate_all_combination_pauli_polynomial(n_qubits=2):
    """
    Generate a PauliPolynomial consisting of all possible pauli strings.

    :param n_qubits:
    :return:
    """
    allowed_angels = [2 * np.pi, np.pi, np.pi / 2, np.pi / 4, np.pi / 8]
    pp = PauliPolynomial(n_qubits)
    for comb in itertools.product([X, Y, Z, I], repeat=n_qubits):
        pp >>= PPhase(np.random.choice(allowed_angels)) @ list(comb)
    return pp


def get_two_qubit_count(circ: QuantumCircuit):
    """
    Get the number of two qubit gates in circuit.
    :param circ:
    :return:
    """
    ops = circ.count_ops()
    two_qubit_count = 0
    two_qubit_ops = ["cx", "cy", "cz"]
    for op_key in two_qubit_ops:
        if op_key in ops.keys():
            two_qubit_count += ops[op_key]

    return two_qubit_count


def generate_random_depth_1_clifford(gate_class, num_qubits):
    """
    Generates a random depth 1 clifford. given it's gate class and the number of qubits it might act on.

    :param gate_class:
    :param num_qubits:
    :return:
    """
    base_names = [cls.__name__ for cls in gate_class.__bases__]
    size = 1 if "SingleQubitClifford" in base_names else 2
    qubits = tuple(np.random.choice(list(range(num_qubits)), size=size, replace=False))
    return gate_class(*qubits)
