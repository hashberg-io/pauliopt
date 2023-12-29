import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, HGate, SGate, SXGate, TGate, TdgGate, SdgGate, \
    SXdgGate, ZGate, YGate, XGate, SwapGate, CYGate, CZGate, CCXGate, CCZGate, RXGate, \
    RZGate, RYGate, CRXGate, CRYGate, CRZGate
from qiskit.quantum_info import Operator

guadalupe_connectivity = [
    [
        0,
        1
    ],
    [
        1,
        0
    ],
    [
        1,
        2
    ],
    [
        1,
        4
    ],
    [
        2,
        1
    ],
    [
        2,
        3
    ],
    [
        3,
        2
    ],
    [
        3,
        5
    ],
    [
        4,
        1
    ],
    [
        4,
        7
    ],
    [
        5,
        3
    ],
    [
        5,
        8
    ],
    [
        6,
        7
    ],
    [
        7,
        4
    ],
    [
        7,
        6
    ],
    [
        7,
        10
    ],
    [
        8,
        5
    ],
    [
        8,
        9
    ],
    [
        8,
        11
    ],
    [
        9,
        8
    ],
    [
        10,
        7
    ],
    [
        10,
        12
    ],
    [
        11,
        8
    ],
    [
        11,
        14
    ],
    [
        12,
        10
    ],
    [
        12,
        13
    ],
    [
        12,
        15
    ],
    [
        13,
        12
    ],
    [
        13,
        14
    ],
    [
        14,
        11
    ],
    [
        14,
        13
    ],
    [
        15,
        12
    ]
]

NAME_CONVERSION = {
    "H": HGate,
    "X": XGate,
    "Y": YGate,
    "Z": ZGate,
    "S": SGate,
    "Sdg": SdgGate,
    # "V": SXGate,
    # "Vdg": SXdgGate,
    "T": TGate,
    "Tdg": TdgGate,

    "swap": SwapGate,
    "CX": CXGate,
    "CY": CYGate,
    "CZ": CZGate,

    "CCX": CCXGate,
    "CCZ": CCZGate,

    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,

    "crx": CRXGate,
    "cry": CRYGate,
    "crz": CRZGate,
}


def random_circuit(nr_gates, nr_qubits, gate_choice=None):
    qc = QuantumCircuit(nr_qubits)

    if gate_choice is None:
        gate_choice = list(NAME_CONVERSION.values())

    else:
        gate_choice = [NAME_CONVERSION[gate_name] for gate_name in gate_choice]

    single_qubit_gates = [HGate, SGate, SXGate,
                          XGate, YGate, ZGate,
                          TGate, TdgGate,
                          RXGate, RYGate, RZGate, ]
    two_qubit_gates = [CXGate, CYGate, CZGate, CRXGate, CRYGate, CRZGate]
    three_qubit_gates = [CCXGate, CCZGate]
    single_param_gates = [RXGate, RYGate, RZGate, CRXGate, CRYGate, CRZGate]

    for _ in range(nr_gates):
        gate_class = np.random.choice(gate_choice)
        if gate_class in single_param_gates:
            gate_instance = gate_class(np.random.uniform(0, 2 * np.pi))
        else:
            gate_instance = gate_class()

        if gate_class in single_qubit_gates:
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.append(gate_instance, [qubit])

        elif gate_class in two_qubit_gates:
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.append(gate_instance, [control, target])
        elif gate_class in three_qubit_gates:
            control1 = np.random.choice([i for i in range(nr_qubits)])
            control2 = np.random.choice([i for i in range(nr_qubits) if i != control1])
            target = np.random.choice(
                [i for i in range(nr_qubits) if i not in [control1, control2]])
            qc.append(gate_instance, [control1, control2, target])

    return qc


def verify_equality(qc_in, qc_out):
    return Operator.from_circuit(qc_in) \
        .equiv(Operator.from_circuit(qc_out))


def random_clifford_circuit(nr_gates=20, nr_qubits=4, gate_choice=None):
    qc = QuantumCircuit(nr_qubits)
    if gate_choice is None:
        gate_choice = ["CY", "CZ", "CX", "H", "S", "V"]
    for _ in range(nr_gates):
        gate_t = np.random.choice(gate_choice)
        if gate_t == "CX":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cx(control, target)
        elif gate_t == "CY":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cy(control, target)
        elif gate_t == "CZ":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cz(control, target)
        elif gate_t == "H":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.h(qubit)
        elif gate_t == "S":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.s(qubit)
        elif gate_t == "V":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.sx(qubit)
        elif gate_t == "CX":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cx(control, target)
    return qc


def random_hscx_circuit(nr_gates=20, nr_qubits=4):
    gate_choice = ["CX", "H", "S"]
    return random_clifford_circuit(nr_gates=nr_gates,
                                   nr_qubits=nr_qubits,
                                   gate_choice=gate_choice)


def apply_permutation(qc: QuantumCircuit, permutation: list):
    if len(qc.qregs) != 1:
        raise ValueError("Quantum circuit must have exactly one quantum register")
    register = qc.qregs[0]
    qc_out = QuantumCircuit(register)
    for instruction in qc:
        op_qubits = [register[permutation[register.index(q)]] for q in instruction.qubits]
        instruction.qubits = tuple(op_qubits)
        qc_out.append(instruction, instruction.qubits)
    return qc_out
