def apply_permutation(
    qc: "qiskit.QuantumCircuit", permutation: list
) -> "qiskit.QuantumCircuit":
    try:
        from qiskit import QuantumCircuit
    except:
        raise Exception("Qiskit is not installed")

    register = qc.qregs[0]
    qc_out = QuantumCircuit(register)
    for instruction in qc:
        op_qubits = [
            register[permutation[register.index(q)]] for q in instruction.qubits
        ]
        qc_out.append(instruction.operation, op_qubits)
    return qc_out


def verify_equality(qc_in: "qiskit.QuantumCircuit", qc_out: "qiskit.QuantumCircuit"):
    try:
        from qiskit.quantum_info import Operator
    except:
        raise Exception("Please install qiskit to compare to quantum circuits")
    return Operator.from_circuit(qc_in).equiv(Operator.from_circuit(qc_out))
