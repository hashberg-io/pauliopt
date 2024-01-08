from qiskit import QuantumCircuit


def tableau_from_circuit(tableau, circ: QuantumCircuit):
    qreg = circ.qregs[0]

    for op in circ:
        if op.operation.name == "h":
            tableau.append_h(qreg.index(op.qubits[0]))
        elif op.operation.name == "s":
            tableau.append_s(qreg.index(op.qubits[0]))
        elif op.operation.name == "cx":
            tableau.append_cnot(qreg.index(op.qubits[0]), qreg.index(op.qubits[1]))
        else:
            raise TypeError(
                f"Unrecongnized Gate type: {op.operation.name} for Clifford Tableaus"
            )
    return tableau


def tableau_from_circuit_prepend(tableau, circ: QuantumCircuit):
    qreg = circ.qregs[0]

    for op in circ:
        if op.operation.name == "h":
            tableau.prepend_h(qreg.index(op.qubits[0]))
        elif op.operation.name == "s":
            tableau.prepend_s(qreg.index(op.qubits[0]))
        elif op.operation.name == "cx":
            tableau.prepend_cnot(qreg.index(op.qubits[0]), qreg.index(op.qubits[1]))
        else:
            raise TypeError(
                f"Unrecongnized Gate type: {op.operation.name} for Clifford Tableaus"
            )
    return tableau
