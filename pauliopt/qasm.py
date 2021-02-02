from pauliopt.phase import PhaseCircuit

gate_width = {
    'cx': 2, 'x': 1
}

def read_qasm(path):
    commands = []
    n_qubits = 0
    with open(path, 'r') as f:
        # skip first two lines
        f.readline()
        f.readline()

        for line in f.readlines():
            line = line.strip().replace(',', '').replace(';', '')
            gate, *qubits_str = line.split(' ')

            # turn q[bit] -> bit 
            qubits = [int(qubit[2:-1]) for qubit in qubits_str]
            gate = gate.lower()

            if gate == 'qreg':
                n_qubits = qubits[0]
            else:
                commands.append((gate, qubits))
    
    circuit: PhaseCircuit = PhaseCircuit(n_qubits)
    for gate, qubits in commands:
        if gate not in gate_width:
            raise Exception(f'Gate {gate} not recognised.')
        if gate_width[gate] != len(qubits):
            raise Exception(f'Expected {gate_width[gate]} qubits, got {len(qubits)} instead.')

        if gate == 'cx':
            circuit.cx(*qubits)
        elif gate == 'x':
            circuit.x(*qubits)

    return circuit
