"""
    This module contatins code to convert CXCircuitLayer into pytket.Circuit
"""
from pytket import Circuit
from pauliopt.phase import CXCircuitLayer
def convert_cx_layer(cx_layer: CXCircuitLayer):
    num_qubits = cx_layer.topology.num_qubits
    circuit = Circuit(num_qubits)

    for gate in cx_layer.gates:
        circuit.CX(*gate)

    return circuit
