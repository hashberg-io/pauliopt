
"""
    This module contains code to optimize circuits of mixed ZX phase gadgets
    using peephole methods.
"""



from collections import deque
from math import pi as real_pi
from math import ceil, log10
from typing import (Deque, Dict, FrozenSet, Generic, List, Optional, Protocol, runtime_checkable,
                    Set, Tuple, Type, TypedDict, Union)
import numpy as np # type: ignore

from pauliopt.phase import PhaseCircuit, PhaseGadget, Z, X
from pauliopt.utils import pi, Angle
from pauliopt.topologies import Topology
from cmath import sin, cos, phase



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






def peephole(circuit: PhaseCircuit) -> PhaseCircuit:
    def peep(l, start):
        i = start
        while i < len(l):
            while l[i].angle == 2*pi:
                l.pop(i)
                return peep(l, start=max(0, i-1))
            if i != len(l)-1:
                if l[i].basis == l[i+1].basis and l[i].qubits == l[i+1].qubits:
                    basis = Z if l[i].basis == 'Z' else X
                    total_angle = l[i].angle + l[i+1].angle
                    l[i] = basis(total_angle) @ l[i].qubits
                    l.pop(i+1)
                    return peep(l, start=i)
            i += 1
    l = list(circuit.gadgets)
    peep(l, start=0)
    return PhaseCircuit(circuit.num_qubits, l)


def _group_pi(table: Dict[FrozenSet[int], Angle]):
    """ Group pi gates into one large pi gadget """
    pi_qubits: Set[int] = set()
    keys_to_remove = []
    for qubits, phase in table.items():
        if phase == pi:
            pi_qubits ^= qubits
            keys_to_remove.append(qubits)
    for key in keys_to_remove:
        del table[key]
    pi_key = frozenset(pi_qubits)
    if len(pi_key) > 0:
        table[pi_key] = table.get(pi_key, Angle(0)) + pi

def split_pi(circuit: PhaseCircuit) -> PhaseCircuit:
    """ Turn pi gadgets into single pi gadgets """
    l = list(circuit.gadgets)

    for i, gadget in enumerate(l):
        if gadget.angle == pi:
            basis: Union[Type[Z], Type[X]] = Z if l[i].basis == 'Z' else X
            split_gadgets = [basis(pi) @ {leg} for leg in gadget.qubits]
            l = l[:i] + split_gadgets + l[i+1:]
    return PhaseCircuit(circuit.num_qubits, l)


def aggregate(circuit: PhaseCircuit) -> PhaseCircuit:
    """
    Combine the gadgets into layers of same colour gadgets,
    potentially applying spider nest identites as well
    """
    if circuit.num_gadgets == 0:
        return circuit
    l = circuit.gadgets
    
    current_basis = l[0].basis
    table: Dict[FrozenSet[int], Angle] = dict()
    blocks = [(current_basis, table)]
    
    for gadget in l:
        basis, qubits, angle = gadget.basis, gadget.qubits, gadget.angle
        if basis == current_basis:
            table[qubits] = table.get(qubits, Angle(0)) + angle
        else:
#             spider_nest(table)
            _group_pi(table)
            current_basis = basis
            table = {qubits: angle}
            blocks.append((current_basis, table))
        _group_pi(table)
    
    
    new_circuit: PhaseCircuit = PhaseCircuit(circuit.num_qubits)
    
    for basis, table in blocks:
        basis_cls: Union[Type[Z], Type[X]] = Z if basis == 'Z' else X
        for qubits, angle in table.items():
            if angle == 2*pi:
                continue
            new_circuit >>= basis_cls(angle) @ qubits
            
    return new_circuit


def commute(circuit: PhaseCircuit) -> PhaseCircuit:
    """ Apply commutation relations to group same-colored gadgets """
    l = list(circuit.gadgets)
    
    for i, gadget1 in enumerate(l):
        j = i-1
        basis1, qubits1 = gadget1.basis, gadget1.qubits
        while j >= 0:
            gadget2 = l[j]
            basis2, qubits2 = gadget2.basis, gadget2.qubits
            
            if basis1 != basis2 and len(qubits1 & qubits2) % 2 == 0:
                j -= 1
            else:
                break
        
        if j != i-1:
            l.pop(i)
            l.insert(j+1, gadget1)
    return PhaseCircuit(circuit.num_qubits, l)

def _round_near_pi(angle: float) -> Angle:
    multiple = angle * 8 / real_pi
    if round(multiple) - multiple < 1e-10:
        return Angle(pi / 8 * round(multiple))
    return Angle(angle / real_pi)

def eu(a1, a2, a3) -> Tuple[Angle, Angle, Angle]:
    h1, h2, h3 = (a1/2, a2/2, a3/2)
    xp = h1 + h3
    xm = h1 - h3
    
    z1 = cos(h2)*cos(xp) + 1j*sin(h2)*cos(xm)
    z2 = cos(h2)*sin(xp) - 1j*sin(h2)*sin(xm)
    
    b1 = phase(z1) + phase(z2)
    b2 = 2*phase(abs(z1) + 1j*abs(z2))
    b3 = phase(z1) - phase(z2)

    angle1, angle2, angle3 = map(_round_near_pi, [b1, b2, b3])
    return angle1, angle2, angle3