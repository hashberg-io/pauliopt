from enum import Enum


class Pauli(Enum):
    I = 0
    X = 1
    Y = 2
    Z = 3


I = Pauli.I
X = Pauli.X
Y = Pauli.Y
Z = Pauli.Z


def _pauli_to_string(pauli: Pauli):
    if pauli == Pauli.I:
        return "I"
    elif pauli == Pauli.X:
        return "X"
    elif pauli == Pauli.Y:
        return "Y"
    elif pauli == Pauli.Z:
        return "Z"
    else:
        raise Exception(f"{pauli} is not a Paulimatrix")
