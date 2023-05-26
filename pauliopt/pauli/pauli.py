class Pauli:
    def __init__(self, pauli_string, coefficient=1.0):
        self.pauli_string = pauli_string
        self.coefficient = coefficient

    def __repr__(self):
        return f"{self.coefficient} * {self.pauli_string}"

    def __xor__(self, other):
        if isinstance(other, Pauli):
            return Pauli(f"{self.pauli_string}{other.pauli_string}", self.coefficient * other.coefficient)
        raise TypeError("Unsupported operation: Pauli can only be XORed with another Pauli.")

    def __add__(self, other):
        if isinstance(other, Pauli):
            if self.pauli_string == other.pauli_string:
                return Pauli(self.pauli_string, self.coefficient + other.coefficient)
            return SummedOp([self, other])
        raise TypeError("Unsupported operation: Pauli can only be added to another Pauli.")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Pauli(self.pauli_string, self.coefficient * other)
        raise TypeError("Unsupported operation: Pauli can only be multiplied by a scalar.")

    def __rmul__(self, other):
        return self * other

    def print_expression(self, indent=""):
        if isinstance(self, Pauli):
            return indent + str(self)
