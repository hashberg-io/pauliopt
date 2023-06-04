from pauliopt.pauli.summedop import SummedOp

class Pauli:
    """
    Class representing a Pauli term with a coefficient.

    Args:
        pauli_string (str): The Pauli string representing the Pauli term.
        coefficient (float, optional): Coefficient of the Pauli term. Defaults to 1.0.
    """

    def __init__(self, pauli_string, coefficient=1.0):
        self.pauli_string = pauli_string
        self.coefficient = coefficient

    def __repr__(self):
        return f"{self.coefficient} * {self.pauli_string}"

    def __xor__(self, other):
        """
        XOR operation between two Pauli terms or a Pauli term and a SummedOp.

        Args:
            other (Pauli or SummedOp): The other Pauli term or SummedOp.

        Returns:
            Pauli or SummedOp: Result of the XOR operation.
        """
        if isinstance(other, Pauli):
            return SummedOp([self, other])
        elif isinstance(other, SummedOp):
            return SummedOp([self] + other.ops)
        raise TypeError("Unsupported operation: Pauli can only be XORed with another Pauli or a SummedOp.")

    def __add__(self, other):
        """
        Addition operation between two Pauli terms or a Pauli term and a SummedOp.

        Args:
            other (Pauli or SummedOp): The other Pauli term or SummedOp.

        Returns:
            Pauli or SummedOp: Result of the addition operation.
        """
        if isinstance(other, Pauli):
            if self.pauli_string == other.pauli_string:
                return Pauli(self.pauli_string, self.coefficient + other.coefficient)
            return SummedOp([self, other])
        raise TypeError("Unsupported operation: Pauli can only be added to another Pauli.")

    def __mul__(self, other):
        """
        Scalar multiplication between a Pauli term and a scalar.

        Args:
            other (int or float): Scalar to multiply.

        Returns:
            Pauli: Result of the scalar multiplication.
        """
        if isinstance(other, (int, float)):
            return Pauli(self.pauli_string, self.coefficient * other)
        raise TypeError("Unsupported operation: Pauli can only be multiplied by a scalar.")

    def __rmul__(self, other):
        return self * other

    def print_expression(self, indent=""):
        """
        Generate a string representation of the Pauli term.

        Args:
            indent (str, optional): Indentation string for the expression. Defaults to "".

        Returns:
            str: String representation of the Pauli term.
        """
        if isinstance(self, Pauli):
            return indent + str(self)


