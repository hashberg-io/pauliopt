import numpy as np
import qiskit.quantum_info


def mult_paulis(p1, p2, sign1, sign2, n_qubits):
    """
    Small helper function to multiply two Pauli strings and correctly update the sign.

    Args:
        p1 (np.ndarray): Pauli string 1
        p2 (np.ndarray): Pauli string 2
        sign1 (int): Sign of Pauli string 1
        sign2 (int): Sign of Pauli string 2
        n_qubits (int): Number of qubits in the Pauli strings

    Returns:
        np.ndarray: Pauli string 1 * Pauli string 2
    """
    x_1 = p1[:n_qubits].copy()
    z_1 = p1[n_qubits:].copy()
    x_2 = p2[:n_qubits].copy()
    z_2 = p2[n_qubits:].copy()

    x_1_z_2 = z_1 * x_2
    z_1_x_2 = x_1 * z_2

    ac = (x_1_z_2 + z_1_x_2) % 2

    x_1 = (x_1 + x_2) % 2
    z_1 = (z_1 + z_2) % 2

    x_1_z_2 = ((x_1_z_2 + x_1 + z_1) % 2) * ac
    sign_change = int(((np.sum(ac) + 2 * np.sum(x_1_z_2)) % 4) > 1)
    new_sign = (sign1 + sign2 + sign_change) % 4
    new_p1 = np.concatenate([x_1, z_1])
    return new_p1, new_sign


class CliffordTableau:
    """
    Class for storing and manipulating Clifford tableau.
    A Clifford tableau is a representation of a Clifford circuit as a
    2n x 2n binary matrix, where n is the number of qubits. The first n rows
    represent the stabilizers, and the last n rows represent the destabilizers.
    The first n columns represent the X operators, and the last n columns
    represent the Z operators. The (i, j) entry of the matrix is 1 if the
    operator in row i anticommutes with the operator in column j, and 0
    otherwise. The sign of the operator in row i is given by the i-th entry of
    the sign vector. The sign of the operator in column j is given by the
    (n + j)-th entry of the sign vector.

    A more readable representation of the tableau is given by the string:
    ```python
    >>> from pauliopt.clifford.tableau import CliffordTableau
    >>> ct = CliffordTableau(2)
    >>> print(ct)
    # Expected Output:
    # X/Z I/I | +
    # I/I X/Z | +
    >>> ct.append_h(0)
    >>> print(ct)
    # Expected Output:
    # Z/X I/I | +
    # I/I X/Z | +
    ```

    To get the raw $2n \times 2n$ matrix representation of the tableau, use:
    ```python
    >>> ct.tableau
    # Expected Output:
    # array([[0, 0, 1, 0],
    #        [0, 1, 0, 0],
    #        [1, 0, 0, 0],
    #        [0, 0, 0, 1]], dtype=uint8)
    ```
    To get the sign vector, use:
    ```python
    >>> ct.signs
    # Expected Output:
    # array([0, 0, 0, 0], dtype=uint8)
    ```

    """

    def __init__(self, n_qubits):
        self.tableau = np.eye(2 * n_qubits, dtype=np.uint8)
        self.signs = np.zeros((2 * n_qubits), dtype=np.uint8)
        self.n_qubits = n_qubits

    def __str__(self) -> str:
        out = str(self.string_repr(sep=" "))
        return out

    @staticmethod
    def from_tableau(tableau, signs):
        """
        Create a CliffordTableau from a tableau and sign vector.

        Args:
            tableau (np.ndarray): $2n \times 2n$ binary matrix representing the tableau.
            signs (np.ndarray): $2n$-dimensional binary vector representing the sign vector.

        Returns:
            CliffordTableau: CliffordTableau object.
        """
        n_qubits = tableau.shape[0] // 2
        if not (tableau.shape == (2 * n_qubits, 2 * n_qubits) and signs.shape == (
                2 * n_qubits,)):
            raise ValueError("Tableau and signs must have shape "
                             "(2 * n_qubits, 2 * n_qubits) and (2 * n_qubits,)")
        ct = CliffordTableau(n_qubits)
        ct.tableau = tableau
        ct.signs = signs
        return ct

    @staticmethod
    def from_qiskit_tableau(qiskit_ct: "qiskit.quantum_info.Clifford"):
        """
        Create a CliffordTableau from a qiskit Clifford object.

        Args:
            qiskit_ct (qiskit.quantum_info.Clifford): Clifford object.

        Returns:
            CliffordTableau: CliffordTableau object.
        """
        n_qubits = qiskit_ct.num_qubits
        ct = CliffordTableau(n_qubits)
        ct.tableau = qiskit_ct.symplectic_matrix
        ct.signs = qiskit_ct.phase
        return ct

    def string_repr(self, sep=" ", sign_sep="| "):
        """
        Get a string representation of the tableau.

        Args:
            sep (str): Separator between the pauli operators
            sign_sep (str): Separator between operators and sign.

        """
        out = ""
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                x_str = ["I", "X", "Z", "Y"][int(self.x_out(i, j))]
                z_str = ["I", "X", "Z", "Y"][int(self.z_out(i, j))]
                out += f"{x_str}/{z_str}" + sep
            out += sign_sep + f"{'+' if self.signs[i] == 0 else '-'} \n"
        return out

    def x_out(self, row, col):
        """
        Get the X operator in row `row` and column `col`.

        Args:
            row (int): Row index.
            col (int): Column index.
        """
        return self.tableau[row, col] + \
               2 * self.tableau[row, col + self.n_qubits]

    def z_out(self, row, col):
        """
        Get the Z operator in row `row` and column `col`.

        Args:
            row (int): Row index.
            col (int): Column index.
        """
        return self.tableau[row + self.n_qubits, col] + \
               2 * self.tableau[row + self.n_qubits, col + self.n_qubits]

    def prepend_h(self, qubit):
        """
        Prepend a Hadamard gate to the tableau.

        Args:
            qubit (int): Qubit the hadamard gate is applied to.
        """
        self.signs[[qubit, self.n_qubits + qubit]] = self.signs[
            [self.n_qubits + qubit, qubit]]
        self.tableau[[self.n_qubits + qubit, qubit], :] = \
            self.tableau[[qubit, self.n_qubits + qubit], :]

    def append_h(self, qubit):
        """
        Append a Hadamard gate to the tableau.

        Args:
            qubit (int): Qubit the hadamard gate is applied to.
        """
        self.signs = (self.signs + self.tableau[:, qubit] * self.tableau[:,
                                                            self.n_qubits + qubit]) % 2

        self.tableau[:, [self.n_qubits + qubit, qubit]] = self.tableau[:,
                                                          [qubit, self.n_qubits + qubit]]

    def prepend_s(self, qubit):
        """
        Prepend a S gate to the tableau.

        Args:
            qubit (int): Qubit the S gate is applied to.
        """
        stabilizer = self.tableau[qubit, :]
        destabilizer = self.tableau[qubit + self.n_qubits, :]
        stab_sign = self.signs[qubit]
        destab_sign = self.signs[qubit + self.n_qubits]

        destabilizer, destab_sign = \
            mult_paulis(stabilizer, destabilizer, stab_sign, destab_sign, self.n_qubits)
        self.insert_pauli_row(destabilizer, destab_sign, qubit)

    def append_s(self, qubit):
        """
        Append a S gate to the tableau.

        Args:
            qubit (int): Qubit the S gate is applied to.
        """
        self.signs = (self.signs + self.tableau[:, qubit] *
                      self.tableau[:, self.n_qubits + qubit]) % 2

        self.tableau[:, self.n_qubits + qubit] = (self.tableau[:, self.n_qubits + qubit] +
                                                  self.tableau[:, qubit]) % 2

    def prepend_cnot(self, control, target):
        """
        Prepend a CNOT gate to the tableau.

        Args:
            control (int): Control qubit.
            target (int): Target qubit.
        """
        stab_ctrl = self.tableau[control, :]
        stab_target = self.tableau[target, :]
        stab_sign_ctrl = self.signs[control]
        stab_sign_target = self.signs[target]

        destab_ctrl = self.tableau[control + self.n_qubits, :]
        destab_target = self.tableau[target + self.n_qubits, :]
        destab_sign_ctrl = self.signs[control + self.n_qubits]
        destab_sign_target = self.signs[target + self.n_qubits]

        stab_ctrl, stab_sign_ctrl = \
            mult_paulis(stab_ctrl, stab_target, stab_sign_ctrl, stab_sign_target,
                        self.n_qubits)

        destab_target, destab_sign_target = \
            mult_paulis(destab_target, destab_ctrl, destab_sign_target, destab_sign_ctrl,
                        self.n_qubits)

        self.insert_pauli_row(stab_ctrl, stab_sign_ctrl, control)
        self.insert_pauli_row(destab_target, destab_sign_target, target + self.n_qubits)

    def append_cnot(self, control, target):
        """
        Append a CNOT gate to the tableau.

        Args:
            control (int): Control qubit.
            target (int): Target qubit.
        """
        x_ia = self.tableau[:, control]
        x_ib = self.tableau[:, target]

        z_ia = self.tableau[:, self.n_qubits + control]
        z_ib = self.tableau[:, self.n_qubits + target]

        self.tableau[:, target] = \
            (self.tableau[:, target] + self.tableau[:, control]) % 2
        self.tableau[:, self.n_qubits + control] = \
            (self.tableau[:, self.n_qubits + control] + self.tableau[:,
                                                        self.n_qubits + target]) % 2

        tmp_sum = ((x_ib + z_ia) % 2 + np.ones(z_ia.shape)) % 2
        self.signs = (self.signs + x_ia * z_ib * tmp_sum) % 2

    def insert_pauli_row(self, pauli, p_sing, row):
        """
        Insert a Pauli row into the tableau.

        Args:
            pauli (np.array): Pauli to be inserted.
            p_sing (int): Sign of the Pauli.
            row (int): Row to insert the Pauli.

        """
        for i in range(self.n_qubits):
            if (self.tableau[row, i] + pauli[i]) % 2 == 1:
                self.tableau[row, i] = (self.tableau[row, i] + 1) % 2

            if (self.tableau[row, i + self.n_qubits] + pauli[i + self.n_qubits]) % 2 == 1:
                self.tableau[row, i + self.n_qubits] = (self.tableau[
                                                            row, i + self.n_qubits] + 1) % 2
        if (self.signs[row] + p_sing) % 2 == 1:
            self.signs[row] = (self.signs[row] + 1) % 2

    def inverse(self):
        """
        Invert the tableau.


        Note: this is will create a deep copy of the tableau.

        Returns:
            CliffordTableau: Inverted tableau.

        """
        n_qubits = self.n_qubits

        x2x = self.tableau[:n_qubits, :n_qubits].copy()
        z2z = self.tableau[n_qubits:2 * n_qubits, n_qubits:2 * n_qubits].copy()

        x2z = self.tableau[:n_qubits, n_qubits:2 * n_qubits].copy()
        z2x = self.tableau[n_qubits:2 * n_qubits, :n_qubits].copy()

        top_row = np.hstack((z2z.T, x2z.T))
        bottom_row = np.hstack((z2x.T, x2x.T))
        new_tableau = np.vstack((top_row, bottom_row))

        ct_new = CliffordTableau.from_tableau(new_tableau, self.signs.copy())

        ct_intermediate = self.apply(ct_new)

        ct_new.signs = (ct_new.signs + ct_intermediate.signs) % 2
        return ct_new

    def apply(self, other: "CliffordTableau"):
        """
        Apply a CliffordTableau to the current tableau.

        Note: this is will create a deep copy of the tableau.

        Args:
            other (CliffordTableau): CliffordTableau to apply.

        Returns:
            CliffordTableau: Applied CliffordTableau.
        """
        new_tableau = np.dot(self.tableau, other.tableau) % 2

        phase = np.mod(other.tableau.dot(self.signs) + other.signs, 2)

        # Correcting for phase due to Pauli multiplication
        ifacts = np.zeros(2 * self.n_qubits, dtype=int)

        for k in range(2 * self.n_qubits):

            row2 = other.tableau[k]
            x2 = other.tableau[k, 0:self.n_qubits]
            z2 = other.tableau[k, self.n_qubits:2 * self.n_qubits]

            # Adding a factor of i for each Y in the image of an operator under the
            # first operation, since Y=iXZ

            ifacts[k] += np.sum(x2 * z2)

            # Adding factors of i due to qubit-wise Pauli multiplication

            for j in range(self.n_qubits):
                x = 0
                z = 0
                for i in range(2 * self.n_qubits):
                    if row2[i]:
                        x1 = self.tableau[i, j]
                        z1 = self.tableau[i, j + self.n_qubits]
                        if (x == 1 or z == 1) and (x1 == 1 or z1 == 1):
                            val = np.mod(np.abs(3 * z1 - x1) - np.abs(3 * z - x) - 1, 3)
                            if val == 0:
                                ifacts[k] += 1
                            elif val == 1:
                                ifacts[k] -= 1
                        x = np.mod(x + x1, 2)
                        z = np.mod(z + z1, 2)

        p = np.mod(ifacts, 4) // 2

        phase = np.mod(phase + p, 2)

        return CliffordTableau.from_tableau(new_tableau, phase)
