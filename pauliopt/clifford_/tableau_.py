import numpy as np


def mult_paulis(p1, p2, sign1, sign2, n_qubits):
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
    def __init__(self, n_qubits):
        self.tableau = np.eye(2*n_qubits)
        self.signs = np.zeros((2*n_qubits))
        self.n_qubits = n_qubits

    @classmethod
    def register(cls, rust_class):
        cls.factorial = rust_class.factorial

    @staticmethod
    def identity(n_qubits: int):
        return CliffordTableau(np.eye(2 * n_qubits), np.zeros((2 * n_qubits)))

    def __str__(self) -> str:
        out = "T: \n"
        out += str(self.string_repr) + "\n"
        return out

    @property
    def string_repr(self):
        out = ""
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                x_str = ["I", "X", "Z", "Y"][int(self.x_out(i, j))]
                z_str = ["I", "X", "Z", "Y"][int(self.z_out(i, j))]
                out += f"{x_str}/{z_str}" + " "
            out += "\n"
        return out

    def x_out(self, row, col):
        return self.tableau[row, col] + \
               2 * self.tableau[row, col + self.n_qubits]

    def z_out(self, row, col):
        return self.tableau[row + self.n_qubits, col] + \
               2 * self.tableau[row + self.n_qubits, col + self.n_qubits]

    def prepend_h(self, qubit):
        self.signs[[qubit, self.n_qubits + qubit]] = \
            self.signs[[self.n_qubits + qubit, qubit]]
        self.tableau[[self.n_qubits + qubit, qubit], :] = \
            self.tableau[[qubit, self.n_qubits + qubit], :]

    def append_h(self, qubit):
        self.signs = (self.signs + self.tableau[:, qubit] * self.tableau[:,
                                                            self.n_qubits + qubit]) % 2

        self.tableau[:, [self.n_qubits + qubit, qubit]] = self.tableau[:,
                                                          [qubit, self.n_qubits + qubit]]

    def prepend_s(self, qubit):
        stabilizer = self.tableau[qubit, :]
        destabilizer = self.tableau[qubit + self.n_qubits, :]
        stab_sign = self.signs[qubit]
        destab_sign = self.signs[qubit + self.n_qubits]

        destabilizer, destab_sign = \
            mult_paulis(stabilizer, destabilizer, stab_sign, destab_sign, self.n_qubits)
        self.insert_pauli_row(destabilizer, destab_sign, qubit)

    def append_s(self, qubit):
        self.signs = (self.signs + self.tableau[:, qubit] *
                      self.tableau[:, self.n_qubits + qubit]) % 2

        self.tableau[:, self.n_qubits + qubit] = (self.tableau[:, self.n_qubits + qubit] +
                                                  self.tableau[:, qubit]) % 2

    def prepend_cnot(self, control, target):
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
        for i in range(self.n_qubits):
            if (self.tableau[row, i] + pauli[i]) % 2 == 1:
                self.tableau[row, i] = (self.tableau[row, i] + 1) % 2

            if (self.tableau[row, i + self.n_qubits] + pauli[i + self.n_qubits]) % 2 == 1:
                self.tableau[row, i + self.n_qubits] = (self.tableau[
                                                            row, i + self.n_qubits] + 1) % 2
        if (self.signs[row] + p_sing) % 2 == 1:
            self.signs[row] = (self.signs[row] + 1) % 2
