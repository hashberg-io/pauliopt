import numpy as np


class CliffordTableau:
    def __init__(self, n_qubits: int = None, tableau: np.array = None):
        if n_qubits is None and tableau is None:
            raise Exception("Either Tableau or number of qubits must be defined")
        if tableau is None:
            self.tableau = np.eye(2 * n_qubits)
            self.n_qubits = n_qubits
        else:
            if tableau.shape[1] == tableau.shape[1]:
                raise Exception("Must be a 2nx2n Tableau!")
            self.tableau_x = tableau
            self.n_qubits = int(tableau.shape[1] / 2.0)

    def apply_h(self, qubit):
        pass
