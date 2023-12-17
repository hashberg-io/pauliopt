use pyo3::prelude::*;

extern crate nalgebra as na;
use na::{Dyn, OMatrix, U1};

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};

type Tableau = OMatrix<u8, Dyn, Dyn>;
type Signs = OMatrix<u8, U1, Dyn>;



#[pyclass]
#[derive(Clone)]
pub struct CliffordTableau {
    n_qubits: usize,
    tableau: Tableau,
    signs: Signs,
}

#[pymethods]
impl CliffordTableau {
    #[new]
    fn new(n_qubits: usize) -> Self {
        let tableau_size = 2* n_qubits;
        let tableau: Tableau = Tableau::identity(tableau_size, tableau_size);
        let signs: Signs = Signs::zeros(tableau_size);
        CliffordTableau {
            n_qubits: n_qubits,
            tableau: tableau,
            signs: signs,
        }
    }

    fn __str__(&self) -> PyResult<String> {
        let mut result = String::new();
        for i in 0..self.n_qubits {
            for j in 0..self.n_qubits {
                let destab = ["I", "X", "Z", "Y"][self.x_out(i, j) as usize];
                let stab = ["I", "X", "Z", "Y"][self.z_out(i, j) as usize];
                result.push_str(&destab);
                result.push_str("/");
                result.push_str(&stab);
                result.push_str(" ");
            }
            result.push_str("|");
            result.push_str(" ");
            let destab_sign = ["+", "-"][self.x_sign(i) as usize];
            let stab_sign = ["+", "-"][self.z_sign(i) as usize];
            result.push_str(&destab_sign);
            result.push_str("/");
            result.push_str(&stab_sign);
            result.push_str("\n"); 
        }
        
        Ok(result)
    }

    fn append_h(&mut self, qubit: usize) {
        for i in 0..2*self.n_qubits {
            self.signs[i] = (self.signs[i] + self.tableau[(i, qubit)]*self.tableau[(i, qubit+self.n_qubits)]) % 2;
            let tmp = self.tableau[(i, qubit)];
            self.tableau[(i, qubit)] = self.tableau[(i, qubit+self.n_qubits)];
            self.tableau[(i, qubit+self.n_qubits)] = tmp;
            
        }

    }

    fn x_sign(&self, row: usize) -> u8 {
        return self.signs[row];
    }

    fn z_sign(&self, row: usize) -> u8 {
        return self.signs[row+self.n_qubits];
    }


    fn x_out(&self, row: usize, col: usize) -> u8 {
        return self.tableau[(row, col)] + 2* self.tableau[(row, col+self.n_qubits)]
    }

    fn z_out(&self, row:usize, col:usize) -> u8 {
        return self.tableau[(row+self.n_qubits, col)] + 2* self.tableau[(row+self.n_qubits, col+self.n_qubits)]
    }


    fn get_tableau(&self, py: Python) -> Py<PyArray2<u8>> {
        let mut matrix = Array2::<u8>::zeros((2*self.n_qubits, 2*self.n_qubits));
        for i in 0..2*self.n_qubits {
            for j in 0..2*self.n_qubits {
                matrix[[i, j]] = self.tableau[(i, j)];
            }
        }
        matrix.into_pyarray(py).to_owned()
    }

    fn get_signs(&self, py: Python) -> Py<PyArray2<u8>> {
        let mut matrix = Array2::<u8>::zeros((2*self.n_qubits, 1));
        for i in 0..2*self.n_qubits {
            matrix[(i, 0)] = self.signs[i];
        }
        matrix.into_pyarray(py).to_owned()
    }

}
