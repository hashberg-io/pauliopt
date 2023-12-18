use pyo3::prelude::*;
use ndarray::{prelude::*};

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray2, PyArray1};



#[pyclass]
#[derive(Clone)]
pub mut struct CliffordTableau {
    n_qubits: usize,
    tableau: Array2<u8>,
    signs: Array1<u8>,
}

#[pymethods]
impl CliffordTableau {
    #[new]
    fn new(n_qubits: usize) -> Self {
        let tableau = Array::eye(2*n_qubits);//.mapv(|x:u8| x != 0);
        let signs = Array::zeros(2*n_qubits);//.mapv(|x:u8| x != 0);
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
        self.signs.swap(qubit, self.n_qubits + qubit);


        // Get mutable slices for the rows you want to swap
        let (mut x_column, mut z_column) = self.tableau.multi_slice_mut((s![.., qubit], s![.., qubit + self.n_qubits]));

        ndarray::Zip::from(&mut x_column).and(&mut z_column).for_each(std::mem::swap);


    }

    fn append_s(&mut self, qubit: usize) {
        let (mut x_column, mut z_column) = self.tableau.multi_slice_mut((s![.., qubit], s![.., qubit + self.n_qubits]));let new_stabilizer_col = &x_column ^ &z_column;

        let mut signs = (self.signs + (&x_column * &z_column)) % 2;
        

        
        self.tableau.slice_mut(s![.., qubit+self.n_qubits]).assign(&new_stabilizer_col);

    }

    fn append_cnot(&mut self, control: usize, target: usize) {
        // let x_ia = self.tableau.slice(s![control, ..]).to_owned();
        // let x_ib= self.tableau.slice(s![target, ..]).to_owned();

        // let z_ia = self.tableau.slice(s![control + self.n_qubits, ..]).to_owned();
        // let z_ib = self.tableau.slice(s![target + self.n_qubits, ..]).to_owned();


        // let new_x_col = (&x_ia + &x_ib) & 2;
        // let new_z_col = (&z_ia + &z_ib) % 2;
        // self.tableau.slice_mut(s![target, ..]).assign(&new_x_col);
        // self.tableau.slice_mut(s![control+self.n_qubits, ..]).assign(&new_z_col);

        // let ones = Array1::<u8>::ones(2*self.n_qubits);
        // let tmp_sum = (&x_ib + &z_ia + &ones) % 2;
        // let old_signs = self.signs.to_owned();
        // self.signs = (&old_signs + &x_ia * &z_ib * &tmp_sum) % 2
        
    }

    fn x_sign(&self, row: usize) -> u8 {
        return self.signs[row] as u8;
    }

    fn z_sign(&self, row: usize) -> u8 {
        return self.signs[row+self.n_qubits] as u8;
    }


    fn x_out(&self, row: usize, col: usize) -> u8 {
        let x = self.tableau[[row, col]] as u8;
        let z = self.tableau[[row, col+self.n_qubits]] as u8;
        return x + 2 * z;
    }

    fn z_out(&self, row:usize, col:usize) -> u8 {
        let x = self.tableau[[row+self.n_qubits, col]] as u8;
        let z = self.tableau[[row+self.n_qubits, col+self.n_qubits]] as u8;
        return x + 2 * z;
    }


    fn get_tableau(&self, py: Python) -> Py<PyArray2<u8>> {
        let matrix = self.tableau.clone().mapv(|x:u8| x as u8);
        matrix.into_pyarray(py).to_owned()
    }

    fn get_signs(&self, py: Python) -> Py<PyArray1<u8>> {
        let matrix = self.signs.clone().mapv(|x:u8| x as u8);
        matrix.into_pyarray(py).to_owned()
    }

}
