use pyo3::prelude::*;

mod clifford;
use clifford::register_clifford;




#[pymodule]
fn pauliopt(_py: Python, module: &PyModule) -> PyResult<()> {
    register_clifford(_py, module)?;
    Ok(())
}
