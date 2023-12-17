use pyo3::prelude::*;


mod tableau;
use tableau::*;
	

pub fn register_clifford(_py: Python, module: &PyModule) -> PyResult<()> {
	let child_module = PyModule::new(_py, "clifford")?;
	
	child_module.add_class::<CliffordTableau>()?;
	
	
	module.add_submodule(child_module)?;
	Ok(())
}	


