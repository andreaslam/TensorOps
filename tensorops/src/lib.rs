pub mod helpers;
pub mod kernel;
pub mod runtime;
pub mod tensor;

use helpers::get_predefined_kernel_source;
use kernel::{KernelResult, KernelTensorOps, KernelType, LogicalInputSource, PredefinedKernel};
use pyo3::prelude::*;
use runtime::Runtime;
use tensor::*;

#[pymodule]
fn tensorops_backend(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Runtime>()?;
    m.add_class::<KernelTensorOps>()?;
    m.add_class::<KernelType>()?;
    m.add_class::<KernelResult>()?;
    m.add_class::<PredefinedKernel>()?;
    m.add_class::<LogicalInputSource>()?;
    m.add_function(wrap_pyfunction!(get_predefined_kernel_source, m)?)?;
    m.add_function(wrap_pyfunction!(get_predefined_kernel_source, m)?)?;
    m.add_function(wrap_pyfunction!(get_shape, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_from_list, m)?)?;

    m.add(
        "__doc__",
        "A Rust backend for executing OpenCL kernel graphs using PyO3.",
    )?;
    Ok(())
}
