use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyList, PyMemoryView};
use pyo3::Bound;

#[pyfunction]
pub fn get_shape<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<usize>> {
    let mut shape = Vec::new();
    let mut current = obj.clone();

    while let Ok(list) = current.downcast::<PyList>() {
        let len = list.len();
        shape.push(len);
        if len == 0 {
            break;
        }
        current = list.get_item(0)?;
    }

    Ok(shape)
}

/// Recursively flattens a nested list of floats into a Vec<f32>
fn flatten_recursive<'py>(obj: &Bound<'py, PyAny>, out: &mut Vec<f32>) -> PyResult<()> {
    if let Ok(list) = obj.downcast::<PyList>() {
        for item in list.iter() {
            flatten_recursive(&item, out)?; 
        }
    } else {
        out.push(obj.extract::<f32>()?);
    }
    Ok(())
}

#[pyfunction]
pub fn tensor_from_list<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(PyObject, Vec<usize>)> {
    let shape = get_shape(obj)?;
    let mut flat = Vec::new();
    flatten_recursive(obj, &mut flat)?;

    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            flat.as_ptr() as *const u8,
            flat.len() * std::mem::size_of::<f32>(),
        )
    };

    let list = PyList::new(py, &flat)?.into_py(py);

    Ok((list, shape))
}
