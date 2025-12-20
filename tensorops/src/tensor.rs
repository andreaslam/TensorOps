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

    let byte_array = PyByteArray::new(py, byte_slice);

    Ok((byte_array.into_py(py), shape))
}

#[pyfunction]
pub fn tensor_max<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    shape: Vec<usize>,
    axis: Option<isize>,
) -> PyResult<(PyObject, Vec<usize>)> {
    let (result, out_shape) = match axis {
        None => {
            // Global max
            let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (vec![max_val], vec![])
        }
        Some(ax) => {
            // Normalise axis to positive index
            let ndim = shape.len() as isize;
            let axis_idx = if ax < 0 {
                (ndim + ax) as usize
            } else {
                ax as usize
            };

            if axis_idx >= shape.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "axis {} out of bounds for tensor of dimension {}",
                    ax,
                    shape.len()
                )));
            }

            reduce_along_axis(&data, &shape, axis_idx, f32::NEG_INFINITY, f32::max)?
        }
    };

    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr() as *const u8,
            result.len() * std::mem::size_of::<f32>(),
        )
    };
    let byte_array = PyByteArray::new(py, byte_slice);
    Ok((byte_array.into_py(py), out_shape))
}

/// Compute min along an axis or across entire tensor
#[pyfunction]
pub fn tensor_min<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    shape: Vec<usize>,
    axis: Option<isize>,
) -> PyResult<(PyObject, Vec<usize>)> {
    let (result, out_shape) = match axis {
        None => {
            // Global min
            let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
            (vec![min_val], vec![])
        }
        Some(ax) => {
            // Normalise axis to positive index
            let ndim = shape.len() as isize;
            let axis_idx = if ax < 0 {
                (ndim + ax) as usize
            } else {
                ax as usize
            };

            if axis_idx >= shape.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "axis {} out of bounds for tensor of dimension {}",
                    ax,
                    shape.len()
                )));
            }

            reduce_along_axis(&data, &shape, axis_idx, f32::INFINITY, f32::min)?
        }
    };

    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr() as *const u8,
            result.len() * std::mem::size_of::<f32>(),
        )
    };
    let byte_array = PyByteArray::new(py, byte_slice);
    Ok((byte_array.into_py(py), out_shape))
}

#[pyfunction]
pub fn tensor_expand<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    src_shape: Vec<usize>,
    tgt_shape: Vec<usize>,
) -> PyResult<PyObject> {
    if src_shape.len() != tgt_shape.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Expand requires same number of dimensions, got src_shape={:?}, tgt_shape={:?}",
            src_shape, tgt_shape
        )));
    }
    let ndim = src_shape.len();
    if ndim == 0 {
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        let byte_array = PyByteArray::new(py, byte_slice);
        return Ok(byte_array.into_py(py));
    }

    for i in 0..ndim {
        let s = src_shape[i];
        let t = tgt_shape[i];
        if s != 1 && s != t {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot expand {:?} to {:?}",
                src_shape, tgt_shape
            )));
        }
    }

    let src_size: usize = src_shape.iter().product();
    if src_size != data.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Data length {} does not match src_shape {:?} (expected {})",
            data.len(),
            src_shape,
            src_size
        )));
    }
    if src_shape == tgt_shape {
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        let byte_array = PyByteArray::new(py, byte_slice);
        return Ok(byte_array.into_py(py));
    }

    let tgt_size: usize = tgt_shape.iter().product();
    let mut out = vec![0.0f32; tgt_size];

    // Row-major strides
    let mut src_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }
    let mut tgt_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        tgt_strides[i] = tgt_strides[i + 1] * tgt_shape[i + 1];
    }

    for out_idx in 0..tgt_size {
        let mut src_flat_idx = 0usize;
        for dim in 0..ndim {
            let coord = (out_idx / tgt_strides[dim]) % tgt_shape[dim];
            let src_coord = if src_shape[dim] == 1 { 0 } else { coord };
            src_flat_idx += src_coord * src_strides[dim];
        }
        out[out_idx] = data[src_flat_idx];
    }

    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            out.as_ptr() as *const u8,
            out.len() * std::mem::size_of::<f32>(),
        )
    };
    let byte_array = PyByteArray::new(py, byte_slice);
    Ok(byte_array.into_py(py))
}

/// Generic reduction along a specific axis
fn reduce_along_axis(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    init_val: f32,
    op: fn(f32, f32) -> f32,
) -> PyResult<(Vec<f32>, Vec<usize>)> {
    // Calculate output shape (remove the axis dimension)
    let mut out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &dim)| dim)
        .collect();

    if out_shape.is_empty() {
        out_shape = vec![1];
    }

    let out_size: usize = out_shape.iter().product();
    let mut result = vec![init_val; out_size];

    // Calculate strides for input tensor
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Iterate through all elements
    for (idx, &val) in data.iter().enumerate() {
        // Convert flat index to multi-dimensional coordinates
        let mut coords = vec![0; shape.len()];
        let mut remaining = idx;
        for i in 0..shape.len() {
            coords[i] = remaining / strides[i];
            remaining %= strides[i];
        }

        // Calculate output index (excluding the reduction axis)
        let mut out_idx = 0;
        let mut out_stride = 1;
        for i in (0..shape.len()).rev() {
            if i != axis {
                out_idx += coords[i] * out_stride;
                out_stride *= shape[i];
            }
        }

        // Apply reduction operation
        result[out_idx] = op(result[out_idx], val);
    }

    Ok((result, out_shape))
}
