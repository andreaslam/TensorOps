use crate::helpers::get_predefined_kernel_source;
use core::fmt;
use ocl::{flags as OclFlags, Buffer, Queue};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[derive(Debug, Clone)]
pub enum ResolvedInput {
    Host(Vec<f32>),
    Device(Buffer<f32>),
}

impl ResolvedInput {
    fn len(&self) -> usize {
        match self {
            ResolvedInput::Host(v) => v.len(),
            ResolvedInput::Device(b) => b.len(),
        }
    }

    fn scalar0(&self) -> Result<f32, PyErr> {
        match self {
            ResolvedInput::Host(v) => v
                .first()
                .copied()
                .ok_or_else(|| PyValueError::new_err("Expected scalar input (len 1)")),
            ResolvedInput::Device(_) => Err(PyValueError::new_err(
                "Scalar inputs must be provided as host vectors (len 1)",
            )),
        }
    }
}

#[pyclass(eq, module = "tensorops_backend")]
#[derive(Debug, PartialEq, Clone)]
pub struct KernelResult {
    #[pyo3(get)]
    pub val: Vec<Vec<f32>>,
    #[pyo3(get)]
    pub kernel_id: usize,
}

#[pymethods]
impl KernelResult {
    #[new]
    fn new(kernel_id: usize, val: Vec<Vec<f32>>) -> Self {
        KernelResult { val, kernel_id }
    }
    fn __repr__(&self) -> String {
        format!(
            "KernelResult(kernel_id={}, val_len={})",
            self.kernel_id,
            self.val.len()
        )
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
    fn tolist(&self, py: Python) -> Py<PyList> {
        // Helper to convert Vec<Vec<f32>> to PyList of PyLists
        let outer_list = PyList::empty(py);
        for inner_vec in &self.val {
            let inner_pylist = PyList::new(py, inner_vec.iter().map(|&x| x.to_object(py))).unwrap();
            outer_list.append(inner_pylist).unwrap();
        }
        outer_list.into()
    }
}

// --- PredefinedKernel & KernelType (no change) ---
#[pyclass(eq, eq_int, module = "tensorops_backend")]
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum PredefinedKernel {
    VecAdd,
    VecSub,
    VecElementMul,
    VecDiv,
    VecPow,
    VecLog,
    VecSin,
    VecCos,
    VecTan,
    VecAbs,
    VecTanh,
    VecLeakyReLU,
    VecSum,
    VecMax,
    VecMin,
}
impl fmt::Display for PredefinedKernel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
#[pymethods]
impl PredefinedKernel {
    fn __repr__(&self) -> String {
        format!("PredefinedKernel.{}", self)
    }
    fn __str__(&self) -> String {
        self.to_string()
    }
}

#[pyclass(eq, module = "tensorops_backend")]
#[derive(Debug, PartialEq, Clone)]
pub enum KernelType {
    Predefined(PredefinedKernel),
    Custom(String),
}
impl fmt::Display for KernelType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KernelType::Predefined(pk) => write!(f, "Predefined({})", pk),
            KernelType::Custom(name) => write!(f, "Custom(\"{}\")", name),
        }
    }
}
#[pymethods]
impl KernelType {
    #[staticmethod]
    pub fn predefined(kernel: PredefinedKernel) -> Self {
        KernelType::Predefined(kernel)
    }
    #[staticmethod]
    pub fn custom(custom_name: String) -> Self {
        KernelType::Custom(custom_name)
    }
    fn __repr__(&self) -> String {
        match self {
            KernelType::Predefined(pk) => format!("KernelType.predefined({:?})", pk),
            KernelType::Custom(name) => format!("KernelType.custom(\"{}\")", name),
        }
    }
    fn __str__(&self) -> String {
        self.to_string()
    }
    #[getter]
    fn is_predefined(&self) -> bool {
        matches!(self, KernelType::Predefined(_))
    }
    #[getter]
    fn is_custom(&self) -> bool {
        matches!(self, KernelType::Custom(_))
    }
    #[getter]
    fn get_predefined_kernel<'py>(&self, _py: Python<'py>) -> PyResult<Option<PredefinedKernel>> {
        match self {
            KernelType::Predefined(pk) => Ok(Some(*pk)),
            KernelType::Custom(_) => Ok(None),
        }
    }
    #[getter]
    fn get_custom_name<'py>(&self, _py: Python<'py>) -> PyResult<Option<String>> {
        match self {
            KernelType::Predefined(_) => Ok(None),
            KernelType::Custom(name) => Ok(Some(name.clone())),
        }
    }
}

// --- LogicalInputSource (New Struct) ---
#[pyclass(eq, module = "tensorops_backend")]
#[derive(Debug, PartialEq, Clone)]
pub struct LogicalInputSource {
    #[pyo3(get)]
    pub source_kernel_id: usize,
    #[pyo3(get)]
    pub source_output_index: usize,
}

#[pymethods]
impl LogicalInputSource {
    #[new]
    fn new(source_kernel_id: usize, source_output_index: usize) -> Self {
        Self {
            source_kernel_id,
            source_output_index,
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "LogicalInputSource(kernel_id={}, output_idx={})",
            self.source_kernel_id, self.source_output_index
        )
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass(eq, module = "tensorops_backend")]
#[derive(Debug, PartialEq, Clone)]
pub struct DirectInput {
    #[pyo3(get)]
    pub data: Vec<f32>,
}

#[pymethods]
impl DirectInput {
    #[new]
    fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
    fn __repr__(&self) -> String {
        format!("DirectInput(len={})", self.data.len())
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass(module = "tensorops_backend")]
#[derive(Debug)]
pub struct KernelTensorOps {
    #[pyo3(get)]
    pub kernel_type: KernelType,
    #[pyo3(get)]
    pub kernel_id: usize,
    #[pyo3(get)]
    pub kernel_src: String,

    #[pyo3(get, set)]
    pub inputs: Option<Vec<PyObject>>,

    #[pyo3(get)]
    pub num_output_bufs: usize,
    pub scalar_inputs: Option<Vec<Vec<f32>>>,
}

impl Clone for KernelTensorOps {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            kernel_type: self.kernel_type.clone(),
            kernel_id: self.kernel_id,
            kernel_src: self.kernel_src.clone(),
            inputs: self
                .inputs
                .as_ref()
                .map(|v| v.iter().map(|x| x.clone_ref(py)).collect()),
            num_output_bufs: self.num_output_bufs,
            scalar_inputs: self.scalar_inputs.clone(),
        })
    }
}

#[pymethods]
impl KernelTensorOps {
    #[new]
    #[pyo3(signature = (kernel_type, kernel_id, num_output_bufs, custom_kernel_src=None, inputs=None, scalar_inputs=None))]
    pub fn new(
        kernel_type: KernelType,
        kernel_id: usize,
        num_output_bufs: usize,
        custom_kernel_src: Option<String>,
        inputs: Option<Vec<PyObject>>,
        scalar_inputs: Option<Vec<Vec<f32>>>,
    ) -> PyResult<Self> {
        let kernel_src = match &kernel_type {
            KernelType::Predefined(predefined_kernel) => {
                if custom_kernel_src.is_some() {
                    eprintln!("Warning: Custom source provided for PredefinedKernel {:?}, it will be ignored.", predefined_kernel);
                }
                get_predefined_kernel_source(&*predefined_kernel) // PredefinedKernel is Copy
                    .map(|s| s.to_string())
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "Failed to get predefined kernel source for {:?}",
                            predefined_kernel
                        ))
                    })?
            }
            KernelType::Custom(ref name) => custom_kernel_src.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "KernelType::Custom(\"{}\") requires custom_kernel_src.",
                    name
                ))
            })?,
        };

        Ok(KernelTensorOps {
            kernel_type,
            kernel_id,
            kernel_src,
            inputs,
            num_output_bufs,
            scalar_inputs,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "KernelTensorOps(kernel_id={}, type={}, inputs_set={}, num_outputs={})",
            self.kernel_id,
            self.kernel_type.__repr__(), // Use repr of KernelType
            self.inputs.is_some(),
            self.num_output_bufs
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl KernelTensorOps {
    pub fn prepare_ocl_buffers_from_resolved_inputs_any(
        &self,
        queue: &Queue,
        resolved_inputs: &[ResolvedInput],
    ) -> Result<(Vec<Buffer<f32>>, Vec<Buffer<f32>>, usize, Vec<f32>), PyErr> {
        let mut scalar_inputs = Vec::new();
        let mut input_ocl_buffers: Vec<Buffer<f32>> = Vec::new();

        let work_size_dims = if !resolved_inputs.is_empty() {
            if resolved_inputs[0].len() == 0 {
                return Err(PyValueError::new_err(format!(
                    "Kernel {}: Input 0 cannot be empty if other inputs exist.",
                    self.kernel_id
                )));
            }

            // VecLog: second input is scalar base
            if self.kernel_type == KernelType::Predefined(PredefinedKernel::VecLog)
                && resolved_inputs.len() >= 2
            {
                if resolved_inputs[1].len() == 1 {
                    scalar_inputs.push(resolved_inputs[1].scalar0()?);
                    // First input must be a buffer
                    input_ocl_buffers.push(match &resolved_inputs[0] {
                        ResolvedInput::Host(v) => Buffer::<f32>::builder()
                            .queue(queue.clone())
                            .flags(OclFlags::MEM_READ_ONLY | OclFlags::MEM_COPY_HOST_PTR)
                            .len(v.len())
                            .copy_host_slice(v.as_slice())
                            .build()
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!(
                                    "Kernel {}: Failed to create VecLog input buffer: {}",
                                    self.kernel_id, e
                                ))
                            })?,
                        ResolvedInput::Device(b) => b.clone(),
                    });
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Kernel {}: VecLog expects scalar base (len 1), got len {}.",
                        self.kernel_id,
                        resolved_inputs[1].len()
                    )));
                }
                input_ocl_buffers[0].len()
            }
            // VecLeakyReLU: second input is scalar alpha
            else if self.kernel_type == KernelType::Predefined(PredefinedKernel::VecLeakyReLU)
                && resolved_inputs.len() >= 2
            {
                if resolved_inputs[1].len() == 1 {
                    scalar_inputs.push(resolved_inputs[1].scalar0()?);
                    input_ocl_buffers.push(match &resolved_inputs[0] {
                        ResolvedInput::Host(v) => Buffer::<f32>::builder()
                            .queue(queue.clone())
                            .flags(OclFlags::MEM_READ_ONLY | OclFlags::MEM_COPY_HOST_PTR)
                            .len(v.len())
                            .copy_host_slice(v.as_slice())
                            .build()
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!(
                                    "Kernel {}: Failed to create VecLeakyReLU input buffer: {}",
                                    self.kernel_id, e
                                ))
                            })?,
                        ResolvedInput::Device(b) => b.clone(),
                    });
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Kernel {}: VecLeakyReLU expects scalar alpha (len 1), got len {}.",
                        self.kernel_id,
                        resolved_inputs[1].len()
                    )));
                }
                input_ocl_buffers[0].len()
            }
            // Reduce ops: [data, pre, axis_len, post]
            else if matches!(
                self.kernel_type,
                KernelType::Predefined(PredefinedKernel::VecSum)
                    | KernelType::Predefined(PredefinedKernel::VecMax)
                    | KernelType::Predefined(PredefinedKernel::VecMin)
            ) && resolved_inputs.len() >= 4
            {
                scalar_inputs.push(resolved_inputs[1].scalar0()?);
                scalar_inputs.push(resolved_inputs[2].scalar0()?);
                scalar_inputs.push(resolved_inputs[3].scalar0()?);

                input_ocl_buffers.push(match &resolved_inputs[0] {
                    ResolvedInput::Host(v) => Buffer::<f32>::builder()
                        .queue(queue.clone())
                        .flags(OclFlags::MEM_READ_ONLY | OclFlags::MEM_COPY_HOST_PTR)
                        .len(v.len())
                        .copy_host_slice(v.as_slice())
                        .build()
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "Kernel {}: Failed to create reduce input buffer: {}",
                                self.kernel_id, e
                            ))
                        })?,
                    ResolvedInput::Device(b) => b.clone(),
                });

                let pre = scalar_inputs[0] as usize;
                let post = scalar_inputs[2] as usize;
                pre * post
            } else {
                // Generic: treat every input as a buffer input.
                for (i, inp) in resolved_inputs.iter().enumerate() {
                    let len = inp.len();
                    if len == 0 {
                        return Err(PyValueError::new_err(format!(
                            "Kernel {}: Input {} is empty.",
                            self.kernel_id, i
                        )));
                    }
                    let buf = match inp {
                        ResolvedInput::Host(v) => Buffer::<f32>::builder()
                            .queue(queue.clone())
                            .flags(OclFlags::MEM_READ_ONLY | OclFlags::MEM_COPY_HOST_PTR)
                            .len(v.len())
                            .copy_host_slice(v.as_slice())
                            .build()
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!(
                                    "Kernel {}: Failed to create input buffer {}: {}",
                                    self.kernel_id, i, e
                                ))
                            })?,
                        ResolvedInput::Device(b) => b.clone(),
                    };
                    input_ocl_buffers.push(buf);
                }

                // Work size is defined by first input buffer length.
                input_ocl_buffers
                    .first()
                    .ok_or_else(|| PyValueError::new_err("No inputs"))?
                    .len()
            }
        } else if self.num_output_bufs > 0 {
            return Err(PyValueError::new_err(format!(
                "Kernel {}: Cannot infer output size without inputs.",
                self.kernel_id
            )));
        } else {
            0
        };

        if work_size_dims == 0 && (!input_ocl_buffers.is_empty() || self.num_output_bufs > 0) {
            return Err(PyValueError::new_err(format!(
                "Kernel {}: Work size is 0, but inputs/outputs exist.",
                self.kernel_id
            )));
        }

        let mut output_ocl_buffers = Vec::with_capacity(self.num_output_bufs);
        for i in 0..self.num_output_bufs {
            let ocl_buffer = Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(OclFlags::MEM_WRITE_ONLY)
                .len(work_size_dims)
                .build()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Kernel {}: Failed to create output buffer {}: {}",
                        self.kernel_id, i, e
                    ))
                })?;
            output_ocl_buffers.push(ocl_buffer);
        }

        Ok((
            input_ocl_buffers,
            output_ocl_buffers,
            work_size_dims,
            scalar_inputs,
        ))
    }

    pub fn prepare_ocl_buffers_from_resolved_inputs(
        &self,
        queue: &Queue,
        resolved_input_data: &Vec<Vec<f32>>,
    ) -> Result<(Vec<Buffer<f32>>, Vec<Buffer<f32>>, usize, Vec<f32>), PyErr> // Added Vec<f32> for scalars
    {
        let mut scalar_inputs = Vec::new();
        let mut buffer_inputs = Vec::new();

        let work_size_dims = if !resolved_input_data.is_empty() {
            if resolved_input_data[0].is_empty() {
                return Err(PyValueError::new_err(format!(
                    "Kernel {}: Input vector 0 for buffer creation cannot be empty if other inputs exist.", self.kernel_id
                )));
            }
            // For VecLog, the second input (base) is a scalar if it has length 1
            if self.kernel_type == KernelType::Predefined(PredefinedKernel::VecLog)
                && resolved_input_data.len() >= 2
            {
                if resolved_input_data[1].len() == 1 {
                    scalar_inputs.push(resolved_input_data[1][0]);
                    buffer_inputs.push(resolved_input_data[0].clone());
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Kernel {}: VecLog expects second input (base) to be a scalar (length 1), got length {}.",
                        self.kernel_id, resolved_input_data[1].len()
                    )));
                }
                buffer_inputs[0].len()
            } else if self.kernel_type == KernelType::Predefined(PredefinedKernel::VecLeakyReLU)
                && resolved_input_data.len() >= 2
            {
                if resolved_input_data[1].len() == 1 {
                    scalar_inputs.push(resolved_input_data[1][0]);
                    buffer_inputs.push(resolved_input_data[0].clone());
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Kernel {}: VecLeakyReLU expects second input (alpha) to be a scalar (length 1), got length {}.",
                        self.kernel_id,
                        resolved_input_data[1].len()
                    )));
                }
                buffer_inputs[0].len()
            } else if matches!(
                self.kernel_type,
                KernelType::Predefined(PredefinedKernel::VecSum)
                    | KernelType::Predefined(PredefinedKernel::VecMax)
                    | KernelType::Predefined(PredefinedKernel::VecMin)
            ) && resolved_input_data.len() >= 4
            {
                // Reduce ops expect:
                //   Input 0: data buffer (len = pre*axis*post)
                //   Input 1: pre_axis (scalar)
                //   Input 2: axis_len (scalar)
                //   Input 3: post_axis (scalar)
                scalar_inputs.push(resolved_input_data[1][0]);
                scalar_inputs.push(resolved_input_data[2][0]);
                scalar_inputs.push(resolved_input_data[3][0]);
                buffer_inputs.push(resolved_input_data[0].clone());

                let pre = resolved_input_data[1][0] as usize;
                let post = resolved_input_data[3][0] as usize;
                pre * post
            } else {
                buffer_inputs.extend(resolved_input_data.clone());
                buffer_inputs[0].len()
            }
        } else if self.num_output_bufs > 0 {
            return Err(PyValueError::new_err(format!(
                "Kernel {}: Cannot determine buffer dimensions for outputs with no resolved input data to infer shape.", self.kernel_id
            )));
        } else {
            0
        };

        if work_size_dims == 0 && (!buffer_inputs.is_empty() || self.num_output_bufs > 0) {
            return Err(PyValueError::new_err(format!(
                "Kernel {}: Work size dimension is 0, but inputs/outputs exist.",
                self.kernel_id
            )));
        }

        let mut input_ocl_buffers = Vec::with_capacity(buffer_inputs.len());
        for (i, vec_data) in buffer_inputs.iter().enumerate() {
            // For reduce ops, the input buffer (index 0) is larger than the work size (output size).
            // So we skip the length check for it.
            let is_reduce_input = matches!(
                self.kernel_type,
                KernelType::Predefined(PredefinedKernel::VecSum)
                    | KernelType::Predefined(PredefinedKernel::VecMax)
                    | KernelType::Predefined(PredefinedKernel::VecMin)
            ) && i == 0;

            if !is_reduce_input && vec_data.len() != work_size_dims {
                return Err(PyValueError::new_err(format!(
                    "Kernel {}: Input vector {} length mismatch: expected {}, got {}.",
                    self.kernel_id,
                    i,
                    work_size_dims,
                    vec_data.len()
                )));
            }
            let ocl_buffer = Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(OclFlags::MEM_READ_ONLY | OclFlags::MEM_COPY_HOST_PTR)
                .len(vec_data.len())
                .copy_host_slice(vec_data.as_slice())
                .build()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Kernel {}: Failed to create input buffer {}: {}",
                        self.kernel_id, i, e
                    ))
                })?;
            input_ocl_buffers.push(ocl_buffer);
        }

        let mut output_ocl_buffers = Vec::with_capacity(self.num_output_bufs);
        for i in 0..self.num_output_bufs {
            let ocl_buffer = Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(OclFlags::MEM_WRITE_ONLY)
                .len(work_size_dims)
                .build()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Kernel {}: Failed to create output buffer {}: {}",
                        self.kernel_id, i, e
                    ))
                })?;
            output_ocl_buffers.push(ocl_buffer);
        }
        Ok((
            input_ocl_buffers,
            output_ocl_buffers,
            work_size_dims,
            scalar_inputs,
        ))
    }
}
