use crate::kernel::{
    DirectInput, KernelResult, KernelTensorOps, KernelType, LogicalInputSource, PredefinedKernel,
    ResolvedInput,
};
use ocl::Buffer;
use ocl::{
    core::QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, Context, Device, Error as OclError,
    Kernel as OclKernel, Platform, Program, Queue,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use pyo3::{ffi, AsPyPointer};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

fn ocl_py_err(e: OclError, context: &str) -> PyErr {
    PyRuntimeError::new_err(format!("OpenCL Error [{}]: {}", context, e))
}

#[pyclass(name = "Runtime", module = "tensorops_backend")]
#[derive(Debug)]
pub struct Runtime {
    context: Context,
    device: Device,
    queue: Queue,
    program_cache: HashMap<String, Program>,
}

#[pymethods]
impl Runtime {
    #[new]
    pub fn new() -> PyResult<Self> {
        let platform = Platform::default();
        let device = Device::first(platform)
            .map_err(|_| PyRuntimeError::new_err("No OpenCL devices found"))?;
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .map_err(|e| ocl_py_err(e, "Context Creation"))?;
        let queue = Queue::new(
            &context,
            device.clone(),
            Some(QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
        )
        .map_err(|e| ocl_py_err(e, "Queue Creation"))?;

        Ok(Self {
            context,
            device,
            queue,
            program_cache: HashMap::new(),
        })
    }

    /// Check if a kernel source is already compiled in the program cache.
    /// This allows Python to skip kernel generation if the result is already cached.
    #[pyo3(text_signature = "($self, kernel_sources)")]
    pub fn has_compiled_kernels(&self, kernel_sources: Vec<String>) -> bool {
        let mut sources_vec: Vec<&str> = kernel_sources.iter().map(|s| s.as_str()).collect();
        sources_vec.sort_unstable();
        let program_src = sources_vec.join("\n\n");
        self.program_cache.contains_key(&program_src)
    }

    #[pyo3(text_signature = "($self, kernels_to_exec_py)")]
    pub fn execute_graph(
        &mut self,
        py: Python,
        kernels_to_exec_py: Vec<PyRef<KernelTensorOps>>,
    ) -> PyResult<Vec<KernelResult>> {
        if kernels_to_exec_py.is_empty() {
            return Ok(Vec::new());
        }

        let current_kernels: Vec<KernelTensorOps> = kernels_to_exec_py
            .iter()
            .map(|k_ref| (**k_ref).clone())
            .collect();

        let mut ids_check = HashSet::new();
        for k_info in &current_kernels {
            if !ids_check.insert(k_info.kernel_id) {
                return Err(PyRuntimeError::new_err(format!(
                    "Duplicate kernel ID found: {}",
                    k_info.kernel_id
                )));
            }
        }

        let mut finished_kernel_ids = HashSet::new();
        // Device-resident outputs per kernel_id. This avoids host readback + re-upload between kernels.
        let mut device_results_map: HashMap<usize, Vec<Buffer<f32>>> = HashMap::new();
        // Work sizes for reading back results at the end.
        let mut work_sizes_map: HashMap<usize, usize> = HashMap::new();

        let mut runnable_kernel_ids: Vec<usize> = current_kernels
            .iter()
            .filter(|k_info| {
                if let Some(inputs) = &k_info.inputs {
                    for inp in inputs {
                        if inp.bind(py).extract::<LogicalInputSource>().is_ok() {
                            return false;
                        }
                    }
                }
                true
            })
            .map(|k_info| k_info.kernel_id)
            .collect();

        let total_kernels_to_run = current_kernels.len();

        while finished_kernel_ids.len() < total_kernels_to_run {
            py.check_signals()?;

            if runnable_kernel_ids.is_empty() {
                let remaining_ids: Vec<_> = current_kernels
                    .iter()
                    .filter(|k| !finished_kernel_ids.contains(&k.kernel_id))
                    .map(|k| k.kernel_id)
                    .collect();
                return Err(PyRuntimeError::new_err(format!(
                    "Deadlock or cycle detected. Remaining: {:?}",
                    remaining_ids
                )));
            }

            let current_batch_ids_to_run = runnable_kernel_ids.clone();
            runnable_kernel_ids.clear();

            let queue = self.queue.clone();

            let mut batch_kernel_sources = HashSet::new();
            let mut batch_resolved_kernel_data: HashMap<
                usize,
                (KernelTensorOps, Vec<ResolvedInput>),
            > = HashMap::new();
            let mut batch_ocl_output_buffers: HashMap<usize, Vec<Buffer<f32>>> = HashMap::new();
            let mut batch_ocl_kernel_args: HashMap<usize, Vec<Buffer<f32>>> = HashMap::new();
            let mut batch_work_sizes: HashMap<usize, usize> = HashMap::new();
            let mut batch_scalar_args: HashMap<usize, Vec<f32>> = HashMap::new();

            for kernel_id_to_run in &current_batch_ids_to_run {
                let kernel_info = current_kernels
                    .iter()
                    .find(|k| k.kernel_id == *kernel_id_to_run)
                    .expect("Missing kernel info");

                let mut resolved_inputs: Vec<ResolvedInput> = Vec::new();

                if let Some(inputs) = &kernel_info.inputs {
                    for inp in inputs {
                        if let Ok(direct) = inp.bind(py).extract::<DirectInput>() {
                            resolved_inputs.push(ResolvedInput::Host(direct.data));
                        } else if let Ok(l_src) = inp.bind(py).extract::<LogicalInputSource>() {
                            let src_bufs = device_results_map
                                .get(&l_src.source_kernel_id)
                                .ok_or_else(|| {
                                    PyRuntimeError::new_err(format!(
                                        "Missing dependency device result: {} -> {}",
                                        kernel_info.kernel_id, l_src.source_kernel_id
                                    ))
                                })?;
                            let buf = src_bufs
                                .get(l_src.source_output_index)
                                .cloned()
                                .ok_or_else(|| {
                                    PyRuntimeError::new_err("Output index out of bounds")
                                })?;
                            resolved_inputs.push(ResolvedInput::Device(buf));
                        } else {
                            return Err(PyRuntimeError::new_err(format!(
                                "Kernel {}: Invalid input type in inputs list",
                                kernel_info.kernel_id
                            )));
                        }
                    }
                }

                batch_resolved_kernel_data
                    .insert(*kernel_id_to_run, (kernel_info.clone(), resolved_inputs));
            }

            for (kernel_id, (kernel_info, resolved_inputs)) in &batch_resolved_kernel_data {
                batch_kernel_sources.insert(kernel_info.kernel_src.as_str());
                let (input_bufs, output_bufs, work_size, scalar_inputs) = kernel_info
                    .prepare_ocl_buffers_from_resolved_inputs_any(&queue, resolved_inputs)?;

                let mut all_args = input_bufs.clone();
                all_args.extend(output_bufs.clone());

                batch_ocl_kernel_args.insert(*kernel_id, all_args);
                batch_ocl_output_buffers.insert(*kernel_id, output_bufs);
                batch_work_sizes.insert(*kernel_id, work_size);
                batch_scalar_args.insert(*kernel_id, scalar_inputs);
            }

            // Stabilise source ordering so cache hits are reliable.
            let mut batch_kernel_sources_vec = batch_kernel_sources.into_iter().collect::<Vec<_>>();
            batch_kernel_sources_vec.sort_unstable();
            let program_src = batch_kernel_sources_vec.join("\n\n");

            let debug_cache = std::env::var_os("TENSOROPS_DEBUG_PROGRAM_CACHE").is_some();
            let mut src_hasher = DefaultHasher::new();
            program_src.hash(&mut src_hasher);
            let program_src_hash = src_hasher.finish();

            let program = if let Some(p) = self.program_cache.get(&program_src) {
                p.clone()
            } else {
                let p = Program::builder()
                    .src(program_src.clone())
                    .devices(self.device.clone())
                    .build(&self.context)
                    .map_err(|e| ocl_py_err(e, "Program Build"))?;
                self.program_cache.insert(program_src.clone(), p.clone());
                p
            };

            for kernel_id in &current_batch_ids_to_run {
                let (kernel_info, _resolved_inputs) =
                    batch_resolved_kernel_data.get(kernel_id).unwrap();
                let args = batch_ocl_kernel_args.get(kernel_id).unwrap();
                let work_size = *batch_work_sizes.get(kernel_id).unwrap();
                let scalar_inputs = batch_scalar_args
                    .get(kernel_id)
                    .cloned()
                    .unwrap_or_else(Vec::new);
                let scalar_count = scalar_inputs.len();

                if work_size == 0 {
                    device_results_map.insert(*kernel_id, Vec::new());
                    work_sizes_map.insert(*kernel_id, 0);
                    finished_kernel_ids.insert(*kernel_id);
                    continue;
                }

                let name = match &kernel_info.kernel_type {
                    KernelType::Predefined(pk) => match pk {
                        PredefinedKernel::VecSum => "VecSum".to_string(),
                        PredefinedKernel::VecMax => "VecMax".to_string(),
                        PredefinedKernel::VecMin => "VecMin".to_string(),
                        _ => pk.to_string(),
                    },
                    KernelType::Custom(cn) => cn.clone(),
                };

                let mut builder = OclKernel::builder();
                builder
                    .program(&program)
                    .name(&name)
                    .queue(queue.clone())
                    .global_work_size(work_size);

                // Buffer args first (all inputs + all outputs)
                for arg in args.iter() {
                    builder.arg(arg);
                }

                // Then scalar args (VecLog, VecLeakyReLU, VecSum, etc.)
                for s in &scalar_inputs {
                    builder.arg(*s);
                }

                let k = match builder.build() {
                    Ok(k) => k,
                    Err(e) => {
                        eprintln!(
                            "[tensorops_backend] Kernel Build failed for '{}' (buffer args={}, scalar args={}, work_size={})",
                            name,
                            args.len(),
                            scalar_count,
                            work_size
                        );
                        return Err(ocl_py_err(e, "Kernel Build"));
                    }
                };
                unsafe { k.enq().map_err(|e| ocl_py_err(e, "Kernel Enqueue"))? };
            }

            // Ensure all enqueued work is complete before scheduling dependent kernels.
            queue.finish().map_err(|e| ocl_py_err(e, "Queue Finish"))?;

            // Persist device output buffers for dependency resolution in subsequent batches.
            for kernel_id_done in &current_batch_ids_to_run {
                let work = *batch_work_sizes.get(kernel_id_done).unwrap_or(&0);
                work_sizes_map.insert(*kernel_id_done, work);

                if work == 0 {
                    continue;
                }

                let output_bufs = batch_ocl_output_buffers
                    .remove(kernel_id_done)
                    .unwrap_or_default();
                device_results_map.insert(*kernel_id_done, output_bufs);
                finished_kernel_ids.insert(*kernel_id_done);
            }

            for next in &current_kernels {
                if finished_kernel_ids.contains(&next.kernel_id)
                    || runnable_kernel_ids.contains(&next.kernel_id)
                {
                    continue;
                }

                let mut ready = true;
                if let Some(inputs) = &next.inputs {
                    for inp in inputs {
                        if let Ok(l_src) = inp.bind(py).extract::<LogicalInputSource>() {
                            if !finished_kernel_ids.contains(&l_src.source_kernel_id) {
                                ready = false;
                                break;
                            }
                        }
                    }
                }

                if ready {
                    runnable_kernel_ids.push(next.kernel_id);
                }
            }
        }

        let mut final_results = Vec::with_capacity(kernels_to_exec_py.len());
        for k_ref in kernels_to_exec_py {
            let id = k_ref.kernel_id;
            let work = *work_sizes_map.get(&id).unwrap_or(&0);
            if work == 0 {
                let empty_outputs = (0..k_ref.num_output_bufs)
                    .map(|_| PyByteArray::new(py, &[]).into_py(py))
                    .collect();
                final_results.push(KernelResult {
                    val: empty_outputs,
                    kernel_id: id,
                });
                continue;
            }

            let output_bufs = device_results_map.get(&id).ok_or_else(|| {
                PyRuntimeError::new_err(format!("Missing device result for kernel ID {}", id))
            })?;

            let mut outputs = Vec::with_capacity(k_ref.num_output_bufs);
            for buf in output_bufs.iter() {
                let size_bytes = work * std::mem::size_of::<f32>();

                let py_byte_array = unsafe {
                    let ptr =
                        ffi::PyByteArray_FromStringAndSize(std::ptr::null(), size_bytes as isize);
                    if ptr.is_null() {
                        return Err(PyErr::fetch(py));
                    }
                    Bound::from_owned_ptr(py, ptr).downcast_into::<PyByteArray>()?
                };

                let buffer_ptr =
                    unsafe { ffi::PyByteArray_AsString(py_byte_array.as_ptr()) as *mut f32 };
                let host_slice = unsafe { std::slice::from_raw_parts_mut(buffer_ptr, work) };

                unsafe { buf.read(host_slice).block(false).enq() }
                    .map_err(|e| ocl_py_err(e, &format!("Read Buffer for Kernel {}", id)))?;

                outputs.push(py_byte_array.into_py(py));
            }
            final_results.push(KernelResult {
                val: outputs,
                kernel_id: id,
            });
        }
        self.queue
            .finish()
            .map_err(|e| ocl_py_err(e, "Final Read Finish"))?;
        Ok(final_results)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "<Runtime device='{}'>",
            self.device.name().unwrap_or_default()
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "OpenCL Runtime on device: {}",
            self.device.name().unwrap_or_default()
        ))
    }
}
