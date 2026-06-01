use crate::kernel::{
    DirectInput, KernelResult, KernelTensorOps, KernelType, LogicalInputSource, PredefinedKernel,
    ResolvedInput,
};
use ocl::core::DeviceInfo;
use ocl::flags as OclFlags;
use ocl::{Buffer, DeviceType};
use ocl::{Context, Device, Error as OclError, Kernel as OclKernel, Platform, Program, Queue};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use pyo3::AsPyPointer;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

const ADAMW_UPDATE_KEY: &str = "__adamw_update__";
const ADAMW_UPDATE_SRC: &str = r#"
__kernel void AdamWUpdate(
    __global float* W,
    __global float* M,
    __global float* V,
    __global const float* G,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bias_c1,
    float bias_c2
) {
    int gid = get_global_id(0);
    float g = G[gid];
    float m = M[gid] = beta1 * M[gid] + (1.0f - beta1) * g;
    float v = V[gid] = beta2 * V[gid] + (1.0f - beta2) * g * g;
    float m_hat = m / bias_c1;
    float v_hat = v / bias_c2;
    float w = W[gid];
    W[gid] = w - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w);
}
"#;

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(val) => {
            let v = val.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "no" || v == "off")
        }
        Err(_) => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

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
    /// Cache of device buffers for stable DirectInput objects.
    /// Keyed by the PyObject pointer address of the DirectInput instance.
    direct_input_buffer_cache: HashMap<usize, Buffer<f32>>,
    /// Cache directory for precompiled kernels.
    cache_dir: PathBuf,
    /// Device identifier for binary cache versioning.
    device_id: String,
    /// Max number of cached OpenCL programs (0 disables eviction).
    program_cache_max_entries: usize,
    /// Enable caching of DirectInput buffers on device.
    direct_input_cache_enabled: bool,
    /// Max number of cached DirectInput buffers (0 disables caching).
    direct_input_cache_max_entries: usize,
    /// Minimum input length for caching DirectInput buffers.
    direct_input_cache_min_len: usize,
    /// AdamW first-moment buffers keyed by weight DirectInput pointer.
    adam_m_cache: HashMap<usize, Buffer<f32>>,
    /// AdamW second-moment buffers keyed by weight DirectInput pointer.
    adam_v_cache: HashMap<usize, Buffer<f32>>,
}

fn pick_platform_and_device() -> (Platform, Device) {
    // Iterate all platforms
    for platform in Platform::list() {
        // Pick first GPU, fallback to CPU
        if let Ok(devices) = Device::list(platform, Some(DeviceType::GPU)) {
            if !devices.is_empty() {
                // Return platform + GPU device
                let device = devices
                    .into_iter()
                    .max_by_key(|d| match d.info(DeviceInfo::MaxComputeUnits).unwrap() {
                        ocl::enums::DeviceInfoResult::MaxComputeUnits(units) => units,
                        _ => 0,
                    })
                    .unwrap();
                return (platform, device);
            }
        }
        if let Ok(devices) = Device::list(platform, Some(DeviceType::CPU)) {
            if !devices.is_empty() {
                return (platform, devices[0]);
            }
        }
    }
    panic!("No OpenCL device found");
}

fn get_cache_dir() -> PathBuf {
    let cache_dir = if let Ok(home) = std::env::var("TENSOROPS_CACHE_DIR") {
        PathBuf::from(home)
    } else {
        let mut path = std::env::temp_dir();
        path.push("tensorops_kernel_cache");
        path
    };
    let _ = fs::create_dir_all(&cache_dir);
    cache_dir
}

fn compute_source_hash(src: &str) -> String {
    let mut hasher = DefaultHasher::new();
    src.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[pymethods]
impl Runtime {
    #[new]
    pub fn new() -> PyResult<Self> {
        let (platform, device) = pick_platform_and_device();
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .map_err(|e| ocl_py_err(e, "Context Creation"))?;
        let queue = Queue::new(&context, device.clone(), None)
            .map_err(|e| ocl_py_err(e, "Queue Creation"))?;

        let device_name = device.name().unwrap_or_default();
        let device_vendor = device.vendor().unwrap_or_default();
        let device_id = format!("{}_{}", device_vendor, device_name).replace(" ", "_");
        let cache_dir = get_cache_dir();

        let direct_input_cache_enabled = env_flag("TENSOROPS_DIRECT_INPUT_CACHE", true);
        let direct_input_cache_max_entries = if direct_input_cache_enabled {
            env_usize("TENSOROPS_DIRECT_INPUT_CACHE_MAX", 1024)
        } else {
            0
        };
        let direct_input_cache_min_len = if direct_input_cache_enabled {
            env_usize("TENSOROPS_DIRECT_INPUT_CACHE_MIN_LEN", 128)
        } else {
            usize::MAX
        };
        let program_cache_max_entries = env_usize("TENSOROPS_PROGRAM_CACHE_MAX", 256);

        Ok(Self {
            context,
            device,
            queue,
            program_cache: HashMap::new(),
            direct_input_buffer_cache: HashMap::new(),
            cache_dir,
            device_id,
            program_cache_max_entries,
            direct_input_cache_enabled,
            direct_input_cache_max_entries,
            direct_input_cache_min_len,
            adam_m_cache: HashMap::new(),
            adam_v_cache: HashMap::new(),
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

        let mut kernel_index_by_id = HashMap::with_capacity(current_kernels.len());
        for (idx, k_info) in current_kernels.iter().enumerate() {
            kernel_index_by_id.insert(k_info.kernel_id, idx);
        }

        let mut finished_kernel_ids = HashSet::new();
        // Device-resident outputs per kernel_id. This avoids host readback + re-upload between kernels.
        let mut device_results_map: HashMap<usize, Vec<Buffer<f32>>> = HashMap::new();
        // Work sizes for reading back results at the end.
        let mut work_sizes_map: HashMap<usize, usize> = HashMap::new();
        // In-order queue execution means we can rely on enqueue order without events.

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

            let mut batch_resolved_kernel_data: HashMap<
                usize,
                (KernelTensorOps, Vec<ResolvedInput>),
            > = HashMap::new();
            let mut batch_ocl_output_buffers: HashMap<usize, Vec<Buffer<f32>>> = HashMap::new();
            let mut batch_ocl_kernel_args: HashMap<usize, Vec<Buffer<f32>>> = HashMap::new();
            let mut batch_ocl_input_count: HashMap<usize, usize> = HashMap::new();
            let mut batch_work_sizes: HashMap<usize, usize> = HashMap::new();
            let mut batch_scalar_args: HashMap<usize, Vec<f32>> = HashMap::new();

            for kernel_id_to_run in &current_batch_ids_to_run {
                let kernel_index = kernel_index_by_id
                    .get(kernel_id_to_run)
                    .copied()
                    .expect("Missing kernel info");
                let kernel_info = &current_kernels[kernel_index];

                let mut resolved_inputs: Vec<ResolvedInput> = Vec::new();

                if let Some(inputs) = &kernel_info.inputs {
                    for inp in inputs {
                        // DirectInput: cache the device buffer keyed by the PyObject pointer.
                        // This is a massive speedup for tight recompute loops.
                        if let Ok(direct_ref) = inp.bind(py).extract::<PyRef<DirectInput>>() {
                            // Some kernels (VecSum/VecMax/VecMin) pass scalar parameters
                            // (pre, axis_len, post) as 1-element DirectInputs, which must remain
                            // host-resident because we extract them via ResolvedInput::scalar0.
                            // Caching them as device buffers would break scalar extraction.
                            if direct_ref.data.len() == 1 {
                                resolved_inputs.push(ResolvedInput::Host(direct_ref.data.clone()));
                                continue;
                            }
                            let data_len = direct_ref.data.len();
                            let use_cache = self.direct_input_cache_enabled
                                && self.direct_input_cache_max_entries > 0
                                && data_len >= self.direct_input_cache_min_len
                                && direct_ref.cacheable;

                            if use_cache {
                                let key = direct_ref.as_ptr() as usize;
                                resolved_inputs.push(ResolvedInput::HostCached {
                                    key,
                                    data: direct_ref.data.clone(),
                                });
                            } else {
                                resolved_inputs.push(ResolvedInput::Host(direct_ref.data.clone()));
                            }
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
                if self.direct_input_cache_enabled
                    && self.direct_input_cache_max_entries > 0
                    && self.direct_input_buffer_cache.len() >= self.direct_input_cache_max_entries
                {
                    self.direct_input_buffer_cache.clear();
                }
                let (input_bufs, output_bufs, work_size, scalar_inputs) = kernel_info
                    .prepare_ocl_buffers_from_resolved_inputs_any(
                        &queue,
                        resolved_inputs,
                        &mut self.direct_input_buffer_cache,
                    )?;

                let mut all_args = input_bufs.clone();
                all_args.extend(output_bufs.clone());

                batch_ocl_kernel_args.insert(*kernel_id, all_args);
                batch_ocl_output_buffers.insert(*kernel_id, output_bufs);
                batch_ocl_input_count.insert(*kernel_id, input_bufs.len()); // Track input count
                batch_work_sizes.insert(*kernel_id, work_size);
                batch_scalar_args.insert(*kernel_id, scalar_inputs);
            }

            for kernel_id in &current_batch_ids_to_run {
                let (kernel_info, _resolved_inputs) =
                    batch_resolved_kernel_data.get(kernel_id).unwrap();
                let args = batch_ocl_kernel_args.get(kernel_id).unwrap();
                let work_size = *batch_work_sizes.get(kernel_id).unwrap();
                let scalar_inputs = batch_scalar_args
                    .get(kernel_id)
                    .cloned()
                    .unwrap_or_else(Vec::new);

                if work_size == 0 {
                    device_results_map.insert(*kernel_id, Vec::new());
                    work_sizes_map.insert(*kernel_id, 0);
                    finished_kernel_ids.insert(*kernel_id);
                    continue;
                }

                // Program Cache Logic - compile each unique kernel source separately
                // This avoids bundling and enables reuse of predefined kernels
                let src = &kernel_info.kernel_src;
                if !self.program_cache.contains_key(src) {
                    if self.program_cache_max_entries > 0
                        && self.program_cache.len() >= self.program_cache_max_entries
                    {
                        self.program_cache.clear();
                    }
                    let _src_hash = compute_source_hash(src);
                    let _cache_path = self
                        .cache_dir
                        .join(format!("{}_{}.bin", self.device_id, _src_hash));

                    // Skip disk cache for now - focus on in-memory caching for perf
                    let p = Program::builder()
                        .src(src.clone())
                        .devices(self.device.clone())
                        .build(&self.context)
                        .map_err(|e| {
                            eprintln!(
                                "ERROR: Program build failed for kernel {}: {:?}",
                                kernel_id, e
                            );
                            ocl_py_err(e, "Program Build")
                        })?;
                    self.program_cache.insert(src.clone(), p);
                }
                let program = self.program_cache.get(src).unwrap();

                let name = match &kernel_info.kernel_type {
                    KernelType::Predefined(pk) => match pk {
                        PredefinedKernel::VecSum => "VecSum".to_string(),
                        PredefinedKernel::VecMax => "VecMax".to_string(),
                        PredefinedKernel::VecMin => "VecMin".to_string(),
                        PredefinedKernel::MatMul => "TiledMatMul_16x16".to_string(),
                        _ => pk.to_string(),
                    },
                    KernelType::Custom(cn) => cn.clone(),
                };

                let mut builder = OclKernel::builder();
                builder.program(program).name(&name).queue(queue.clone());

                // Handle 2D vs 1D global work size
                if let Some((gx, gy)) = kernel_info.global_work_size_2d {
                    builder.global_work_size([gx, gy]);
                    if let Some((lx, ly)) = kernel_info.local_work_size {
                        builder.local_work_size([lx, ly]);
                    }
                } else {
                    builder.global_work_size(work_size);
                }

                // For tiled MatMul, insert local memory BETWEEN input and output args
                let needs_local_mem = name.contains("TiledMatMul");
                let input_count = batch_ocl_input_count.get(kernel_id).copied().unwrap_or(0);

                if needs_local_mem && input_count > 0 && args.len() > input_count {
                    // Add input buffers first (A, B, M, N, K)
                    for i in 0..input_count {
                        builder.arg(&args[i]);
                    }

                    // Then add local memory (A_tile, B_tile)
                    let tile_size = if let Some(pos) = name.rfind('_') {
                        let suffix = &name[pos + 1..];
                        if let Some(x_pos) = suffix.find('x') {
                            suffix[..x_pos].parse::<usize>().unwrap_or(16)
                        } else {
                            16
                        }
                    } else {
                        16
                    };
                    let local_mem_elements = tile_size * tile_size;
                    builder.arg_local::<f32>(local_mem_elements); // A_tile
                    builder.arg_local::<f32>(local_mem_elements); // B_tile

                    // Finally add output buffers (C)
                    for i in input_count..args.len() {
                        builder.arg(&args[i]);
                    }
                } else {
                    // Regular kernel: just add all buffer args
                    for arg in args.iter() {
                        builder.arg(arg);
                    }
                }

                // Then scalar args (VecLog, VecLeakyReLU, VecSum, etc.)
                for s in scalar_inputs {
                    builder.arg(s);
                }

                let k = builder.build().map_err(|e| {
                    eprintln!("ERROR building kernel '{}': {:?}", name, e);
                    ocl_py_err(e, "Kernel Build")
                })?;

                let (gws_arr, lws_arr, dim) =
                    if let Some((gx, gy)) = kernel_info.global_work_size_2d {
                        let g = [gx, gy, 1];
                        let l = if let Some((lx, ly)) = kernel_info.local_work_size {
                            Some([lx, ly, 1])
                        } else {
                            None
                        };
                        (g, l, 2)
                    } else {
                        ([work_size, 1, 1], None, 1)
                    };

                unsafe {
                    ocl::core::enqueue_kernel(
                        &queue,
                        &k,
                        dim,
                        None,
                        &gws_arr,
                        lws_arr,
                        None::<&ocl::EventList>,
                        None::<&mut ocl::Event>,
                    )
                    .map_err(|e| ocl_py_err(ocl::Error::from(e), "Kernel Enqueue"))?;
                }
            }

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

        // Final wait
        self.queue
            .finish()
            .map_err(|e| ocl_py_err(e, "Queue Finish"))?;

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
                let mut host = vec![0.0f32; work];
                buf.read(&mut host)
                    .enq()
                    .map_err(|e| ocl_py_err(e, &format!("Read Buffer for Kernel {}", id)))?;

                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        host.as_ptr() as *const u8,
                        host.len() * std::mem::size_of::<f32>(),
                    )
                };
                let byte_array = PyByteArray::new(py, byte_slice);
                outputs.push(byte_array.into_py(py));
            }
            final_results.push(KernelResult {
                val: outputs,
                kernel_id: id,
            });
        }
        Ok(final_results)
    }

    #[pyo3(text_signature = "($self, kernels_to_exec_py, updates, skip_readback=False)")]
    pub fn execute_graph_with_updates(
        &mut self,
        py: Python,
        kernels_to_exec_py: Vec<PyRef<KernelTensorOps>>,
        updates: Vec<(
            Py<DirectInput>,
            usize,
            usize,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        )>,
        skip_readback: Option<bool>,
    ) -> PyResult<Vec<KernelResult>> {
        if kernels_to_exec_py.is_empty() {
            return Ok(Vec::new());
        }

        let skip_readback = skip_readback.unwrap_or(false);
        let debug_vec_element_mul = env_flag("TENSOROPS_DEBUG_VEC_ELEMENT_MUL", false);
        let sync_each_kernel = env_flag("TENSOROPS_SYNC_EACH_KERNEL", false);
        if env_flag("TENSOROPS_CLEAR_PROGRAM_CACHE", false) {
            self.program_cache.clear();
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

        let mut kernel_index_by_id = HashMap::with_capacity(current_kernels.len());
        for (idx, k_info) in current_kernels.iter().enumerate() {
            kernel_index_by_id.insert(k_info.kernel_id, idx);
        }

        let mut finished_kernel_ids = HashSet::new();
        let mut device_results_map: HashMap<usize, Vec<Buffer<f32>>> = HashMap::new();
        let mut work_sizes_map: HashMap<usize, usize> = HashMap::new();

        let mut kernel_output_use_counts: HashMap<usize, Vec<usize>> = HashMap::new();
        for k in &current_kernels {
            kernel_output_use_counts.insert(k.kernel_id, vec![0; k.num_output_bufs]);
        }

        // Count logical input consumers per output buffer.
        for k in &current_kernels {
            if let Some(inputs) = &k.inputs {
                for inp in inputs {
                    if let Ok(l_src) = inp.bind(py).extract::<LogicalInputSource>() {
                        if let Some(counts) =
                            kernel_output_use_counts.get_mut(&l_src.source_kernel_id)
                        {
                            if l_src.source_output_index < counts.len() {
                                counts[l_src.source_output_index] += 1;
                            }
                        }
                    }
                }
            }
        }

        // Keep outputs needed for readback.
        if !skip_readback {
            for kernel_py in &kernels_to_exec_py {
                if let Some(counts) = kernel_output_use_counts.get_mut(&kernel_py.kernel_id) {
                    for c in counts.iter_mut() {
                        *c += 1;
                    }
                }
            }
        }

        // Keep outputs needed for device updates.
        for (_, grad_kernel_id, grad_output_idx, _, _, _, _, _, _, _) in updates.iter() {
            if let Some(counts) = kernel_output_use_counts.get_mut(grad_kernel_id) {
                if *grad_output_idx < counts.len() {
                    counts[*grad_output_idx] += 1;
                }
            }
        }

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

            let mut batch_resolved_kernel_data: HashMap<
                usize,
                (KernelTensorOps, Vec<ResolvedInput>),
            > = HashMap::new();
            let mut batch_input_sources: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            let mut batch_ocl_output_buffers: HashMap<usize, Vec<Buffer<f32>>> = HashMap::new();
            let mut batch_ocl_kernel_args: HashMap<usize, Vec<Buffer<f32>>> = HashMap::new();
            let mut batch_ocl_input_count: HashMap<usize, usize> = HashMap::new();
            let mut batch_work_sizes: HashMap<usize, usize> = HashMap::new();
            let mut batch_scalar_args: HashMap<usize, Vec<f32>> = HashMap::new();

            for kernel_id_to_run in &current_batch_ids_to_run {
                let kernel_index = kernel_index_by_id
                    .get(kernel_id_to_run)
                    .copied()
                    .expect("Missing kernel info");
                let kernel_info = &current_kernels[kernel_index];

                let mut resolved_inputs: Vec<ResolvedInput> = Vec::new();
                let mut input_sources: Vec<(usize, usize)> = Vec::new();

                if let Some(inputs) = &kernel_info.inputs {
                    for inp in inputs {
                        if let Ok(direct_ref) = inp.bind(py).extract::<PyRef<DirectInput>>() {
                            if direct_ref.data.len() == 1 {
                                resolved_inputs.push(ResolvedInput::Host(direct_ref.data.clone()));
                                continue;
                            }
                            let data_len = direct_ref.data.len();
                            let use_cache = self.direct_input_cache_enabled
                                && self.direct_input_cache_max_entries > 0
                                && data_len >= self.direct_input_cache_min_len
                                && direct_ref.cacheable;

                            if use_cache {
                                let key = direct_ref.as_ptr() as usize;
                                resolved_inputs.push(ResolvedInput::HostCached {
                                    key,
                                    data: direct_ref.data.clone(),
                                });
                            } else {
                                resolved_inputs.push(ResolvedInput::Host(direct_ref.data.clone()));
                            }
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
                            input_sources.push((l_src.source_kernel_id, l_src.source_output_index));
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
                batch_input_sources.insert(*kernel_id_to_run, input_sources);
            }

            for (kernel_id, (kernel_info, resolved_inputs)) in &batch_resolved_kernel_data {
                if self.direct_input_cache_enabled
                    && self.direct_input_cache_max_entries > 0
                    && self.direct_input_buffer_cache.len() >= self.direct_input_cache_max_entries
                {
                    self.direct_input_buffer_cache.clear();
                }
                let (input_bufs, output_bufs, work_size, scalar_inputs) = kernel_info
                    .prepare_ocl_buffers_from_resolved_inputs_any(
                        &queue,
                        resolved_inputs,
                        &mut self.direct_input_buffer_cache,
                    )?;

                let mut all_args = input_bufs.clone();
                all_args.extend(output_bufs.clone());

                batch_ocl_kernel_args.insert(*kernel_id, all_args);
                batch_ocl_output_buffers.insert(*kernel_id, output_bufs);
                batch_ocl_input_count.insert(*kernel_id, input_bufs.len());
                batch_work_sizes.insert(*kernel_id, work_size);
                batch_scalar_args.insert(*kernel_id, scalar_inputs);
            }

            for kernel_id in &current_batch_ids_to_run {
                let (kernel_info, _resolved_inputs) =
                    batch_resolved_kernel_data.get(kernel_id).unwrap();
                let args = batch_ocl_kernel_args.get(kernel_id).unwrap();
                let work_size = *batch_work_sizes.get(kernel_id).unwrap();
                let scalar_inputs = batch_scalar_args
                    .get(kernel_id)
                    .cloned()
                    .unwrap_or_else(Vec::new);

                if work_size == 0 {
                    device_results_map.insert(*kernel_id, Vec::new());
                    work_sizes_map.insert(*kernel_id, 0);
                    finished_kernel_ids.insert(*kernel_id);
                    continue;
                }

                let src = &kernel_info.kernel_src;
                if !self.program_cache.contains_key(src) {
                    if self.program_cache_max_entries > 0
                        && self.program_cache.len() >= self.program_cache_max_entries
                    {
                        self.program_cache.clear();
                    }
                    let _src_hash = compute_source_hash(src);
                    let _cache_path = self
                        .cache_dir
                        .join(format!("{}_{}.bin", self.device_id, _src_hash));

                    let p = Program::builder()
                        .src(src.clone())
                        .devices(self.device.clone())
                        .build(&self.context)
                        .map_err(|e| {
                            eprintln!(
                                "ERROR: Program build failed for kernel {}: {:?}",
                                kernel_id, e
                            );
                            ocl_py_err(e, "Program Build")
                        })?;
                    self.program_cache.insert(src.clone(), p);
                }
                let program = self.program_cache.get(src).unwrap();

                let name = match &kernel_info.kernel_type {
                    KernelType::Predefined(pk) => match pk {
                        PredefinedKernel::VecSum => "VecSum".to_string(),
                        PredefinedKernel::VecMax => "VecMax".to_string(),
                        PredefinedKernel::VecMin => "VecMin".to_string(),
                        PredefinedKernel::MatMul => "TiledMatMul_16x16".to_string(),
                        _ => pk.to_string(),
                    },
                    KernelType::Custom(cn) => cn.clone(),
                };

                if debug_vec_element_mul && name == "VecElementMul" {
                    let input_count = batch_ocl_input_count.get(kernel_id).copied().unwrap_or(0);
                    let mut input_lens = Vec::new();
                    let mut output_lens = Vec::new();
                    for (idx, buf) in args.iter().enumerate() {
                        if idx < input_count {
                            input_lens.push(buf.len());
                        } else {
                            output_lens.push(buf.len());
                        }
                    }
                    eprintln!(
                        "VecElementMul kernel_id={} work_size={} input_lens={:?} output_lens={:?}",
                        kernel_info.kernel_id, work_size, input_lens, output_lens
                    );
                }

                let mut builder = OclKernel::builder();
                builder.program(program).name(&name).queue(queue.clone());

                if let Some((gx, gy)) = kernel_info.global_work_size_2d {
                    builder.global_work_size([gx, gy]);
                    if let Some((lx, ly)) = kernel_info.local_work_size {
                        builder.local_work_size([lx, ly]);
                    }
                } else {
                    builder.global_work_size(work_size);
                }

                let needs_local_mem = name.contains("TiledMatMul");
                let input_count = batch_ocl_input_count.get(kernel_id).copied().unwrap_or(0);

                if needs_local_mem && input_count > 0 && args.len() > input_count {
                    for i in 0..input_count {
                        builder.arg(&args[i]);
                    }

                    let tile_size = if let Some(pos) = name.rfind('_') {
                        let suffix = &name[pos + 1..];
                        if let Some(x_pos) = suffix.find('x') {
                            suffix[..x_pos].parse::<usize>().unwrap_or(16)
                        } else {
                            16
                        }
                    } else {
                        16
                    };
                    let local_mem_elements = tile_size * tile_size;
                    builder.arg_local::<f32>(local_mem_elements);
                    builder.arg_local::<f32>(local_mem_elements);

                    for i in input_count..args.len() {
                        builder.arg(&args[i]);
                    }
                } else {
                    for arg in args.iter() {
                        builder.arg(arg);
                    }
                }

                for s in scalar_inputs {
                    builder.arg(s);
                }

                let k = builder.build().map_err(|e| {
                    eprintln!("ERROR building kernel '{}': {:?}", name, e);
                    ocl_py_err(e, "Kernel Build")
                })?;

                let (gws_arr, lws_arr, dim) =
                    if let Some((gx, gy)) = kernel_info.global_work_size_2d {
                        let g = [gx, gy, 1];
                        let l = if let Some((lx, ly)) = kernel_info.local_work_size {
                            Some([lx, ly, 1])
                        } else {
                            None
                        };
                        (g, l, 2)
                    } else {
                        ([work_size, 1, 1], None, 1)
                    };

                unsafe {
                    ocl::core::enqueue_kernel(
                        &queue,
                        &k,
                        dim,
                        None,
                        &gws_arr,
                        lws_arr,
                        None::<&ocl::EventList>,
                        None::<&mut ocl::Event>,
                    )
                    .map_err(|e| ocl_py_err(ocl::Error::from(e), "Kernel Enqueue"))?;
                }

                if let Some(sources) = batch_input_sources.get(kernel_id) {
                    for (src_id, out_idx) in sources.iter() {
                        if let Some(counts) = kernel_output_use_counts.get_mut(src_id) {
                            if *out_idx < counts.len() {
                                if counts[*out_idx] > 0 {
                                    counts[*out_idx] -= 1;
                                }
                                if counts.iter().all(|c| *c == 0) {
                                    device_results_map.remove(src_id);
                                }
                            }
                        }
                    }
                }

                if sync_each_kernel {
                    queue
                        .finish()
                        .map_err(|e| ocl_py_err(e, "Queue Finish (Kernel)"))?;
                }
            }

            // Finish the OpenCL queue after each batch to reduce command-queue resource pressure.
            queue
                .finish()
                .map_err(|e| ocl_py_err(e, "Queue Finish (Batch)"))?;

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

        if !updates.is_empty() {
            if !self.program_cache.contains_key(ADAMW_UPDATE_KEY) {
                let p = Program::builder()
                    .src(ADAMW_UPDATE_SRC)
                    .devices(self.device.clone())
                    .build(&self.context)
                    .map_err(|e| ocl_py_err(e, "AdamW Program Build"))?;
                self.program_cache.insert(ADAMW_UPDATE_KEY.to_string(), p);
            }

            let update_program = self.program_cache.get(ADAMW_UPDATE_KEY).unwrap();

            for (
                weight_input,
                grad_kernel_id,
                grad_output_idx,
                lr,
                beta1,
                beta2,
                eps,
                wd,
                bias_c1,
                bias_c2,
            ) in updates
            {
                let weight_bound = weight_input.bind(py);
                let weight_key = weight_bound.as_ptr() as usize;
                let weight_ref = weight_bound.borrow();
                let grad_bufs = device_results_map.get(&grad_kernel_id).ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "Missing gradient buffer for kernel ID {}",
                        grad_kernel_id
                    ))
                })?;
                let grad_buf = grad_bufs.get(grad_output_idx).cloned().ok_or_else(|| {
                    PyRuntimeError::new_err("Gradient output index out of bounds")
                })?;

                let weight_buf = if let Some(buf) = self.direct_input_buffer_cache.get(&weight_key)
                {
                    buf.clone()
                } else {
                    let data = &weight_ref.data;
                    let buf = Buffer::<f32>::builder()
                        .queue(self.queue.clone())
                        .flags(OclFlags::MEM_READ_WRITE | OclFlags::MEM_COPY_HOST_PTR)
                        .len(data.len())
                        .copy_host_slice(data.as_slice())
                        .build()
                        .map_err(|e| ocl_py_err(e, "Weight Buffer Build"))?;
                    self.direct_input_buffer_cache
                        .insert(weight_key, buf.clone());
                    buf
                };

                let weight_len = weight_buf.len();
                if grad_buf.len() != weight_len {
                    return Err(PyRuntimeError::new_err(format!(
                        "Gradient length {} does not match weight length {}",
                        grad_buf.len(),
                        weight_len
                    )));
                }

                let m_buf = if let Some(buf) = self.adam_m_cache.get(&weight_key) {
                    buf.clone()
                } else {
                    let zeros = vec![0.0f32; weight_len];
                    let buf = Buffer::<f32>::builder()
                        .queue(self.queue.clone())
                        .flags(OclFlags::MEM_READ_WRITE | OclFlags::MEM_COPY_HOST_PTR)
                        .len(weight_len)
                        .copy_host_slice(zeros.as_slice())
                        .build()
                        .map_err(|e| ocl_py_err(e, "AdamW M Buffer Build"))?;
                    self.adam_m_cache.insert(weight_key, buf.clone());
                    buf
                };

                let v_buf = if let Some(buf) = self.adam_v_cache.get(&weight_key) {
                    buf.clone()
                } else {
                    let zeros = vec![0.0f32; weight_len];
                    let buf = Buffer::<f32>::builder()
                        .queue(self.queue.clone())
                        .flags(OclFlags::MEM_READ_WRITE | OclFlags::MEM_COPY_HOST_PTR)
                        .len(weight_len)
                        .copy_host_slice(zeros.as_slice())
                        .build()
                        .map_err(|e| ocl_py_err(e, "AdamW V Buffer Build"))?;
                    self.adam_v_cache.insert(weight_key, buf.clone());
                    buf
                };

                let mut builder = OclKernel::builder();
                builder
                    .program(update_program)
                    .name("AdamWUpdate")
                    .queue(self.queue.clone())
                    .global_work_size(weight_len)
                    .arg(&weight_buf)
                    .arg(&m_buf)
                    .arg(&v_buf)
                    .arg(&grad_buf)
                    .arg(lr)
                    .arg(beta1)
                    .arg(beta2)
                    .arg(eps)
                    .arg(wd)
                    .arg(bias_c1)
                    .arg(bias_c2);

                let k = builder.build().map_err(|e| {
                    eprintln!("ERROR building AdamW update kernel: {:?}", e);
                    ocl_py_err(e, "AdamW Kernel Build")
                })?;

                let gws_arr = [weight_len, 1, 1];
                unsafe {
                    ocl::core::enqueue_kernel(
                        &self.queue,
                        &k,
                        1,
                        None,
                        &gws_arr,
                        None::<[usize; 3]>,
                        None::<&ocl::EventList>,
                        None::<&mut ocl::Event>,
                    )
                    .map_err(|e| ocl_py_err(ocl::Error::from(e), "AdamW Enqueue"))?;
                }
            }
        }

        self.queue
            .finish()
            .map_err(|e| ocl_py_err(e, "Queue Finish"))?;

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

            if skip_readback {
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
                let mut host = vec![0.0f32; work];
                buf.read(&mut host)
                    .enq()
                    .map_err(|e| ocl_py_err(e, &format!("Read Buffer for Kernel {}", id)))?;

                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        host.as_ptr() as *const u8,
                        host.len() * std::mem::size_of::<f32>(),
                    )
                };
                let byte_array = PyByteArray::new(py, byte_slice);
                outputs.push(byte_array.into_py(py));
            }
            final_results.push(KernelResult {
                val: outputs,
                kernel_id: id,
            });
        }

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
