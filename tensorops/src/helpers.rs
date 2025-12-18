use lazy_static::lazy_static;
use pyo3::pyfunction;
use std::collections::HashMap;

use crate::kernel::PredefinedKernel;

const KERNEL_SRC: &str = include_str!("./kernel.cl");

const KERNEL_VEC_ADD: &str = "VecAdd";
const KERNEL_VEC_SUB: &str = "VecSub";
const KERNEL_VEC_ELEMENT_MUL: &str = "VecElementMul";
const KERNEL_VEC_DIV: &str = "VecDiv";
const KERNEL_VEC_POW: &str = "VecPow";
const KERNEL_VEC_LOG: &str = "VecLog";
const KERNEL_VEC_SIN: &str = "VecSin";
const KERNEL_VEC_COS: &str = "VecCos";
const KERNEL_VEC_TAN: &str = "VecTan";
const KERNEL_VEC_ABS: &str = "VecAbs";
const KERNEL_VEC_TANH: &str = "VecTanh";
const KERNEL_VEC_LEAKY_RELU: &str = "VecLeakyReLU";

fn extract_kernel_code(source: &str, kernel_name: &str) -> Option<String> {
    let kernel_sig_pattern = format!("__kernel void {}", kernel_name);
    if let Some(sig_start_index) = source.find(&kernel_sig_pattern) {
        let search_start_index = source[..sig_start_index]
            .rfind("/*")
            .unwrap_or(sig_start_index);
        if let Some(body_brace_offset) = source[sig_start_index..].find('{') {
            let body_start_index = sig_start_index + body_brace_offset;
            let mut brace_level = 0;
            let mut body_end_index = None;
            for (i, char) in source[body_start_index..].char_indices() {
                match char {
                    '{' => brace_level += 1,
                    '}' => {
                        brace_level -= 1;
                        if brace_level == 0 {
                            body_end_index = Some(body_start_index + i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if let Some(end_index) = body_end_index {
                let snippet = source[search_start_index..=end_index].trim().to_string();
                return Some(snippet);
            }
        }
    }
    eprintln!(
        "Warning: Could not extract kernel code for '{}'",
        kernel_name
    );
    None
}

lazy_static! {
    static ref PREDEFINED_KERNEL_SOURCES: HashMap<PredefinedKernel, String> = {
        let mut m = HashMap::new();
        let kernels_to_extract = [
            (PredefinedKernel::VecAdd, KERNEL_VEC_ADD),
            (PredefinedKernel::VecSub, KERNEL_VEC_SUB),
            (PredefinedKernel::VecElementMul, KERNEL_VEC_ELEMENT_MUL),
            (PredefinedKernel::VecDiv, KERNEL_VEC_DIV),
            (PredefinedKernel::VecPow, KERNEL_VEC_POW),
            (PredefinedKernel::VecLog, KERNEL_VEC_LOG),
            (PredefinedKernel::VecSin, KERNEL_VEC_SIN),
            (PredefinedKernel::VecCos, KERNEL_VEC_COS),
            (PredefinedKernel::VecTan, KERNEL_VEC_TAN),
            (PredefinedKernel::VecAbs, KERNEL_VEC_ABS),
            (PredefinedKernel::VecTanh, KERNEL_VEC_TANH),
            (PredefinedKernel::VecLeakyReLU, KERNEL_VEC_LEAKY_RELU),
        ];
        for (kernel_enum, kernel_name) in kernels_to_extract.iter() {
            if let Some(snippet) = extract_kernel_code(&KERNEL_SRC, kernel_name) {
                m.insert(kernel_enum.clone(), snippet);
            } else {
                panic!("Kernel definition for '{}' not found or could not be parsed", kernel_name);
            }
        }
        m
    };
}

#[pyfunction]
pub fn get_predefined_kernel_source(kernel: &PredefinedKernel) -> Option<&'static String> {
    let result = PREDEFINED_KERNEL_SOURCES.get(kernel);
    result
}
