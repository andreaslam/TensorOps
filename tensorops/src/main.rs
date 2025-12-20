use ocl::{Device, DeviceType, enums::DeviceInfo, Platform};

fn main() {
    // Collect all GPUs across all platforms
    let mut gpus: Vec<Device> = Platform::list().iter()
        .filter_map(|platform| {
            Device::list(*platform, Some(DeviceType::GPU)).ok()
        })
        .flatten()
        .collect();

    // Pick GPU with maximum compute units
    let device = if !gpus.is_empty() {
        gpus.into_iter().max_by_key(|d| {
            match d.info(DeviceInfo::MaxComputeUnits).unwrap() {
                ocl::enums::DeviceInfoResult::MaxComputeUnits(units) => units,
                _ => 0,
            }
        }).unwrap()
    } else {
        // Fallback: pick the first available device (CPU)
        Platform::list().iter()
            .filter_map(|platform| Device::list_all(*platform).ok())
            .flatten()
            .next()
            .expect("No OpenCL device found")
    };

    println!("Using device: {}", device.name().unwrap_or_default());
    println!("Type: {:?}", device.info(DeviceInfo::Type).unwrap());
    if let ocl::enums::DeviceInfoResult::MaxComputeUnits(units) = device.info(DeviceInfo::MaxComputeUnits).unwrap() {
        println!("Max compute units: {}", units);
    }
}
