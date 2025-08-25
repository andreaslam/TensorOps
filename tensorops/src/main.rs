use ocl::{Device, Platform};

fn main() {
    let platform = Platform::default();
    println!("Platforms: {:?}", Platform::list());
    let device_spec = Device::first(platform).unwrap();
    println!(
        "Device on default platform: {:?}",
        Device::list_all(platform)
    );

    println!("Using device: {}", device_spec.name().unwrap_or_default()); // Keep for debugging if needed
}
