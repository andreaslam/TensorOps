use ocl::{Device, Platform};

fn main() {
    // List all platforms
    let platforms = Platform::list();
    println!("Number of platforms detected: {}", platforms.len());
    println!();

    for (i, platform) in platforms.iter().enumerate() {
        println!("Platform {}: {}", i, platform.name().unwrap_or_default());
        println!("  Vendor: {}", platform.vendor().unwrap_or_default());
        println!("  Version: {}", platform.version().unwrap_or_default());
        
        // List all devices on this platform
        let devices = Device::list_all(*platform).unwrap_or_default();
        println!("  Number of devices: {}", devices.len());
        
        for (j, device) in devices.iter().enumerate() {
            println!("    Device {}: {}", j, device.name().unwrap_or_default());
            println!("      Type: {:?}", device.info(ocl::enums::DeviceInfo::Type));
            println!("      Vendor: {}", device.vendor().unwrap_or_default());
        }
        println!();
    }

    // Use default platform
    let platform = Platform::default();
    println!("Default platform: {}", platform.name().unwrap_or_default());
    
    let device = Device::first(platform).unwrap();
    println!("Using device: {}", device.name().unwrap_or_default());
}