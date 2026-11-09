//! Machine probe: memory, pressure, Metal working set, cores, per-process footprint.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MachineInfo {
    pub model_identifier: String,
    pub os: String,
    pub cpu_brand: String,
    pub logical_cpus: u32,
    pub perf_cores: u32,
    pub efficiency_cores: u32,
    pub gpu_cores: Option<u32>,
    pub unified_memory: bool,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub swap_used_bytes: u64,
    /// Metal `recommendedMaxWorkingSetSize`: how much the GPU may use before macOS pages.
    pub metal_working_set_bytes: Option<u64>,
    /// normal, warn, critical (macOS kern.memorystatus_vm_pressure_level), or unknown.
    pub pressure: String,
}

pub trait Probe: Send + Sync {
    fn snapshot(&self) -> MachineInfo;
    /// Memory attributed to a process. On macOS this is `phys_footprint`, which includes Metal
    /// allocations on unified memory; elsewhere it is the resident set size.
    fn process_footprint(&self, pid: u32) -> Option<u64>;
}

pub type SharedProbe = Arc<dyn Probe>;

pub struct SystemProbe {
    static_info: MachineInfo,
    sys: Mutex<sysinfo::System>,
}

impl SystemProbe {
    pub fn new() -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());
        let logical = sys.cpus().len() as u32;
        let brand = sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
        let mut info = MachineInfo {
            os: format!("{} {}", sysinfo::System::name().unwrap_or_default(), sysinfo::System::os_version().unwrap_or_default()),
            cpu_brand: brand,
            logical_cpus: logical,
            perf_cores: logical,
            efficiency_cores: 0,
            total_bytes: sys.total_memory(),
            ..Default::default()
        };
        platform::fill_static(&mut info);
        Self { static_info: info, sys: Mutex::new(sys) }
    }
}

impl Default for SystemProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl Probe for SystemProbe {
    fn snapshot(&self) -> MachineInfo {
        let mut info = self.static_info.clone();
        {
            let mut sys = self.sys.lock();
            sys.refresh_memory();
            info.available_bytes = sys.available_memory();
            info.swap_used_bytes = sys.used_swap();
        }
        info.pressure = platform::pressure();
        info
    }

    fn process_footprint(&self, pid: u32) -> Option<u64> {
        platform::footprint(pid)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::MachineInfo;
    use std::ffi::CString;

    fn sysctl_u64(name: &str) -> Option<u64> {
        let c = CString::new(name).ok()?;
        let mut v: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let r = unsafe { libc::sysctlbyname(c.as_ptr(), &mut v as *mut u64 as *mut _, &mut len, std::ptr::null_mut(), 0) };
        if r != 0 {
            return None;
        }
        // Values may be 32-bit.
        Some(if len == 4 { v & 0xffff_ffff } else { v })
    }

    fn sysctl_string(name: &str) -> Option<String> {
        let c = CString::new(name).ok()?;
        let mut len = 0usize;
        unsafe { libc::sysctlbyname(c.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) };
        if len == 0 {
            return None;
        }
        let mut buf = vec![0u8; len];
        let r = unsafe { libc::sysctlbyname(c.as_ptr(), buf.as_mut_ptr() as *mut _, &mut len, std::ptr::null_mut(), 0) };
        if r != 0 {
            return None;
        }
        Some(String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string())
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {}

    fn metal_working_set() -> Option<u64> {
        use objc2_metal::MTLDevice;
        let dev = objc2_metal::MTLCreateSystemDefaultDevice()?;
        Some(dev.recommendedMaxWorkingSetSize())
    }

    fn gpu_cores() -> Option<u32> {
        let out = std::process::Command::new("/usr/sbin/ioreg").args(["-rc", "AGXAccelerator", "-d", "1"]).output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let line = text.lines().find(|l| l.contains("\"gpu-core-count\""))?;
        line.rsplit('=').next()?.trim().parse().ok()
    }

    pub fn fill_static(info: &mut MachineInfo) {
        info.model_identifier = sysctl_string("hw.model").unwrap_or_default();
        if let Some(b) = sysctl_string("machdep.cpu.brand_string") {
            info.cpu_brand = b;
        }
        if let Some(p) = sysctl_u64("hw.perflevel0.physicalcpu") {
            info.perf_cores = p as u32;
        }
        if let Some(e) = sysctl_u64("hw.perflevel1.physicalcpu") {
            info.efficiency_cores = e as u32;
        }
        if let Some(m) = sysctl_u64("hw.memsize") {
            info.total_bytes = m;
        }
        info.unified_memory = std::env::consts::ARCH == "aarch64";
        info.metal_working_set_bytes = metal_working_set();
        info.gpu_cores = gpu_cores();
    }

    pub fn pressure() -> String {
        match sysctl_u64("kern.memorystatus_vm_pressure_level") {
            Some(1) => "normal",
            Some(2) => "warn",
            Some(4) => "critical",
            _ => "unknown",
        }
        .into()
    }

    #[repr(C)]
    #[derive(Default)]
    struct RusageInfoV2 {
        uuid: [u8; 16],
        user_time: u64,
        system_time: u64,
        pkg_idle_wkups: u64,
        interrupt_wkups: u64,
        pageins: u64,
        wired_size: u64,
        resident_size: u64,
        phys_footprint: u64,
        proc_start_abstime: u64,
        proc_exit_abstime: u64,
        child_user_time: u64,
        child_system_time: u64,
        child_pkg_idle_wkups: u64,
        child_interrupt_wkups: u64,
        child_pageins: u64,
        child_elapsed_abstime: u64,
        diskio_bytesread: u64,
        diskio_byteswritten: u64,
    }

    extern "C" {
        fn proc_pid_rusage(pid: libc::c_int, flavor: libc::c_int, buffer: *mut RusageInfoV2) -> libc::c_int;
    }

    pub fn footprint(pid: u32) -> Option<u64> {
        let mut info = RusageInfoV2::default();
        let r = unsafe { proc_pid_rusage(pid as libc::c_int, 2, &mut info) };
        (r == 0).then_some(info.phys_footprint)
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::MachineInfo;

    pub fn fill_static(info: &mut MachineInfo) {
        info.model_identifier = std::fs::read_to_string("/sys/devices/virtual/dmi/id/product_name")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| std::env::consts::ARCH.to_string());
        info.unified_memory = false;
    }

    pub fn pressure() -> String {
        // /proc/pressure/memory "some avg10" as a rough equivalent.
        let Ok(text) = std::fs::read_to_string("/proc/pressure/memory") else { return "unknown".into() };
        let avg10 = text
            .lines()
            .find(|l| l.starts_with("some"))
            .and_then(|l| l.split_whitespace().find_map(|kv| kv.strip_prefix("avg10=")))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0);
        if avg10 > 40.0 {
            "critical"
        } else if avg10 > 10.0 {
            "warn"
        } else {
            "normal"
        }
        .into()
    }

    pub fn footprint(pid: u32) -> Option<u64> {
        let statm = std::fs::read_to_string(format!("/proc/{pid}/statm")).ok()?;
        let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        Some(pages * page)
    }
}

/// A probe with fixed numbers, for tests and dry runs.
pub struct FakeProbe {
    pub info: Mutex<MachineInfo>,
}

impl FakeProbe {
    pub fn new(info: MachineInfo) -> Self {
        Self { info: Mutex::new(info) }
    }
    pub fn mac_mini_24gb() -> MachineInfo {
        MachineInfo {
            model_identifier: "Mac16,10".into(),
            os: "macOS 27.2".into(),
            cpu_brand: "Apple M4".into(),
            logical_cpus: 10,
            perf_cores: 4,
            efficiency_cores: 6,
            gpu_cores: Some(10),
            unified_memory: true,
            total_bytes: 24 * 1024 * 1024 * 1024,
            available_bytes: 14 * 1024 * 1024 * 1024,
            swap_used_bytes: 0,
            metal_working_set_bytes: Some(17_179_869_184),
            pressure: "normal".into(),
        }
    }
}

impl Probe for FakeProbe {
    fn snapshot(&self) -> MachineInfo {
        self.info.lock().clone()
    }
    fn process_footprint(&self, pid: u32) -> Option<u64> {
        platform::footprint(pid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn system_probe_reports_memory() {
        let p = SystemProbe::new();
        let s = p.snapshot();
        assert!(s.total_bytes > 0);
        assert!(s.available_bytes <= s.total_bytes);
        assert!(p.process_footprint(std::process::id()).unwrap_or(1) > 0);
    }
}
