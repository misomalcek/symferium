//! The governor: decides whether a load fits, with every number in the answer.

use crate::config::{Config, GB};
use crate::estimate::Estimate;
use crate::probe::MachineInfo;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Default)]
pub struct Budget {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub min_free_bytes: u64,
    pub metal_working_set_bytes: Option<u64>,
    /// Ceiling for everything Symferium loads.
    pub cap_bytes: u64,
    /// Sum of estimates (or measured, if larger) of loaded models.
    pub committed_bytes: u64,
    /// Estimates of models still starting (not yet visible in available memory).
    pub pending_bytes: u64,
    /// Largest load that would pass right now.
    pub headroom_bytes: u64,
    pub pressure: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    pub allowed: bool,
    pub reasons: Vec<String>,
    pub estimate: Estimate,
    pub budget: Budget,
}

pub fn budget(cfg: &Config, m: &MachineInfo, committed: u64, pending: u64) -> Budget {
    let g = &cfg.governor;
    let min_free = (g.min_free_gb * GB) as u64;
    let cap = if g.max_total_gb > 0.0 {
        (g.max_total_gb * GB) as u64
    } else {
        let by_total = m.total_bytes.saturating_sub(min_free);
        m.metal_working_set_bytes.map(|w| w.min(by_total)).unwrap_or(by_total)
    };
    let by_cap = cap.saturating_sub(committed);
    let by_free = m.available_bytes.saturating_sub(pending).saturating_sub(min_free);
    Budget {
        total_bytes: m.total_bytes,
        available_bytes: m.available_bytes,
        min_free_bytes: min_free,
        metal_working_set_bytes: m.metal_working_set_bytes,
        cap_bytes: cap,
        committed_bytes: committed,
        pending_bytes: pending,
        headroom_bytes: by_cap.min(by_free),
        pressure: m.pressure.clone(),
    }
}

fn gb(b: u64) -> String {
    format!("{:.2} GB", b as f64 / GB)
}

fn pressure_rank(p: &str) -> u8 {
    match p {
        "warn" => 1,
        "critical" => 2,
        _ => 0,
    }
}

/// `loaded_classes`: classes of chat models currently loaded (for the exclusive-large rule).
pub fn check(
    cfg: &Config,
    m: &MachineInfo,
    est: Estimate,
    committed: u64,
    pending: u64,
    loaded_classes: &[String],
    new_class: &str,
) -> Decision {
    let b = budget(cfg, m, committed, pending);
    let mut reasons = Vec::new();
    if committed + est.total > b.cap_bytes {
        reasons.push(format!(
            "would exceed the Symferium ceiling: loaded {} + this model {} = {} > ceiling {}",
            gb(committed),
            gb(est.total),
            gb(committed + est.total),
            gb(b.cap_bytes)
        ));
    }
    let free_after = m.available_bytes as i64 - pending as i64 - est.total as i64;
    if free_after < b.min_free_bytes as i64 {
        reasons.push(format!(
            "would leave {} free (available {} − starting {} − this model {}), below the minimum {}",
            if free_after < 0 { format!("-{}", gb(free_after.unsigned_abs())) } else { gb(free_after as u64) },
            gb(m.available_bytes),
            gb(pending),
            gb(est.total),
            gb(b.min_free_bytes)
        ));
    }
    let limit = match cfg.governor.refuse_on_pressure.as_str() {
        "warn" => 1,
        "critical" => 2,
        _ => u8::MAX,
    };
    if pressure_rank(&m.pressure) >= limit {
        reasons.push(format!("macOS memory pressure is {}", m.pressure));
    }
    if cfg.governor.exclusive_large {
        let chat_classes = ["small", "standard", "large", "decision"];
        if new_class == "large" && loaded_classes.iter().any(|c| chat_classes.contains(&c.as_str())) {
            reasons.push("a large model loads alone: unload the other chat models first (governor.exclusive_large)".into());
        }
        if chat_classes.contains(&new_class) && loaded_classes.iter().any(|c| c == "large") {
            reasons.push("a large model is loaded and runs alone (governor.exclusive_large)".into());
        }
    }
    Decision { allowed: reasons.is_empty(), reasons, estimate: est, budget: b }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::FakeProbe;

    fn est(gb_: f64) -> Estimate {
        Estimate { total: (gb_ * GB) as u64, ..Default::default() }
    }

    #[test]
    fn fits_and_refuses_with_numbers() {
        let cfg = Config::defaults_for(&FakeProbe::mac_mini_24gb());
        let m = FakeProbe::mac_mini_24gb(); // 14 GB available, min free 4 GB
        let ok = check(&cfg, &m, est(8.0), 0, 0, &[], "standard");
        assert!(ok.allowed, "{:?}", ok.reasons);
        let no = check(&cfg, &m, est(11.0), 0, 0, &[], "standard");
        assert!(!no.allowed);
        assert!(no.reasons[0].contains("below the minimum 4.00 GB"), "{:?}", no.reasons);
        // ceiling = min(24 − 4, Metal 16) = 16 GB
        let no2 = check(&cfg, &FakeProbe::mac_mini_24gb(), est(6.0), (11.0 * GB) as u64, 0, &[], "small");
        assert!(no2.reasons.iter().any(|r| r.contains("ceiling 16.00 GB")), "{:?}", no2.reasons);
    }

    #[test]
    fn pressure_and_exclusive_large() {
        let cfg = Config::defaults_for(&FakeProbe::mac_mini_24gb());
        let mut m = FakeProbe::mac_mini_24gb();
        m.pressure = "warn".into();
        assert!(!check(&cfg, &m, est(1.0), 0, 0, &[], "small").allowed);
        let m = FakeProbe::mac_mini_24gb();
        let d = check(&cfg, &m, est(1.0), 0, 0, &["standard".into()], "large");
        assert!(!d.allowed);
        assert!(check(&cfg, &m, est(1.0), 0, 0, &["embed".into()], "large").allowed);
    }
}
