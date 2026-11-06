//! launchd LaunchAgent: start at login, restart if it crashes, never block login.
//! The daemon itself starts immediately (the API is up in well under a second); the saved model
//! set loads afterwards, one model at a time, after autostart.delay_s.

use crate::config::Config;
use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

pub fn plist_path(cfg: &Config) -> PathBuf {
    crate::config::expand(&format!("~/Library/LaunchAgents/{}.plist", cfg.autostart.launchd_label))
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

pub fn plist(cfg: &Config, binary: &Path, config_path: &Path) -> String {
    let logs = cfg.logs_dir();
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{bin}</string>
    <string>serve</string>
    <string>--config</string>
    <string>{config}</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>{ptype}</string>
  <key>Nice</key><integer>{nice}</integer>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>RUST_LOG</key><string>info</string>
  </dict>
  <key>StandardOutPath</key><string>{out}</string>
  <key>StandardErrorPath</key><string>{err}</string>
</dict>
</plist>
"#,
        label = esc(&cfg.autostart.launchd_label),
        bin = esc(&binary.to_string_lossy()),
        config = esc(&config_path.to_string_lossy()),
        ptype = esc(&cfg.autostart.process_type),
        nice = cfg.autostart.nice,
        out = esc(&logs.join("daemon.out.log").to_string_lossy()),
        err = esc(&logs.join("daemon.err.log").to_string_lossy()),
    )
}

fn uid() -> u32 {
    unsafe { libc::getuid() }
}

pub fn install(cfg: &Config, binary: &Path, config_path: &Path) -> Result<PathBuf> {
    let path = plist_path(cfg);
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::create_dir_all(cfg.logs_dir())?;
    std::fs::write(&path, plist(cfg, binary, config_path))?;
    if cfg!(target_os = "macos") {
        let domain = format!("gui/{}", uid());
        let _ = std::process::Command::new("launchctl").args(["bootout", &format!("{domain}/{}", cfg.autostart.launchd_label)]).status();
        let st = std::process::Command::new("launchctl").args(["bootstrap", &domain, &path.to_string_lossy()]).status()?;
        if !st.success() {
            bail!("launchctl bootstrap failed ({st})");
        }
    }
    Ok(path)
}

pub fn uninstall(cfg: &Config) -> Result<()> {
    let path = plist_path(cfg);
    if cfg!(target_os = "macos") {
        let _ = std::process::Command::new("launchctl").args(["bootout", &format!("gui/{}/{}", uid(), cfg.autostart.launchd_label)]).status();
    }
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn plist_has_the_essentials() {
        let cfg = crate::config::Config::default();
        let p = super::plist(&cfg, std::path::Path::new("/usr/local/bin/symferium"), std::path::Path::new("/tmp/s.toml"));
        assert!(p.contains("<string>dev.symferium.daemon</string>"));
        assert!(p.contains("<key>RunAtLoad</key><true/>"));
        assert!(p.contains("/opt/homebrew/bin"));
    }
}
