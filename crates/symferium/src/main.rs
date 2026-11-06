use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use symferium::config;
use symferium::probe::{Probe, SystemProbe};
use symferium::state::StateStore;
use symferium::Runtime;

#[derive(Parser)]
#[command(name = "symferium", version, about = "Local multi-model AI runtime for Apple Silicon")]
struct Cli {
    /// Config file (default: ~/Library/Application Support/Symferium/symferium.toml or $SYMFERIUM_CONFIG)
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the daemon: control API + UI, model ports, supervisor, governor.
    Serve {
        /// Do not restore the saved model set.
        #[arg(long)]
        no_restore: bool,
    },
    /// Print what the probe sees and the memory budget derived from it.
    Probe,
    /// Print the config path, or the whole config.
    Config {
        #[arg(long)]
        show: bool,
    },
    /// List models found in the models folders.
    Models,
    /// Memory estimate for a model from the library, without loading it.
    Estimate {
        model: String,
        #[arg(long)]
        ctx: Option<u32>,
        #[arg(long)]
        engine: Option<String>,
    },
    /// Manage the launchd LaunchAgent (start at login).
    Launchd {
        #[command(subcommand)]
        action: LaunchdCmd,
    },
    /// Ask the running daemon to bench these models one after another and print the table.
    Bench {
        models: Vec<String>,
        #[arg(long)]
        ctx: Option<u32>,
    },
}

#[derive(Subcommand)]
enum LaunchdCmd {
    Install,
    Uninstall,
    Print,
}

const TTS_SIDECAR: &str = include_str!("../../../sidecars/tts_server.py");

fn setup(cli: &Cli) -> Result<(PathBuf, config::Config, Arc<dyn Probe>)> {
    let probe: Arc<dyn Probe> = Arc::new(SystemProbe::new());
    let path = cli.config.clone().unwrap_or_else(config::default_config_path);
    let cfg = config::load_or_create(&path, &probe.snapshot())?;
    Ok((path, cfg, probe))
}

fn install_sidecars(cfg: &config::Config) -> Result<()> {
    let dir = config::expand(&cfg.paths.sidecar_dir);
    std::fs::create_dir_all(&dir)?;
    let p = dir.join("tts_server.py");
    if std::fs::read_to_string(&p).ok().as_deref() != Some(TTS_SIDECAR) {
        std::fs::write(p, TTS_SIDECAR)?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())).init();
    let cli = Cli::parse();
    match &cli.cmd {
        Cmd::Serve { no_restore } => {
            let (path, cfg, probe) = setup(&cli)?;
            install_sidecars(&cfg).context("installing sidecars")?;
            let state = StateStore::load(cfg.state_dir().join("state.json"));
            let addr = (cfg.server.host.clone(), cfg.server.control_port);
            let rt = Runtime::new(cfg, Some(path.clone()), probe, state);
            rt.spawn_watchdog();
            if !no_restore {
                rt.spawn_restore();
            }
            let listener = tokio::net::TcpListener::bind((addr.0.as_str(), addr.1)).await.with_context(|| format!("binding {}:{}", addr.0, addr.1))?;
            tracing::info!("Symferium on http://{}:{} (config {})", addr.0, addr.1, path.display());
            let app = symferium::api::router(rt.clone());
            let rt2 = rt.clone();
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    shutdown_signal().await;
                    tracing::info!("stopping all models");
                    rt2.unload_all().await;
                })
                .await?;
        }
        Cmd::Probe => {
            let (_, cfg, probe) = setup(&cli)?;
            let m = probe.snapshot();
            let b = symferium::governor::budget(&cfg, &m, 0, 0);
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({"machine": m, "budget": b}))?);
        }
        Cmd::Config { show } => {
            let (path, cfg, _) = setup(&cli)?;
            if *show {
                println!("{}", toml::to_string_pretty(&cfg)?);
            } else {
                println!("{}", path.display());
            }
        }
        Cmd::Models => {
            let (_, cfg, _) = setup(&cli)?;
            let lib = symferium::library::Library::new();
            for e in lib.scan(&cfg) {
                println!(
                    "{:<40} {:<9} {:<6} {:<8} {:>8.2} GB  {}{}",
                    e.id,
                    e.class,
                    e.format,
                    e.quant.unwrap_or_default(),
                    e.size_bytes as f64 / config::GB,
                    if e.flags.vision { "vision " } else { "" },
                    e.path.display()
                );
            }
        }
        Cmd::Estimate { model, ctx, engine } => {
            let (_, cfg, probe) = setup(&cli)?;
            let state = StateStore::load(cfg.state_dir().join("state.json"));
            let rt = Runtime::new(cfg, None, probe, state);
            let req = symferium::state::LoadRequest { model: model.clone(), ctx: *ctx, engine: engine.clone(), ..Default::default() };
            let p = rt.plan(&req)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({"engine": p.engine, "class": p.class, "ctx": p.ctx, "allowed": p.decision.allowed, "reasons": p.decision.reasons, "estimate": p.estimate, "budget": p.decision.budget}))?
            );
        }
        Cmd::Launchd { action } => {
            let (path, cfg, _) = setup(&cli)?;
            let bin = std::env::current_exe()?;
            match action {
                LaunchdCmd::Install => println!("installed {}", symferium::launchd::install(&cfg, &bin, &path)?.display()),
                LaunchdCmd::Uninstall => {
                    symferium::launchd::uninstall(&cfg)?;
                    println!("removed");
                }
                LaunchdCmd::Print => print!("{}", symferium::launchd::plist(&cfg, &bin, &path)),
            }
        }
        Cmd::Bench { models, ctx } => {
            let (_, cfg, _) = setup(&cli)?;
            let base = format!("http://{}:{}", cfg.server.host, cfg.server.control_port);
            let http = reqwest::Client::new();
            let auth = |rb: reqwest::RequestBuilder| if cfg.server.admin_token.is_empty() { rb } else { rb.bearer_auth(&cfg.server.admin_token) };
            let before: Vec<serde_json::Value> = auth(http.get(format!("{base}/api/bench"))).send().await?.json().await?;
            auth(http.post(format!("{base}/api/bench/compare"))).json(&serde_json::json!({"models": models, "ctx": ctx})).send().await?.error_for_status()?;
            eprintln!("benchmarking {} model(s) one at a time…", models.len());
            let mut seen_failures = 0;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                let now: Vec<serde_json::Value> = auth(http.get(format!("{base}/api/bench"))).send().await?.json().await?;
                let hist: Vec<serde_json::Value> = auth(http.get(format!("{base}/api/events/history"))).send().await?.json().await?;
                let failures = hist.iter().filter(|e| e["kind"] == "bench.failed").count();
                if failures > seen_failures {
                    for e in hist.iter().filter(|e| e["kind"] == "bench.failed").skip(seen_failures) {
                        eprintln!("failed: {}", e["data"]);
                    }
                    seen_failures = failures;
                }
                if now.len() - before.len() + seen_failures >= models.len() {
                    println!("{:<40} {:<12} {:>12} {:>12} {:>10}", "model", "engine", "prefill t/s", "decode t/s", "ttft ms");
                    for r in &now[before.len()..] {
                        println!(
                            "{:<40} {:<12} {:>12.1} {:>12.1} {:>10.0}",
                            r["model"].as_str().unwrap_or(""),
                            r["engine"].as_str().unwrap_or(""),
                            r["prefill_tps"].as_f64().unwrap_or(0.0),
                            r["decode_tps"].as_f64().unwrap_or(0.0),
                            r["ttft_ms"].as_f64().unwrap_or(0.0)
                        );
                    }
                    break;
                }
            }
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("signal handler");
        tokio::select! { _ = ctrl_c => {}, _ = term.recv() => {} }
    }
    #[cfg(not(unix))]
    ctrl_c.await.ok();
}
