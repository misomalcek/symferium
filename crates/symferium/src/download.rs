//! Resumable, checksummed downloads from Hugging Face into the models folder.

use crate::catalog::FileRef;
use crate::events::now_ms;
use crate::runtime::Runtime;
use anyhow::{anyhow, bail, Context, Result};
use futures::StreamExt;
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Serialize, Default)]
pub struct JobStatus {
    /// queued, running, verifying, done, failed, cancelled
    pub state: String,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub current_file: Option<String>,
    pub speed_bps: f64,
    pub error: Option<String>,
    pub started_ms: u64,
    pub finished_ms: Option<u64>,
}

pub struct Job {
    pub id: String,
    pub repo: String,
    pub dest: PathBuf,
    pub files: Vec<FileRef>,
    pub status: RwLock<JobStatus>,
    pub cancel: CancellationToken,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobView {
    pub id: String,
    pub repo: String,
    pub dest: PathBuf,
    pub files: Vec<FileRef>,
    pub status: JobStatus,
}

#[derive(Default)]
pub struct Downloads {
    jobs: Mutex<BTreeMap<String, Arc<Job>>>,
    running: AtomicUsize,
    counter: AtomicUsize,
}

impl Downloads {
    pub fn list(&self) -> Vec<JobView> {
        self.jobs
            .lock()
            .values()
            .map(|j| JobView { id: j.id.clone(), repo: j.repo.clone(), dest: j.dest.clone(), files: j.files.clone(), status: j.status.read().clone() })
            .collect()
    }
    pub fn cancel(&self, id: &str) -> bool {
        match self.jobs.lock().get(id) {
            Some(j) => {
                j.cancel.cancel();
                true
            }
            None => false,
        }
    }
}

fn safe_rel(p: &str) -> Result<PathBuf> {
    let pb = PathBuf::from(p);
    if pb.is_absolute() || pb.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        bail!("unsafe file path {p}");
    }
    Ok(pb)
}

/// Queue a download of `files` from `repo`. Returns the job id immediately.
pub fn start(rt: &Arc<Runtime>, repo: &str, files: Vec<FileRef>) -> Result<String> {
    if files.is_empty() {
        bail!("nothing to download");
    }
    if !repo.chars().all(|c| c.is_ascii_alphanumeric() || "-_./".contains(c)) || repo.contains("..") {
        bail!("invalid repo name {repo}");
    }
    for f in &files {
        safe_rel(&f.path)?;
    }
    let cfg = rt.cfg();
    let dest = cfg.models_dir().join(repo);
    let n = rt.downloads.counter.fetch_add(1, Ordering::Relaxed) + 1;
    let id = format!("dl{n}");
    let job = Arc::new(Job {
        id: id.clone(),
        repo: repo.into(),
        dest,
        status: RwLock::new(JobStatus {
            state: "queued".into(),
            bytes_total: files.iter().filter_map(|f| f.size).sum(),
            started_ms: now_ms(),
            ..Default::default()
        }),
        files,
        cancel: CancellationToken::new(),
    });
    rt.downloads.jobs.lock().insert(id.clone(), job.clone());
    rt.events.emit("download.queued", serde_json::json!({"id": id, "repo": repo}));
    let rt2 = rt.clone();
    tokio::spawn(async move {
        // Respect download.concurrent.
        loop {
            let limit = rt2.cfg().download.concurrent.max(1);
            let cur = rt2.downloads.running.load(Ordering::SeqCst);
            if cur < limit && rt2.downloads.running.compare_exchange(cur, cur + 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                break;
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(500)) => {}
                _ = job.cancel.cancelled() => { job.status.write().state = "cancelled".into(); return; }
            }
        }
        let res = run_job(&rt2, &job).await;
        rt2.downloads.running.fetch_sub(1, Ordering::SeqCst);
        let mut st = job.status.write();
        st.finished_ms = Some(now_ms());
        match res {
            Ok(()) => {
                st.state = "done".into();
                drop(st);
                rt2.library.scan(&rt2.cfg());
                rt2.events.emit("download.done", serde_json::json!({"id": job.id, "repo": job.repo, "dest": job.dest}));
            }
            Err(e) => {
                st.state = if job.cancel.is_cancelled() { "cancelled" } else { "failed" }.into();
                st.error = Some(format!("{e:#}"));
                drop(st);
                rt2.events.emit("download.failed", serde_json::json!({"id": job.id, "repo": job.repo, "error": format!("{e:#}")}));
            }
        }
    });
    Ok(id)
}

async fn run_job(rt: &Arc<Runtime>, job: &Arc<Job>) -> Result<()> {
    let cfg = rt.cfg();
    job.status.write().state = "running".into();
    let base = cfg.catalog.hf_endpoint.trim_end_matches('/').to_string();
    let mut done_before = 0u64;
    for f in &job.files {
        let url = format!("{base}/{}/resolve/main/{}", job.repo, f.path);
        let dest = job.dest.join(safe_rel(&f.path)?);
        job.status.write().current_file = Some(f.path.clone());
        let mut attempt = 0;
        loop {
            let j = job.clone();
            let progress = move |file_done: u64, speed: f64| {
                let mut st = j.status.write();
                st.bytes_done = done_before + file_done;
                st.speed_bps = speed;
            };
            match fetch_file(&rt.http, &url, &dest, f, &cfg.catalog.hf_token, cfg.download.verify_sha256, &job.cancel, progress, cfg.download.timeout_s).await {
                Ok(()) => break,
                Err(e) if job.cancel.is_cancelled() => return Err(e),
                Err(e) if attempt < cfg.download.retries && !format!("{e:#}").contains("sha256 mismatch") && !format!("{e:#}").contains("HTTP 4") => {
                    attempt += 1;
                    tracing::warn!("download {} attempt {attempt}: {e:#}", f.path);
                    tokio::time::sleep(Duration::from_secs((1u64 << attempt.min(5)).min(30))).await;
                }
                Err(e) => return Err(e),
            }
        }
        done_before += dest.metadata().map(|m| m.len()).unwrap_or(0);
        job.status.write().bytes_done = done_before;
    }
    Ok(())
}

async fn sha256_of(path: &Path) -> Result<String> {
    let mut f = tokio::fs::File::open(path).await?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// Download one file to `dest`, resuming from `dest.part` with an HTTP Range request.
#[allow(clippy::too_many_arguments)]
pub async fn fetch_file(
    http: &reqwest::Client,
    url: &str,
    dest: &Path,
    f: &FileRef,
    token: &str,
    verify: bool,
    cancel: &CancellationToken,
    progress: impl Fn(u64, f64),
    timeout_s: u64,
) -> Result<()> {
    if let Some(d) = dest.parent() {
        tokio::fs::create_dir_all(d).await?;
    }
    // Already complete?
    if let (Ok(md), Some(size)) = (dest.metadata(), f.size) {
        if md.len() == size && (!verify || f.sha256.is_none() || sha256_of(dest).await.ok() == f.sha256) {
            progress(size, 0.0);
            return Ok(());
        }
    }
    let part = dest.with_file_name(format!("{}.part", dest.file_name().unwrap().to_string_lossy()));
    let mut have = part.metadata().map(|m| m.len()).unwrap_or(0);
    if f.size.is_some_and(|s| have > s) {
        tokio::fs::remove_file(&part).await.ok();
        have = 0;
    }
    let mut rb = http.get(url).timeout(Duration::from_secs(60 * 60 * 6)).header("user-agent", "symferium");
    if !token.is_empty() {
        rb = rb.bearer_auth(token);
    }
    if have > 0 && f.size != Some(have) {
        rb = rb.header("range", format!("bytes={have}-"));
    }
    let resp = if f.size == Some(have) && have > 0 {
        None
    } else {
        let r = tokio::time::timeout(Duration::from_secs(timeout_s), rb.send()).await.map_err(|_| anyhow!("timeout connecting"))??;
        if r.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            tokio::fs::remove_file(&part).await.ok();
            bail!("range not satisfiable; restarting file");
        }
        if !r.status().is_success() {
            bail!("GET {url}: HTTP {}", r.status());
        }
        Some(r)
    };
    if let Some(resp) = resp {
        let append = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        if !append {
            have = 0;
        }
        let mut out = tokio::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(&part).await.context("opening .part")?;
        let mut stream = resp.bytes_stream();
        let start = Instant::now();
        let mut got = 0u64;
        let mut last = Instant::now();
        loop {
            let next = tokio::select! {
                _ = cancel.cancelled() => bail!("cancelled"),
                n = tokio::time::timeout(Duration::from_secs(timeout_s), stream.next()) => n.map_err(|_| anyhow!("stalled for {timeout_s} s"))?,
            };
            let Some(chunk) = next else { break };
            let chunk = chunk?;
            out.write_all(&chunk).await?;
            got += chunk.len() as u64;
            if last.elapsed() > Duration::from_millis(500) {
                progress(have + got, got as f64 / start.elapsed().as_secs_f64().max(0.001));
                last = Instant::now();
            }
        }
        out.flush().await?;
        have += got;
    }
    if let Some(size) = f.size {
        if have != size {
            bail!("incomplete: {have} of {size} bytes");
        }
    }
    if verify {
        if let Some(want) = &f.sha256 {
            let got = sha256_of(&part).await?;
            if &got != want {
                tokio::fs::remove_file(&part).await.ok();
                bail!("sha256 mismatch for {}: expected {want}, got {got}", f.path);
            }
        }
    }
    tokio::fs::rename(&part, dest).await?;
    progress(have, 0.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;

    async fn serve(data: Arc<Vec<u8>>) -> String {
        let app = axum::Router::new().route(
            "/{*p}",
            axum::routing::get(move |headers: HeaderMap| {
                let data = data.clone();
                async move {
                    if let Some(r) = headers.get("range").and_then(|v| v.to_str().ok()) {
                        let from: usize = r.trim_start_matches("bytes=").trim_end_matches('-').parse().unwrap();
                        return (StatusCode::PARTIAL_CONTENT, data[from..].to_vec()).into_response();
                    }
                    (StatusCode::OK, data.to_vec()).into_response()
                }
            }),
        );
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn resumes_and_verifies() {
        let data: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        let sha = hex::encode(Sha256::digest(&data));
        let base = serve(Arc::new(data.clone())).await;
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("m.gguf");
        // A previous attempt left half the file.
        std::fs::write(tmp.path().join("m.gguf.part"), &data[..150_000]).unwrap();
        let f = FileRef { path: "m.gguf".into(), size: Some(data.len() as u64), sha256: Some(sha.clone()) };
        let http = reqwest::Client::new();
        fetch_file(&http, &format!("{base}/m.gguf"), &dest, &f, "", true, &CancellationToken::new(), |_, _| {}, 10).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), data);
        assert!(!tmp.path().join("m.gguf.part").exists());

        let bad = FileRef { sha256: Some("00".repeat(32)), path: "x.gguf".into(), ..f };
        let e = fetch_file(&http, &format!("{base}/x.gguf"), &tmp.path().join("x.gguf"), &bad, "", true, &CancellationToken::new(), |_, _| {}, 10).await.unwrap_err();
        assert!(e.to_string().contains("sha256 mismatch"));
        assert!(!tmp.path().join("x.gguf").exists());
    }

    #[test]
    fn rejects_path_traversal() {
        assert!(safe_rel("../etc/passwd").is_err());
        assert!(safe_rel("/abs").is_err());
        assert!(safe_rel("sub/model.gguf").is_ok());
    }
}
