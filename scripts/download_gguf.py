#!/usr/bin/env python3
"""Resumable multi-thread HTTP Range downloader for Hy4-Flash GGUFs.

Features:
- Pre-flight net free-disk check (accounts for already allocated blocks on disk)
- Immediate abort on ENOSPC (no pointless retry backoff)
- Multi-thread HTTP 206 Range download with atomic `.dlstate.json` resume
- Optional SHA-256 verification against Hugging Face X-Linked-ETag (`--verify-sha`)
- Optional offline reuse of already-complete local files (`--offline-ok`)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    import requests

DEFAULT_MAIN_URL = "https://huggingface.co/AngelSlim/Hy4-flash-GGUF/resolve/main/Hy4-flash-SPARK-4Bit.gguf"
DEFAULT_OUT = "/content/hy4-flash/Hy4-flash-SPARK-4Bit.gguf"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
DISK_SAFETY_MARGIN_BYTES = 256 * 1024 * 1024  # 256 MiB headroom for metadata/state


def _import_requests():
    import requests
    from requests.adapters import HTTPAdapter
    return requests, HTTPAdapter


def _same_origin(url_a: str, url_b: str) -> bool:
    a, b = urlsplit(url_a), urlsplit(url_b)
    return (a.scheme.lower(), a.netloc.lower()) == (b.scheme.lower(), b.netloc.lower())


def _extract_sha256(resp) -> str | None:
    """Extract 64-hex SHA-256 strictly from X-Linked-ETag across response and redirect chain.

    Do NOT fall back to generic ETag: Hugging Face Xet storage emits 64-hex CAS
    hashes in ETag that are not whole-file SHA-256 digests.
    """
    chain = [resp, *reversed(getattr(resp, "history", []) or [])]
    for r in chain:
        raw = (r.headers.get("X-Linked-ETag") or "").strip()
        if raw.startswith("W/"):
            raw = raw[2:].strip()
        raw = raw.strip('"')
        if SHA256_RE.match(raw):
            return raw.lower()
    return None


def _allocated_bytes(path: Path) -> int:
    """Return actual physical bytes currently allocated on disk (handles sparse files)."""
    if not path.exists():
        return 0
    st = path.stat()
    if hasattr(st, "st_blocks"):
        return int(st.st_blocks) * 512
    return int(st.st_size)


def verify_file_sha256(path: Path, expected_sha256: str) -> None:
    print(f"[dl] Verifying SHA-256 of {path.name}...", flush=True)
    h = hashlib.sha256()
    t0 = time.monotonic()
    with open(path, "rb") as f:
        while True:
            buf = f.read(8 * 1024 * 1024)
            if not buf:
                break
            h.update(buf)
    actual = h.hexdigest().lower()
    if actual != expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    print(f"[dl] SHA-256 OK ({actual[:16]}...) in {time.monotonic() - t0:.1f}s", flush=True)


class URLResolver:
    def __init__(self, source_url: str, hf_token: str | None) -> None:
        self.source_url = source_url
        self.hf_token = hf_token
        self.lock = threading.Lock()
        self.resolved_url: str | None = None
        self.resolved_at = 0.0

    def get_url(self, force_refresh: bool = False, stale_url: str | None = None) -> str:
        requests, _ = _import_requests()
        with self.lock:
            now = time.monotonic()
            if self.resolved_url:
                if not force_refresh and (now - self.resolved_at) < 1500:
                    return self.resolved_url
                if force_refresh and stale_url is not None and self.resolved_url != stale_url:
                    return self.resolved_url
            headers = {"Range": "bytes=0-0", "Accept-Encoding": "identity"}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
            with requests.Session() as sess:
                with sess.get(
                    self.source_url,
                    headers=headers,
                    stream=True,
                    timeout=30,
                    allow_redirects=True,
                ) as resp:
                    resp.raise_for_status()
                    self.resolved_url = resp.url
                    self.resolved_at = time.monotonic()
            return self.resolved_url


def fetch_remote_meta(url: str, hf_token: str | None) -> tuple[int, str | None]:
    requests, _ = _import_requests()
    headers = {"Range": "bytes=0-0", "Accept-Encoding": "identity"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    with requests.Session() as sess:
        with sess.get(url, headers=headers, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            if resp.status_code != 206:
                raise RuntimeError(
                    f"Server did not honor HTTP Range request (expected 206, got {resp.status_code})"
                )
            cr = resp.headers.get("Content-Range", "")
            if "/" not in cr:
                raise RuntimeError(f"Missing Content-Range header: {resp.headers}")
            try:
                total = int(cr.split("/")[-1])
            except ValueError as exc:
                raise RuntimeError(f"Invalid Content-Range total size: {cr!r}") from exc
            if total <= 0:
                raise RuntimeError(f"Invalid Content-Range total size: {cr!r}")
            return total, _extract_sha256(resp)


def _save_state(state_path: Path, total: int, chunk_bytes: int, done_set: set[int]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{state_path}.tmp.{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps({"total": total, "chunk_bytes": chunk_bytes, "done": sorted(done_set)}),
            encoding="utf-8",
        )
        os.replace(tmp, state_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def download(
    url: str,
    out_path: Path,
    state_path: Path,
    workers: int,
    chunk_mb: int,
    verify_sha: bool = False,
    offline_ok: bool = False,
) -> int:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if chunk_mb < 1:
        raise ValueError(f"chunk_mb must be >= 1, got {chunk_mb}")

    requests, HTTPAdapter = _import_requests()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        total, remote_sha256 = fetch_remote_meta(url, hf_token)
    except Exception as exc:
        if offline_ok and not verify_sha and out_path.exists() and not state_path.exists():
            sz = out_path.stat().st_size
            if sz > 0:
                print(
                    f"[warn] Remote metadata lookup failed ({exc}); reusing complete local file "
                    f"{out_path} ({sz/(1024**3):.3f} GiB) because --offline-ok is set.",
                    flush=True,
                )
                return 0
        raise

    if out_path.exists() and out_path.stat().st_size == total and not state_path.exists():
        print(f"[dl] Already complete: {out_path} ({total/(1024**3):.3f} GiB)", flush=True)
        if verify_sha:
            if remote_sha256:
                verify_file_sha256(out_path, remote_sha256)
            else:
                print("[warn] --verify-sha requested, but server returned no 64-hex SHA-256 ETag; skipping.", flush=True)
        return 0

    # Pre-flight disk space check accounting for physical blocks already allocated to out_path.
    alloc = min(_allocated_bytes(out_path), total)
    net_need = max(0, total - alloc) + DISK_SAFETY_MARGIN_BYTES
    free_bytes = shutil.disk_usage(out_path.parent).free
    if free_bytes < net_need:
        raise RuntimeError(
            f"Insufficient disk space in {out_path.parent}: need {net_need/(1024**3):.2f} GiB more "
            f"(target={total/(1024**3):.2f} GiB, already_allocated={alloc/(1024**3):.2f} GiB, "
            f"margin=0.25 GiB), but only {free_bytes/(1024**3):.2f} GiB free."
        )

    chunk_bytes = chunk_mb * 1024 * 1024
    tasks: list[tuple[int, int, int]] = []
    off = 0
    tid = 0
    while off < total:
        ln = min(chunk_bytes, total - off)
        tasks.append((tid, off, ln))
        tid += 1
        off += ln

    done_set: set[int] = set()
    if state_path.exists() and out_path.exists() and out_path.stat().st_size == total:
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict) and saved.get("total") == total and saved.get("chunk_bytes") == chunk_bytes:
                raw_done = saved.get("done", [])
                if isinstance(raw_done, list):
                    done_set = {
                        i for i in raw_done
                        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(tasks)
                    }
        except Exception:
            pass

    resolver = URLResolver(url, hf_token)
    resolver.get_url(force_refresh=True)

    # Persist normalized state file BEFORE ftruncate so even a SIGKILL mid-download
    # cannot leave an all-zero full-sized file without a matching .dlstate.json marker.
    _save_state(state_path, total, chunk_bytes, done_set)

    pending = [t for t in tasks if t[0] not in done_set]
    bytes_done_initial = sum(t[2] for t in tasks if t[0] in done_set)
    print(
        f"[dl] Target={out_path} ({total/(1024**3):.3f} GiB) | chunks={len(tasks)} pending={len(pending)} "
        f"({bytes_done_initial/(1024**3):.2f} GiB already done)",
        flush=True,
    )

    t_local = threading.local()
    lock = threading.Lock()
    sessions: list[requests.Session] = []
    cancel_event = threading.Event()
    bytes_session = 0
    since_flush = 0
    t_start = time.monotonic()

    def get_sess() -> requests.Session:
        s = getattr(t_local, "sess", None)
        if s is None:
            s = requests.Session()
            adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            t_local.sess = s
            with lock:
                sessions.append(s)
        return s

    fd = os.open(str(out_path), os.O_RDWR | os.O_CREAT, 0o644)
    stop_rep = threading.Event()
    rt: threading.Thread | None = None
    completed = False
    synced = False
    try:
        os.ftruncate(fd, total)

        def worker(task: tuple[int, int, int]) -> None:
            nonlocal bytes_session, since_flush
            tid_i, start, length = task
            sess = get_sess()
            refresh = False
            last_cdn: str | None = None
            for attempt in range(10):
                if cancel_event.is_set():
                    return
                try:
                    cdn = resolver.get_url(force_refresh=refresh, stale_url=last_cdn if refresh else None)
                    last_cdn = cdn
                    refresh = False
                    req_headers = {
                        "Range": f"bytes={start}-{start+length-1}",
                        "Accept-Encoding": "identity",
                    }
                    if hf_token and _same_origin(cdn, url):
                        req_headers["Authorization"] = f"Bearer {hf_token}"
                    with sess.get(
                        cdn,
                        headers=req_headers,
                        stream=True,
                        timeout=(15, 180),
                    ) as resp:
                        if resp.status_code in (401, 403, 410, 429):
                            if attempt >= 9:
                                resp.raise_for_status()
                            refresh = True
                            wait = min(2 ** min(attempt, 5), 30)
                            print(
                                f"[warn] chunk {tid_i} attempt {attempt+1}: HTTP {resp.status_code}; "
                                f"refreshing URL, retry {wait}s",
                                flush=True,
                            )
                            if cancel_event.wait(wait):
                                return
                            continue
                        resp.raise_for_status()
                        if resp.status_code != 206:
                            raise IOError(
                                f"Chunk {tid_i}: expected HTTP 206 Partial Content, got {resp.status_code}"
                            )
                        written = 0
                        for piece in resp.iter_content(chunk_size=1024 * 1024):
                            if cancel_event.is_set():
                                return
                            if not piece:
                                continue
                            if written + len(piece) > length:
                                raise IOError(
                                    f"Overlong read chunk {tid_i}: > {length} bytes"
                                )
                            mv = memoryview(piece)
                            off_p = 0
                            while off_p < len(piece):
                                n = os.pwrite(fd, mv[off_p:], start + written + off_p)
                                if n <= 0:
                                    raise IOError("pwrite returned 0")
                                off_p += n
                            written += len(piece)
                        if written != length:
                            raise IOError(f"Short read chunk {tid_i}: {written} != {length}")
                    with lock:
                        done_set.add(tid_i)
                        bytes_session += length
                        since_flush += 1
                        if since_flush >= 16:
                            since_flush = 0
                            _save_state(state_path, total, chunk_bytes, done_set)
                    return
                except OSError as exc:
                    if exc.errno == errno.ENOSPC:
                        cancel_event.set()
                        raise RuntimeError(f"Disk full (ENOSPC) while writing chunk {tid_i}: {exc}") from exc
                    if attempt >= 9:
                        cancel_event.set()
                        raise RuntimeError(f"Chunk {tid_i} failed: {exc}") from exc
                    if attempt >= 2:
                        refresh = True
                    wait = min(2 ** min(attempt, 5), 30)
                    print(f"[warn] chunk {tid_i} attempt {attempt+1}: {exc}; retry {wait}s", flush=True)
                    if cancel_event.wait(wait):
                        return
                except Exception as exc:
                    if attempt >= 9:
                        cancel_event.set()
                        raise RuntimeError(f"Chunk {tid_i} failed: {exc}") from exc
                    if attempt >= 2:
                        refresh = True
                    wait = min(2 ** min(attempt, 5), 30)
                    print(f"[warn] chunk {tid_i} attempt {attempt+1}: {exc}; retry {wait}s", flush=True)
                    if cancel_event.wait(wait):
                        return

        def reporter() -> None:
            while not stop_rep.wait(10.0):
                with lock:
                    done_tot = bytes_done_initial + bytes_session
                    sess_done = bytes_session
                el = max(time.monotonic() - t_start, 0.01)
                mbps = (sess_done / (1024 * 1024)) / el
                pct = 100.0 * done_tot / total
                eta = (total - done_tot) / max(sess_done / el, 1.0)
                print(
                    f"[progress] {done_tot/(1024**3):.2f}/{total/(1024**3):.2f} GiB ({pct:.1f}%) "
                    f"| speed={mbps:.1f} MiB/s | eta={eta:.0f}s",
                    flush=True,
                )

        rt = threading.Thread(target=reporter, daemon=True)
        rt.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(worker, t) for t in pending]
            try:
                for fut in concurrent.futures.as_completed(futs):
                    fut.result()
            except Exception:
                cancel_event.set()
                for f in futs:
                    f.cancel()
                raise
        completed = (len(done_set) == len(tasks))
        if completed:
            os.fsync(fd)
            synced = True
    finally:
        stop_rep.set()
        if rt is not None:
            rt.join(timeout=2.0)
        for s in sessions:
            s.close()
        try:
            if not synced:
                try:
                    os.fsync(fd)
                except OSError:
                    pass
        finally:
            os.close(fd)
        if completed and synced:
            state_path.unlink(missing_ok=True)
        else:
            try:
                _save_state(state_path, total, chunk_bytes, done_set)
            except OSError as exc:
                print(
                    f"[warn] failed to persist download state {state_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    if not completed:
        raise RuntimeError(f"Download incomplete: {len(done_set)}/{len(tasks)} chunks completed")

    el = max(time.monotonic() - t_start, 0.01)
    print(f"[dl] COMPLETED {total/(1024**3):.3f} GiB in {el:.1f}s ({(bytes_session/(1024*1024))/el:.1f} MiB/s avg)", flush=True)

    if verify_sha:
        if remote_sha256:
            verify_file_sha256(out_path, remote_sha256)
        else:
            print("[warn] --verify-sha requested, but server returned no 64-hex SHA-256 ETag; skipping.", flush=True)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Resumable HTTP Range downloader for Hy4-Flash GGUF weights.")
    ap.add_argument("--url", default=DEFAULT_MAIN_URL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--state", default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk-mb", type=int, default=32)
    ap.add_argument("--verify-sha", action="store_true", help="Verify downloaded file SHA-256 against HF X-Linked-ETag")
    ap.add_argument(
        "--offline-ok",
        action="store_true",
        help="Reuse an already-complete local file (no .dlstate.json) if remote metadata lookup fails",
    )
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.chunk_mb < 1:
        ap.error("--chunk-mb must be >= 1")

    out_path = Path(args.out)
    state_path = Path(args.state) if args.state else out_path.with_suffix(out_path.suffix + ".dlstate.json")

    return download(
        args.url,
        out_path,
        state_path,
        args.workers,
        args.chunk_mb,
        verify_sha=args.verify_sha,
        offline_ok=args.offline_ok,
    )


if __name__ == "__main__":
    raise SystemExit(main())
