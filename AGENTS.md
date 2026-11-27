# AGENTS.md — Coding Agent Execution Guide for `hy4-flash-colab-g4`

This repository is a **cloud-side Colab G4 Linux recipe** for deploying and verifying **Tencent Hy4-Flash** on a single **Google Colab G4** (`RTX PRO 6000 Blackwell Server Edition`, `sm_120a`, `97,887 MiB` VRAM, `176 GiB` DDR5 RAM) or an equivalent bare-metal Linux host.

It intentionally does **not** prescribe host-OS-specific tunnel wrappers; the recipe brings up a verified server on remote `127.0.0.1:8000`, and you expose or forward `:8000` using whatever transport the user already uses.

Read this file end-to-end before executing any cloud command.

---

## 1. Prerequisites & Human-in-the-Loop Gates

Stop and ask the user before proceeding when any of the following applies:
1. **Remote G4 shell access**: You must already have Linux shell execution access to a target Colab G4 runtime (provided by the user via SSH, terminal, or notebook). If you do not yet have shell access to a G4 host, stop and ask the user how to obtain or connect to one.
2. **Paid GPU allocation / OAuth**: Allocating a Colab G4 runtime bills `~10–11.2 CU/hr` (a fresh cold deployment consumes roughly `3.5–9 CU` before the first token). Interactive browser OAuth (`colab auth login` or Colab Web UI G4 selection) requires the user.
3. **Instance ownership**:
   - If the user already provided an active G4 runtime (or asked for a persistent server), treat it as **user-owned** — NEVER stop or delete it when done.
   - Only auto-stop a G4 runtime if the user explicitly agreed to an ephemeral benchmark/CI run.
4. **Ephemeral `/content` disk**: Releasing a Colab runtime wipes `/content` completely (compiled binary + `153.16 GiB` weights). Warn the user before stopping or recreating a runtime.

---

## 2. Hard Boundaries (Do Not Violate)

| Constraint | Hard Value / Rule | Failure Mode If Violated |
| --- | --- | --- |
| **GPU** | `RTX PRO 6000 Blackwell` (`sm_120a`, `>= 94,000 MiB` VRAM, exact `compute_cap == 12.0` when `HY4_CUDA_ARCH=120a`) | OOM during weight load (`88.65 GiB` GPU weights + KV + graph `~= 95,000 MiB`), or PTX/SASS mismatch on non-`sm_120` GPUs; `bootstrap-hy4.sh` Step 1 aborts immediately |
| **CUDA Toolchain** | `nvcc >= 12.8` (for `HY4_CUDA_ARCH=120a`) | `sm_120a` unsupported by older nvcc; Step 1 aborts immediately |
| **Host RAM** | `>= 165 GiB` `MemTotal` (`176 GiB` on Colab G4) | OOM killer during `--load-mode none` (`62.70 GiB` Engram + `3.71 GiB` `blk.42` + `32 GiB` cache-ram + DDR5 KV) |
| **Disk (`/content`)** | `>= 154 GiB` available (`153.24 GiB` exact footprint: `153.16 GiB` weights + `0.08 GiB` binary) | Download fails partway; Step 1 and `download_gguf.py` pre-check net free space before downloading |
| **Memory Load Mode** | **MUST** keep `--load-mode none --lazy-mode off` | Reverting to `mmap` causes Linux page-cache eviction of the `62.70 GiB` Engram table -> `10 t/s` major-fault cliff |
| **Tensor Overrides (`-ot`)** | `^token_embd\.weight$=CUDA0,^per_layer_token_embd\.weight$=CPU,^blk\.42\..*=CPU` | Dropping `token_embd=CUDA0` breaks DFly binding; dropping `blk.42=CPU` wastes `3.71 GiB` VRAM and OOMs |
| **CUDA Env Flags** | `GGML_CUDA_MOE_CONCURRENT=1`, `GGML_CUDA_GRAPH_OPT=1` | **NEVER** combine `GGML_CUDA_REUSE_Q8=1` with `GGML_CUDA_GRAPH_OPT=1` (~10% regression + greedy drift) |
| **Listen Port & Auth** | `127.0.0.1:8000` (never `:8080`); always keep API key auth enabled when exposing beyond loopback | Colab reserves `:8080`; `bootstrap-hy4.sh` auto-generates and persists a key to `/content/.hy4-g4/api_key` (0600) and refuses unauthenticated non-loopback binds |
| **`reasoning_effort`** | Allowed: `no_think`, `low`, `high`, `max` | Passing **`medium`** or **`xhigh`** returns **`HTTP 400`**; passing **`none`** silently falls back to `high` |
| **Cold Load Timeout** | Wall-clock `START_TIMEOUT_S=1800` (expect `11–24 min` on first start) | Killing/restarting early wastes CU and restarts full `151.35 GiB` read |

---

## 3. Cloud Deployment Playbook (Anti-Timeout / Anti-Loop)

> **CRITICAL**: Never run `bootstrap-hy4.sh` as a blocking foreground command inside a short-timeout shell tool (2–10 min timeouts will kill the caller and risk interrupting downloads). Always launch it **detached** with `setsid nohup` using **`>>`** (append, never `>` truncate), then poll `/content/.hy4-g4/state.json`.

### Step 1: Place This Repository on the G4 Host

Clone directly on the G4 runtime (while this repo is private, authenticate with a read-only GitHub PAT or upload/extract a tarball; keep `scripts/` and `patches/` as sibling directories):

```bash
git clone https://github.com/MELODY-TOYS/hy4-flash-colab-g4.git /content/hy4-flash-colab-g4
# While private (use a short-lived read-only PAT, then strip it from .git/config):
# git clone "https://${GITHUB_PAT}@github.com/MELODY-TOYS/hy4-flash-colab-g4.git" /content/hy4-flash-colab-g4
# git -C /content/hy4-flash-colab-g4 remote set-url origin https://github.com/MELODY-TOYS/hy4-flash-colab-g4.git
```

### Step 2: Launch `scripts/bootstrap-hy4.sh` Detached in the Background

Do **not** copy literal placeholder strings for `HY4_API_KEY` or `HF_TOKEN` (the script rejects literal placeholder keys, and an invalid `HF_TOKEN` causes Hugging Face HTTP 401). Leave `HY4_API_KEY` unset by default so `bootstrap-hy4.sh` auto-generates and reuses `/content/.hy4-g4/api_key` (mode `0600`) across re-runs without conflicting with shell env vars:

```bash
mkdir -p /content/.hy4-g4
# Optional — set only if you want a custom key (otherwise auto-generated into /content/.hy4-g4/api_key):
# export HY4_API_KEY="sk-$(openssl rand -hex 24)"
# Optional — uncomment ONLY if you have a real Hugging Face token:
# export HF_TOKEN="hf_..."

setsid nohup bash /content/hy4-flash-colab-g4/scripts/bootstrap-hy4.sh \
  >> /content/.hy4-g4/bootstrap.log 2>&1 < /dev/null &
```

Notes on idempotency, locking, and secrets:
- `bootstrap-hy4.sh` holds an exclusive `flock` on `/content/.hy4-g4/bootstrap.lock`. A concurrent second invocation prints the current `state.json` and exits with **exit code `75`** (non-zero — do NOT treat exit `75` as ready). Always use `>>` on `bootstrap.log` so an accidental concurrent call never truncates the active run's log.
- If `HY4_FORCE_RESTART=0` (default) and `/health` is already `200 {"status":"ok"}`, the script updates `state.json` to `ready` and exits `0` (printing a warning if the requested key/flags differ from the running instance).
- If `HY4_FORCE_RESTART=0` and the managed `llama-server` PID is already alive and still loading weights (`/health` returning `503`), the script **attaches and waits** instead of killing and restarting the 11–24 min load.
- `HY4_API_KEY` is persisted to `/content/.hy4-g4/api_key` (`0600`) and passed to `llama-server` via the `LLAMA_API_KEY` environment variable (never exposed in `ps` argv). Previous `/tmp/hy4-server.log` is rotated to `/tmp/hy4-server.log.prev` on restart.

### Step 3: Poll `/content/.hy4-g4/state.json` Until Terminal State

Poll every 30–60 seconds:

```bash
cat /content/.hy4-g4/state.json
tail -n 20 /content/.hy4-g4/bootstrap.log
```

**`state.json` Schema & State Machine**:
```json
{
  "stage": "init | env_check | build | download | start_server | ready",
  "step": 0..4,
  "total_steps": 4,
  "status": "running | ready | error",
  "detail": "Human-readable progress or exact failure reason",
  "updated_at": "ISO-8601 timestamp"
}
```

- **Expected duration per stage**:
  - `env_check` (step 1): `< 15s`
  - `build` (step 2): `2–5 min` (caches SHA-256 build signature in `/content/hy4-bin/llama-server.sig`)
  - `download` (step 3): `5–10 min` (`153.16 GiB` via resumable HTTP 206; complete files skip automatically)
  - `start_server` (step 4): **`11–24 min`** (`~11 min` warm right after download, `~20–24 min` cold disk; `detail` updates every ~10s with `elapsed`, `health` HTTP code, and live `vram`)
- **Stop polling when**:
  - `status == "ready"` (and `stage == "ready"`) -> proceed to Step 4 verification.
  - `status == "error"` -> stop immediately, inspect `detail`, `/content/.hy4-g4/bootstrap.log`, and `/tmp/hy4-server.log`, and fix root cause before re-running (if a previous server timed out wedged, re-run with `HY4_FORCE_RESTART=1`).
  - Stale-process check: if `status == "running"` and `updated_at` has not advanced for > 5 minutes AND `flock -n /content/.hy4-g4/bootstrap.lock true` succeeds, the previous bootstrap died with `SIGKILL`; inspect logs before restarting.

### Step 4: Run Cloud-Side Log & End-to-End Verification

Verify **both** startup invariants and real API behavior on the G4 host before exposing to clients:

1. **Startup log invariants** (must show `DFly attention = FA on`, `K_effective = 7`, `pairing = bound`, and zero fatal load/CUDA/OOM lines):
   ```bash
   grep -iE "dfly_print_i:|loaded multimodal|failed to load|CUDA error|out of memory" /tmp/hy4-server.log
   ```
2. **End-to-end API verification** (zero external Python dependencies; automatically loads `/content/.hy4-g4/api_key` if `--api-key` and `HY4_API_KEY` are omitted, runs every check independently, and prints full `[PASS]`/`[FAIL]`/`[SKIP]` summary):
   ```bash
   python3 /content/hy4-flash-colab-g4/scripts/verify.py \
     --base-url http://127.0.0.1:8000 \
     --api-key "$(cat /content/.hy4-g4/api_key)"
   # Optional concurrency probe (reports both aggregate and per-slot average TPS):
   python3 /content/hy4-flash-colab-g4/scripts/verify.py \
     --base-url http://127.0.0.1:8000 \
     --concurrency 8
   ```
3. **Post-verify DFly health counters** (must show `no_snapshot = 0, bad_snapshot = 0, bad_token = 0, compute_failed = 0`):
   ```bash
   grep -i "statistics.*draft-dfly" /tmp/hy4-server.log | tail -n 3
   ```

### Step 5: Expose `:8000` (Optional) & Deliver Integration Block

Expose remote `127.0.0.1:8000` via whichever transport the user prefers (SSH port forward, Tailscale, or detached `cloudflared` quick tunnel with streaming enabled to avoid Cloudflare's ~100s non-streaming origin timeout).

Deliver a concise, copy-pasteable summary containing:
- Verified Base URLs for OpenAI Chat (`/v1/chat/completions`), OpenAI Responses (`/v1/responses`), and Anthropic Messages (`/v1/messages`, base URL without `/v1`), plus `NO_PROXY` note if the user has a local proxy;
- Model ID (`hy4-flash`), active API Key (read from `/content/.hy4-g4/api_key`, never committed to git), context limit (`262144`), concurrency (`8`), vision status, and `K_effective`;
- Client snippets for Claude Code / Codex (explicit `model_reasoning_effort = "high"` + `export HY4_API_KEY=...`) and the mandatory `reasoning_effort` warning (`no_think` / `low` / `high` / `max` only; never `medium` or `xhigh`).

---

## 4. Environment Variable Contract (`scripts/bootstrap-hy4.sh`)

| Variable | Default | Purpose & Safe Adjustment Range |
| --- | --- | --- |
| `HY4_API_KEY` | auto-generated (saved to `HY4_API_KEY_FILE`) | Bearer / `x-api-key` secret passed via `LLAMA_API_KEY` env var. Mandatory for non-loopback `HY4_HOST` or public tunnels. |
| `HY4_API_KEY_FILE` | `/content/.hy4-g4/api_key` | Mode `0600` file persisting the active API key across shells/cells (`verify.py` reads this automatically). |
| `HF_TOKEN` | `""` | Optional Hugging Face token to avoid anonymous download rate limits. Never set a placeholder value. |
| `HY4_DFLY_N_MAX` | `7` | DFly speculative draft tokens per step (`1..7`; draft `block_size=8`). `7` (`W=8`) hits the Blackwell `NVFP4 MMVQ` fast-path cap. |
| `HY4_ENABLE_VISION` | `1` | Set `0` to skip `mmproj-Hy4-flash-bf16.gguf` (`0.97 GiB`) and disable image input (`verify.py` auto-skips vision checks). |
| `HY4_MMPROJ_OFFLOAD` | `0` | Keep `0` (`--no-mmproj-offload`) on 96 GiB G4 so vision does not steal VRAM from the KV pool. |
| `HY4_FORCE_RESTART` | `0` | Default `0` keeps `bootstrap-hy4.sh` idempotent (and attaches to an already-loading managed server). Set `1` to stop managed server and relaunch. |
| `HY4_PARALLEL` | `8` | Concurrent slot count (`-np`). Keep `<= 8` on 96 GiB VRAM. |
| `HY4_PER_SLOT_CTX` | `262144` | Max context per slot (`256K`). |
| `HY4_CACHE_RAM_MIB` | `32768` | Host RAM prompt-cache pool (`32 GiB`). Verified post-load against `MemAvailable >= CACHE_RAM + RESERVE`. |
| `HY4_HOST_RESERVE_GIB` | `12` | Post-startup host RAM safety margin (`GiB`). |
| `HY4_HOST` / `HY4_PORT` | `127.0.0.1` / `8000` | Server bind address & port (never `:8080` on Colab). |
| `HY4_ALLOW_UNAUTHENTICATED` | `0` | Set `1` only to explicitly disable API key generation/auth guard on an isolated private network. |
| `HY4_MODEL_ALIAS` | `hy4-flash` | Served model ID returned by `/v1/models`. |
| `HY4_START_TIMEOUT_S` | `1800` | Wall-clock startup health-check timeout in seconds. |
| `HY4_DL_WORKERS` | `12` | Concurrent HTTP 206 chunk workers for the `151.35 GiB` main GGUF. |
| `HY4_VERIFY_SHA` | `0` | Set `1` to verify downloaded GGUF files' SHA-256 strictly against HF `X-Linked-ETag` (adds `~80s–4 min` full-disk read pass). |
| `HY4_BUILD_JOBS` | `min(nproc, 24)` | Ninja parallel compile jobs. |
| `HY4_CUDA_ARCH` | `120a` | `CMAKE_CUDA_ARCHITECTURES` target (`120a` requires exact `compute_cap == 12.0` Blackwell RTX PRO 6000). |
| `HY4_NVCC` | auto (`command -v nvcc`) | Explicit path to `nvcc` binary (`>= 12.8` required for `120a`). |
| `HY4_MIN_VRAM_MIB` / `HY4_MIN_RAM_GIB` | `94000` / `165` | Step 1 hardware pre-flight thresholds. |
| `HY4_SKIP_HW_CHECK` | `0` | Set `1` to bypass Step 1 hardware/disk pre-flight checks (debugging only). |
| `MAIN_URL` / `DFLY_URL` / `MMPROJ_URL` | `AngelSlim/Hy4-flash-GGUF` HF URLs | Override download URLs (e.g. for an internal mirror). |
| `HY4_WORK_DIR` / `HY4_BIN_DIR` / `HY4_MODEL_DIR` / `HY4_SERVER_LOG` | `/content/.hy4-g4`, `/content/hy4-bin`, `/content/hy4-flash`, `/tmp/hy4-server.log` | Override filesystem paths on non-Colab bare-metal Linux hosts. |

---

## 5. Diagnostics & Troubleshooting Decision Tree

- **`bootstrap-hy4.sh` exits immediately with code `75`**:
  - Cause: Another `bootstrap-hy4.sh` instance currently holds `/content/.hy4-g4/bootstrap.lock`.
  - Fix: Do NOT start another copy; poll `/content/.hy4-g4/state.json` and `/content/.hy4-g4/bootstrap.log`.
- **Step 1 fails with VRAM / compute capability / RAM / nvcc / disk error**:
  - Cause: Runtime is not a 96 GiB Blackwell `sm_120` G4 (e.g., Colab fell back to A100/L4), CUDA < 12.8, or `/content` has `< 154 GiB` available.
  - Fix: Stop and inform the user; do not bypass checks with `HY4_SKIP_HW_CHECK=1` on undersized GPUs.
- **`bootstrap-hy4.sh` warns that API key / flags differ from running server**:
  - Cause: Server was already healthy and `HY4_FORCE_RESTART=0`.
  - Fix: Re-run with `HY4_FORCE_RESTART=1` if you actually want to apply the new key/flags, or use the existing key from `/content/.hy4-g4/api_key`.
- **`HTTP 400` on Chat/Responses request (common with Codex)**:
  - Cause: Client sent `reasoning_effort: "medium"` or `"xhigh"`.
  - Fix: Set `model_reasoning_effort = "high"` (or `"low"`), or use `chat_template_kwargs.reasoning_effort = "no_think"`.
- **Decode speed drops to `~10 t/s` with `0%` GPU util and high `pgmajfault/s`**:
  - Cause: Server was launched without `--load-mode none --lazy-mode off` (mmap Engram thrashing).
  - Fix: Always launch via `scripts/bootstrap-hy4.sh`.
- **`Permission denied` when running manually built test binaries in `/dev/shm`**:
  - Cause: Colab mounts `/dev/shm` with `noexec`.
  - Fix: Copy test binaries (e.g. `test-dfly`) to `/tmp/` and `chmod +x` before running.
- **`HTTP 429 Already-active SSH session` when using `colab ssh`**:
  - Cause: Multiple concurrent `colab ssh` processes targeting the same Colab runtime.
  - Fix: Reuse a single SSH session / control socket instead of spawning parallel `colab ssh` commands.
- **Running CUDA unit tests for patch verification**:
  - Configure with `-DLLAMA_BUILD_TESTS=ON`, build `--target test-dfly`, copy to `/tmp/test-dfly`, and run `CUDA_VISIBLE_DEVICES=0 /tmp/test-dfly "hy4_g4_extensions.*"`. Expect `HY4_dfly_capture_pull_rows_CUDA0` and `HY4_dfly_flash_attn_576_CUDA0` to pass.

---

## 6. Cloud Billing Stop-Loss

Colab G4 bills `~10–11.2 CU/hr` whenever a runtime assignment exists, **even if no tunnel is connected and no requests are running**:

1. Disconnecting a browser tab, SSH tunnel, or `cloudflared` **does NOT stop billing**.
2. When stopping a runtime (only with user consent for user-owned instances), use the Colab Web UI (**"Disconnect and delete runtime"**) or public CLI commands (`colab stop -s <session>` followed by `colab sessions`).
3. Always remind the user to double-check the Colab **Compute Units / Usage** page after stopping or after a failed/timed-out G4 allocation to confirm zero active billable runtimes remain.
