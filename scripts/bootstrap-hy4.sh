#!/usr/bin/env bash
# Idempotent Google Colab G4 (RTX PRO 6000 Blackwell 96 GiB) bootstrap for Tencent Hy4-Flash:
#   SPARK-4Bit (Q8_0 trunk + NVFP4 experts) + external DFly Q8_0 (K=7 + native FP16 FA)
#   + tiered 8x256K KV (F16 SWA + Q8_0 VRAM/DDR5 MLA) + CPU mmproj vision.
# Keeps /content total footprint at 153.24 GiB (requires >= 154 GiB available space on /content).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/patches"

WORK_DIR="${HY4_WORK_DIR:-/content/.hy4-g4}"
BIN_DIR="${HY4_BIN_DIR:-/content/hy4-bin}"
MODEL_DIR="${HY4_MODEL_DIR:-/content/hy4-flash}"
STATE_FILE="${WORK_DIR}/state.json"
LOCK_FILE="${WORK_DIR}/bootstrap.lock"
SERVER_LOG="${HY4_SERVER_LOG:-/tmp/hy4-server.log}"
SERVER_PID_FILE="${WORK_DIR}/llama-server.pid"
API_KEY_FILE="${HY4_API_KEY_FILE:-${WORK_DIR}/api_key}"
CONFIG_SIG_FILE="${WORK_DIR}/server.sig"
BIN_SIG_FILE="${BIN_DIR}/llama-server.sig"

LLAMA_COMMIT="2cbca389f515df455412dafa07e2b1359cc72a6d"
MAIN_GGUF="${MODEL_DIR}/Hy4-flash-SPARK-4Bit.gguf"
DFLY_GGUF="${MODEL_DIR}/hy4-flash_dfly_Q8_0.gguf"
MMPROJ_GGUF="${MODEL_DIR}/mmproj-Hy4-flash-bf16.gguf"
MAIN_URL="${MAIN_URL:-https://huggingface.co/AngelSlim/Hy4-flash-GGUF/resolve/main/Hy4-flash-SPARK-4Bit.gguf}"
DFLY_URL="${DFLY_URL:-https://huggingface.co/AngelSlim/Hy4-flash-GGUF/resolve/main/hy4-flash_dfly_Q8_0.gguf}"
MMPROJ_URL="${MMPROJ_URL:-https://huggingface.co/AngelSlim/Hy4-flash-GGUF/resolve/main/mmproj-Hy4-flash-bf16.gguf}"

ENABLE_VISION="${HY4_ENABLE_VISION:-1}"
MMPROJ_OFFLOAD="${HY4_MMPROJ_OFFLOAD:-0}"
FORCE_RESTART="${HY4_FORCE_RESTART:-0}"
CUDA_ARCH="${HY4_CUDA_ARCH:-120a}"
VERIFY_SHA="${HY4_VERIFY_SHA:-0}"
SKIP_HW_CHECK="${HY4_SKIP_HW_CHECK:-0}"
ALLOW_UNAUTHENTICATED="${HY4_ALLOW_UNAUTHENTICATED:-0}"

MIN_VRAM_MIB="${HY4_MIN_VRAM_MIB:-94000}"
MIN_RAM_GIB="${HY4_MIN_RAM_GIB:-165}"

HOST="${HY4_HOST:-127.0.0.1}"
PORT="${HY4_PORT:-8000}"
ALIAS="${HY4_MODEL_ALIAS:-hy4-flash}"
USER_SUPPLIED_API_KEY="${HY4_API_KEY:-}"
API_KEY="${USER_SUPPLIED_API_KEY}"
N_PARALLEL="${HY4_PARALLEL:-8}"
PER_SLOT_CTX="${HY4_PER_SLOT_CTX:-262144}"
DFLY_N_MAX="${HY4_DFLY_N_MAX:-7}"
CACHE_RAM_MIB="${HY4_CACHE_RAM_MIB:-32768}"
HOST_RESERVE_GIB="${HY4_HOST_RESERVE_GIB:-12}"
START_TIMEOUT_S="${HY4_START_TIMEOUT_S:-1800}"
DL_WORKERS="${HY4_DL_WORKERS:-12}"
NPROC_AVAIL="$(nproc)"
BUILD_JOBS="${HY4_BUILD_JOBS:-$(( NPROC_AVAIL > 24 ? 24 : NPROC_AVAIL ))}"

resolve_probe_host() {
  local item raw=""
  local IFS=','
  local -a items=()
  read -r -a items <<< "$1" || true
  for item in "${items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [[ -n "${item}" ]]; then
      raw="${item}"
      break
    fi
  done
  case "${raw}" in
    ""|0.0.0.0) echo "127.0.0.1" ;;
    "::"|"[::]") echo "[::1]" ;;
    *.sock) echo "${raw}" ;;
    \[*\]) echo "${raw}" ;;
    *:*) echo "[${raw}]" ;;
    *) echo "${raw}" ;;
  esac
}

is_loopback_host_list() {
  local item
  local IFS=','
  local -a items=()
  read -r -a items <<< "$1" || true
  if [[ "${#items[@]}" -eq 0 ]]; then
    return 0
  fi
  for item in "${items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    case "${item}" in
      ""|127.0.0.1|localhost|"::1"|"[::1]"|*.sock) ;;
      *) return 1 ;;
    esac
  done
  return 0
}

is_placeholder_secret() {
  local v="${1:-}"
  case "${v}" in
    "" ) return 1 ;;
    *replace-with*|*STRONG_RANDOM*|*OPTIONAL_HF_TOKEN*|*YOUR_API_KEY*|*YOUR_KEY*|sk-your-key|sk-xxx|hf_xxx|changeme|"<"*">")
      return 0 ;;
    *) return 1 ;;
  esac
}

compute_runtime_sig() {
  local key_val="$1"
  python3 - "$HOST" "$PORT" "$ALIAS" "$N_PARALLEL" "$PER_SLOT_CTX" "$DFLY_N_MAX" \
    "$ENABLE_VISION" "$MMPROJ_OFFLOAD" "$CACHE_RAM_MIB" "$key_val" <<'PYEOF'
import hashlib, json, sys
fields = sys.argv[1:10]
key_hash = hashlib.sha256(sys.argv[10].encode("utf-8")).hexdigest()[:16] if sys.argv[10] else "none"
payload = json.dumps({"fields": fields, "key_sha": key_hash}, separators=(",", ":"))
print(hashlib.sha256(payload.encode("utf-8")).hexdigest())
PYEOF
}

write_secret_file() {
  local path="$1" val="$2"
  mkdir -p "$(dirname "${path}")"
  (
    umask 077
    local tmp="${path}.tmp.$$"
    printf '%s\n' "${val}" > "${tmp}"
    chmod 600 "${tmp}"
    mv -f "${tmp}" "${path}"
  )
}

PROBE_HOST="$(resolve_probe_host "${HOST}")"
if [[ "${PROBE_HOST}" == *.sock ]]; then
  ENDPOINT_URL="unix://${PROBE_HOST}/v1"
else
  ENDPOINT_URL="http://${PROBE_HOST}:${PORT}/v1"
fi

mkdir -p "${WORK_DIR}" "${BIN_DIR}" "${MODEL_DIR}"

LOCK_ACQUIRED=0
CURRENT_STAGE="init"
CURRENT_STEP=0
LAST_STATUS=""
BUILD_TMP_DIR=""

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[bootstrap-hy4] Another bootstrap process is already running (lock: ${LOCK_FILE})." >&2
  if [[ -f "${STATE_FILE}" ]]; then
    echo "[bootstrap-hy4] Current state: $(cat "${STATE_FILE}" 2>/dev/null || true)" >&2
  fi
  exit 75
fi
LOCK_ACQUIRED=1

write_state() {
  local stage="$1" step="$2" total="$3" status="$4" detail="$5"
  CURRENT_STAGE="$stage"
  CURRENT_STEP="$step"
  LAST_STATUS="$status"
  python3 - "$STATE_FILE" "$stage" "$step" "$total" "$status" "$detail" <<'PYEOF'
import json, os, sys
from datetime import datetime, timezone
p, stage, step, total, status, detail = sys.argv[1:7]
tmp = f"{p}.tmp.{os.getpid()}"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump({
        "stage": stage, "step": int(step), "total_steps": int(total),
        "status": status, "detail": detail,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }, f, ensure_ascii=False)
os.replace(tmp, p)
PYEOF
}

fail_with_error() {
  local msg="$1"
  echo "[bootstrap-hy4] ERROR: ${msg}" >&2
  write_state "${CURRENT_STAGE}" "${CURRENT_STEP}" 4 "error" "${msg}" || true
  exit 1
}

on_err() {
  local rc=$?
  local line="${BASH_LINENO[0]:-?}"
  trap - ERR
  echo "[bootstrap-hy4] ERROR at stage=${CURRENT_STAGE} line=${line} (rc=${rc})" >&2
  if [[ "${LAST_STATUS}" != "error" ]]; then
    write_state "${CURRENT_STAGE}" "${CURRENT_STEP}" 4 "error" "Stage ${CURRENT_STAGE} failed (line ${line}, rc=${rc})" || true
  fi
  exit "${rc}"
}

on_signal() {
  local sig="$1" code="$2"
  trap - ERR HUP INT TERM
  echo "[bootstrap-hy4] Interrupted by SIG${sig} at stage=${CURRENT_STAGE}" >&2
  write_state "${CURRENT_STAGE}" "${CURRENT_STEP}" 4 "error" "Interrupted by SIG${sig} at stage=${CURRENT_STAGE}" || true
  exit "${code}"
}

on_exit() {
  local rc=$?
  if [[ "${LOCK_ACQUIRED}" == "1" ]]; then
    if [[ -n "${BUILD_TMP_DIR}" && -d "${BUILD_TMP_DIR}" ]]; then
      rm -rf "${BUILD_TMP_DIR}" || true
    fi
    rm -f "${BIN_DIR}/llama-server.new" || true
    if [[ "${LAST_STATUS}" != "ready" && "${LAST_STATUS}" != "error" ]]; then
      write_state "${CURRENT_STAGE}" "${CURRENT_STEP}" 4 "error" "Bootstrap exited unexpectedly at stage=${CURRENT_STAGE} (rc=${rc})" || true
    fi
  fi
}

trap on_err ERR
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap on_exit EXIT

# Clean stale temporary build directories left behind by any previous SIGKILL.
cleanup_stale_build_dirs() {
  local base d pid
  for base in /tmp /dev/shm; do
    for d in "${base}"/hy4-build.*; do
      [[ -d "${d}" ]] || continue
      pid="${d##*.}"
      if [[ "${pid}" =~ ^[0-9]+$ ]] && ! kill -0 "${pid}" 2>/dev/null; then
        echo "[bootstrap-hy4] Cleaning stale build directory: ${d}"
        rm -rf "${d}" || true
      fi
    done
  done
}
cleanup_stale_build_dirs

# Reject literal placeholder secrets that would silently weaken auth or cause HF 401.
if is_placeholder_secret "${API_KEY}"; then
  fail_with_error "HY4_API_KEY looks like a literal documentation placeholder; generate a real key or leave HY4_API_KEY unset to auto-generate one."
fi
if is_placeholder_secret "${HF_TOKEN:-}"; then
  echo "[bootstrap-hy4] WARNING: HF_TOKEN looks like a documentation placeholder; ignoring it to avoid Hugging Face HTTP 401." >&2
  unset HF_TOKEN
fi

# Resolve effective API key (auto-generate & reuse persisted key unless ALLOW_UNAUTHENTICATED=1).
if [[ -z "${API_KEY}" ]]; then
  if [[ "${ALLOW_UNAUTHENTICATED}" == "1" ]]; then
    API_KEY=""
  elif [[ -f "${API_KEY_FILE}" ]]; then
    API_KEY="$(tr -d '\r\n' < "${API_KEY_FILE}" 2>/dev/null || true)"
  fi
fi

# Refuse unauthenticated non-loopback bind unless explicitly overridden.
if ! is_loopback_host_list "${HOST}" && [[ -z "${API_KEY}" && "${ALLOW_UNAUTHENTICATED}" != "1" ]]; then
  fail_with_error "Refusing to bind non-loopback HY4_HOST=${HOST} without HY4_API_KEY (set HY4_API_KEY, or HY4_ALLOW_UNAUTHENTICATED=1 on an isolated private network)."
fi

# Auto-generate a strong random key if still empty and authentication is enabled.
AUTO_GENERATED_KEY=0
if [[ -z "${API_KEY}" && "${ALLOW_UNAUTHENTICATED}" != "1" ]]; then
  API_KEY="sk-$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
  AUTO_GENERATED_KEY=1
fi

HEALTH_CODE=""
HEALTH_BODY=""
probe_health() {
  local out
  if [[ "${PROBE_HOST}" == *.sock ]]; then
    out="$(curl -sS --noproxy '*' --unix-socket "${PROBE_HOST}" -m 3 -w $'\n%{http_code}' "http://localhost/health" 2>/dev/null || printf '\n000')"
  else
    out="$(curl -sS -g --noproxy '*' -m 3 -w $'\n%{http_code}' "http://${PROBE_HOST}:${PORT}/health" 2>/dev/null || printf '\n000')"
  fi
  HEALTH_CODE="${out##*$'\n'}"
  HEALTH_BODY="${out%$'\n'*}"
}

check_ready() {
  probe_health
  [[ "${HEALTH_CODE}" == "200" && "${HEALTH_BODY}" == *'"status"'*'"ok"'* ]]
}

managed_server_pid() {
  if [[ -f "${SERVER_PID_FILE}" ]]; then
    local pid comm
    pid="$(tr -dc '0-9' < "${SERVER_PID_FILE}" || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      comm="$(cat "/proc/${pid}/comm" 2>/dev/null || ps -p "${pid}" -o comm= 2>/dev/null || true)"
      comm="${comm#"${comm%%[![:space:]]*}"}"
      comm="${comm%"${comm##*[![:space:]]}"}"
      if [[ "${comm}" == "llama-server" ]]; then
        echo "${pid}"
        return 0
      fi
    fi
  fi
  return 1
}

stop_managed_server() {
  local pid=""
  pid="$(managed_server_pid || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${SERVER_PID_FILE}"
    return 0
  fi
  echo "[bootstrap-hy4] Stopping managed llama-server (pid=${pid})..."
  kill "${pid}" 2>/dev/null || true
  local deadline=$(( SECONDS + 15 ))
  while (( SECONDS < deadline )); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[bootstrap-hy4] Sending SIGKILL to llama-server (pid=${pid})..."
    kill -9 "${pid}" 2>/dev/null || true
    local kdeadline=$(( SECONDS + 10 ))
    while (( SECONDS < kdeadline )); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
  fi
  rm -f "${SERVER_PID_FILE}"
}

wait_for_ready_and_check_ram() {
  local spid="$1"
  local t0="$2"
  local start_sec="${SECONDS}"
  local deadline=$(( start_sec + START_TIMEOUT_S ))
  local last_report=0
  local ready=0

  while (( SECONDS < deadline )); do
    if check_ready; then
      ready=1
      break
    fi
    if ! kill -0 "${spid}" 2>/dev/null; then
      echo "[bootstrap-hy4] ERROR: llama-server (pid=${spid}) exited early:" >&2
      tail -n 40 "${SERVER_LOG}" >&2 || true
      fail_with_error "llama-server exited during startup (see ${SERVER_LOG})"
    fi
    local elapsed=$(( SECONDS - start_sec ))
    if (( elapsed - last_report >= 10 )); then
      last_report="${elapsed}"
      local vram_now
      vram_now="$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -n 1 || echo "n/a")"
      write_state "start_server" 4 4 "running" \
        "Loading weights (pid=${spid}, elapsed=${elapsed}s/${START_TIMEOUT_S}s, health=${HEALTH_CODE}, vram=${vram_now})"
    fi
    sleep 1
  done

  if [[ "${ready}" -ne 1 ]]; then
    echo "[bootstrap-hy4] ERROR: timed out waiting for llama-server /health (${START_TIMEOUT_S}s)" >&2
    tail -n 40 "${SERVER_LOG}" >&2 || true
    fail_with_error "Timed out waiting for llama-server /health after ${START_TIMEOUT_S}s; inspect ${SERVER_LOG} and re-run with HY4_FORCE_RESTART=1 if wedged"
  fi

  local t1
  t1="$(date +%s.%N)"
  python3 -c 'import sys; print(f"[bootstrap-hy4] llama-server ready in {float(sys.argv[2]) - float(sys.argv[1]):.2f}s (pid={sys.argv[3]})")' "${t0}" "${t1}" "${spid}"

  # Host-RAM budget check. With --load-mode none the CPU tensors (62.7 GiB Engram +
  # 3.7 GiB blk.42) live in anonymous memory and Colab has no swap, so they can never
  # be paged out; the GGUF page cache is pure reclaimable cache. The only remaining
  # growth is the bounded --cache-ram pool, so it must fit in what is still available.
  local mem_avail_kb need_kb
  mem_avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  need_kb=$(( CACHE_RAM_MIB * 1024 + HOST_RESERVE_GIB * 1024 * 1024 ))
  echo "[bootstrap-hy4] Host RAM: available=$(( mem_avail_kb / 1048576 )) GiB, need cache-ram $(( CACHE_RAM_MIB / 1024 )) GiB + reserve ${HOST_RESERVE_GIB} GiB = $(( need_kb / 1048576 )) GiB"
  if (( mem_avail_kb < need_kb )); then
    stop_managed_server
    fail_with_error "Host RAM budget exceeded (available=$(( mem_avail_kb / 1048576 )) GiB < required=$(( need_kb / 1048576 )) GiB); lower HY4_CACHE_RAM_MIB / HY4_PARALLEL / HY4_PER_SLOT_CTX."
  fi

  local dfly_line
  dfly_line="$(grep -m 1 'dfly_print_i:' "${SERVER_LOG}" 2>/dev/null || true)"
  if [[ -n "${dfly_line}" ]]; then
    echo "[bootstrap-hy4] ${dfly_line}"
  fi

  write_state "ready" 4 4 "ready" "Hy4-Flash ready (${ENDPOINT_URL})"
  echo "[bootstrap-hy4] READY: ${ENDPOINT_URL} (bind=${HOST}:${PORT}, model=${ALIAS}, dfly_k=${DFLY_N_MAX})"
  if [[ -f "${API_KEY_FILE}" ]]; then
    echo "[bootstrap-hy4] API key file (0600): ${API_KEY_FILE}"
  fi
  nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader || true
  free -h || true
  du -sh "${MODEL_DIR}" "${BIN_DIR}" || true
}

WANTED_RUNTIME_SIG="$(compute_runtime_sig "${API_KEY}")"

# Idempotency fast-paths:
# 1) Already healthy -> exit 0 immediately (warn if flags/key differ from active server).
# 2) Managed llama-server already running and still loading weights (HTTP 503) -> attach and wait.
if [[ "${FORCE_RESTART}" == "0" ]]; then
  if check_ready; then
    EXISTING_PID="$(managed_server_pid || pgrep -x -n llama-server 2>/dev/null || echo unknown)"
    echo "[bootstrap-hy4] llama-server is already healthy on ${ENDPOINT_URL} (bind=${HOST}:${PORT}, pid=${EXISTING_PID})."
    if [[ -f "${CONFIG_SIG_FILE}" ]]; then
      RUNNING_SIG="$(tr -d '\r\n' < "${CONFIG_SIG_FILE}" 2>/dev/null || true)"
      if [[ -n "${RUNNING_SIG}" && "${RUNNING_SIG}" != "${WANTED_RUNTIME_SIG}" ]]; then
        echo "[bootstrap-hy4] WARNING: Requested API key or server flags differ from the currently running llama-server; existing instance was left running. Set HY4_FORCE_RESTART=1 to apply changes." >&2
      fi
    fi
    echo "[bootstrap-hy4] Set HY4_FORCE_RESTART=1 to force restart with updated flags."
    write_state "ready" 4 4 "ready" "llama-server already healthy"
    exit 0
  fi
  LOADING_PID="$(managed_server_pid || true)"
  if [[ -n "${LOADING_PID}" ]]; then
    echo "[bootstrap-hy4] Detected managed llama-server (pid=${LOADING_PID}) still starting/loading (health=${HEALTH_CODE}); waiting for readiness instead of killing it."
    echo "[bootstrap-hy4] Set HY4_FORCE_RESTART=1 if you want to terminate and relaunch."
    write_state "start_server" 4 4 "running" "Attached to loading llama-server (pid=${LOADING_PID}, health=${HEALTH_CODE})"
    wait_for_ready_and_check_ram "${LOADING_PID}" "$(date +%s.%N)"
    exit 0
  fi
fi

# ---------------------------------------------------------------------------
# Step 1/4: Hard-verify Blackwell G4 hardware, host RAM, disk space & toolchain
# ---------------------------------------------------------------------------
write_state "env_check" 1 4 "running" "Verifying GPU, host RAM, disk budget, and build toolchain"
export PATH="/usr/local/cuda/bin:${PATH}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail_with_error "nvidia-smi not found; this recipe requires an NVIDIA Blackwell 96 GiB GPU (Colab G4)."
fi

GPU_INFO="$(nvidia-smi --id=0 --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -n 1)"
echo "[bootstrap-hy4] GPU 0: ${GPU_INFO}"

if [[ "${SKIP_HW_CHECK}" != "1" ]]; then
  HW_CHECK_ERR="$(python3 - "${GPU_INFO}" "${CUDA_ARCH}" "${MIN_VRAM_MIB}" "${MIN_RAM_GIB}" "${MODEL_DIR}" "${ENABLE_VISION}" "${MAIN_GGUF}" "${DFLY_GGUF}" "${MMPROJ_GGUF}" <<'PYEOF'
import os, shutil, sys
from pathlib import Path

gpu_info, cuda_arch, min_vram_s, min_ram_s, model_dir, enable_vision = sys.argv[1:7]
target_files = [Path(p) for p in sys.argv[7:10]]
if enable_vision == "0":
    target_files = target_files[:2]

parts = [x.strip() for x in gpu_info.split(",")]
if len(parts) < 3:
    print(f"Unexpected nvidia-smi output: {gpu_info!r}")
    sys.exit(0)
gpu_name, vram_mib_s, cc_s = parts[0], parts[1], parts[2]
vram_mib = int(float(vram_mib_s))
cc = tuple(int(x) for x in cc_s.split(".")[:2])
min_vram = int(min_vram_s)
min_ram_gib = int(min_ram_s)

if vram_mib < min_vram:
    print(f"GPU VRAM too small: {gpu_name} has {vram_mib} MiB (< {min_vram} MiB required for 96 GiB G4 recipe).")
    sys.exit(0)
if cuda_arch == "120a":
    if cc != (12, 0):
        print(f"GPU compute capability {cc_s} ({gpu_name}) does not match HY4_CUDA_ARCH=120a (exact sm_120 / compute_cap 12.0 required; adjust HY4_CUDA_ARCH for other Blackwell variants).")
        sys.exit(0)
elif cuda_arch.startswith("120") and cc < (12, 0):
    print(f"GPU compute capability {cc_s} ({gpu_name}) is incompatible with HY4_CUDA_ARCH={cuda_arch} (Blackwell >= 12.0 required).")
    sys.exit(0)

mem_total_kb = 0
with open("/proc/meminfo", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("MemTotal:"):
            mem_total_kb = int(line.split()[1])
            break
mem_total_gib = mem_total_kb / (1024.0 * 1024.0)
if mem_total_gib < min_ram_gib:
    print(f"Host RAM too small: {mem_total_gib:.1f} GiB (< {min_ram_gib} GiB required for --load-mode none Engram + cache-ram).")
    sys.exit(0)

# Disk budget: 151.35 (main) + 0.84 (dfly) + 0.97 (mmproj if enabled) + 0.84 GiB headroom
required_gib = 154.0 if enable_vision != "0" else 153.0
required_bytes = int(required_gib * (1024 ** 3))
allocated = 0
for p in target_files:
    if p.exists():
        st = p.stat()
        allocated += int(st.st_blocks) * 512 if hasattr(st, "st_blocks") else int(st.st_size)
free_bytes = shutil.disk_usage(model_dir).free
net_need = max(0, required_bytes - allocated)
if free_bytes < net_need:
    print(
        f"Insufficient disk space in {model_dir}: need {net_need/(1024**3):.2f} GiB more "
        f"(budget={required_gib:.1f} GiB, already_allocated={allocated/(1024**3):.2f} GiB), "
        f"only {free_bytes/(1024**3):.2f} GiB free."
    )
    sys.exit(0)
PYEOF
)"
  if [[ -n "${HW_CHECK_ERR}" ]]; then
    fail_with_error "${HW_CHECK_ERR}"
  fi
fi

if ! command -v cmake >/dev/null 2>&1 \
  || ! command -v ninja >/dev/null 2>&1 \
  || ! command -v git >/dev/null 2>&1 \
  || ! command -v curl >/dev/null 2>&1 \
  || ! python3 -c "import requests" >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq cmake ninja-build git curl python3-requests
fi

NVCC_BIN="${HY4_NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
if [[ ! -x "${NVCC_BIN}" ]]; then
  fail_with_error "nvcc not found at ${NVCC_BIN}"
fi

NVCC_VER="$("${NVCC_BIN}" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n 1)"
echo "[bootstrap-hy4] nvcc: ${NVCC_BIN} (release ${NVCC_VER:-unknown})"
if [[ "${SKIP_HW_CHECK}" != "1" && "${CUDA_ARCH}" == 120* ]]; then
  NVCC_ERR="$(python3 - "${NVCC_VER}" <<'PYEOF'
import sys
ver = (sys.argv[1] or "").strip()
try:
    parts = tuple(int(x) for x in ver.split(".")[:2])
except ValueError:
    parts = (0, 0)
if parts < (12, 8):
    print(f"nvcc release {ver!r} is too old for sm_120a (CUDA >= 12.8 required).")
PYEOF
)"
  if [[ -n "${NVCC_ERR}" ]]; then
    fail_with_error "${NVCC_ERR}"
  fi
fi

# ---------------------------------------------------------------------------
# Step 2/4: Build stripped self-contained llama-server binary (in /tmp or /dev/shm)
# ---------------------------------------------------------------------------
CMAKE_FLAGS_SIG="Release_CUDA_ON_SHARED_OFF_OPENSSL_OFF_UI_OFF_TESTS_OFF_v1"
BUILD_SIG="$(sha256sum "${PATCH_DIR}/01-angelslim-hy4-flash-spark.patch.gz" "${PATCH_DIR}/02-hy4-flash-g4.patch" | awk '{print $1}' | sha256sum | awk '{print $1}')_${LLAMA_COMMIT}_${CUDA_ARCH}_nvcc${NVCC_VER:-unk}_${CMAKE_FLAGS_SIG}"
EXISTING_SIG="$(cat "${BIN_SIG_FILE}" 2>/dev/null || true)"

if [[ ! -x "${BIN_DIR}/llama-server" || "${EXISTING_SIG}" != "${BUILD_SIG}" ]]; then
  write_state "build" 2 4 "running" "Building pure C++/CUDA sm_${CUDA_ARCH} llama-server (nvcc ${NVCC_VER:-unknown})"
  TMP_AVAIL_GB="$(df -BG --output=avail /tmp | tail -n 1 | tr -dc '0-9')"
  SHM_AVAIL_GB="$(df -BG --output=avail /dev/shm 2>/dev/null | tail -n 1 | tr -dc '0-9' || echo 0)"
  if [[ "${TMP_AVAIL_GB:-0}" -ge 10 ]]; then
    BUILD_TMP_DIR="/tmp/hy4-build.$$"
  elif [[ "${SHM_AVAIL_GB:-0}" -ge 12 ]]; then
    BUILD_TMP_DIR="/dev/shm/hy4-build.$$"
    echo "[bootstrap-hy4] /tmp free=${TMP_AVAIL_GB} GiB < 10 GiB; building in ${BUILD_TMP_DIR} (/dev/shm free=${SHM_AVAIL_GB} GiB)."
  else
    fail_with_error "Insufficient build space (/tmp=${TMP_AVAIL_GB} GiB, /dev/shm=${SHM_AVAIL_GB} GiB; need >= 10 GiB)"
  fi

  rm -rf "${BUILD_TMP_DIR}"
  mkdir -p "${BUILD_TMP_DIR}"
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "${BUILD_TMP_DIR}/src"
  git -C "${BUILD_TMP_DIR}/src" checkout "${LLAMA_COMMIT}"
  gzip -dc "${PATCH_DIR}/01-angelslim-hy4-flash-spark.patch.gz" | git -C "${BUILD_TMP_DIR}/src" apply
  git -C "${BUILD_TMP_DIR}/src" apply "${PATCH_DIR}/02-hy4-flash-g4.patch"

  cmake -S "${BUILD_TMP_DIR}/src" -B "${BUILD_TMP_DIR}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${NVCC_BIN}" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF

  cmake --build "${BUILD_TMP_DIR}/build" --target llama-server -j "${BUILD_JOBS}"
  strip -o "${BIN_DIR}/llama-server.new" "${BUILD_TMP_DIR}/build/bin/llama-server"
  chmod +x "${BIN_DIR}/llama-server.new"
  mv -f "${BIN_DIR}/llama-server.new" "${BIN_DIR}/llama-server"
  echo "${BUILD_SIG}" > "${BIN_SIG_FILE}"
  rm -rf "${BUILD_TMP_DIR}"
  BUILD_TMP_DIR=""
  echo "[bootstrap-hy4] Installed stripped binary: $(ls -lh "${BIN_DIR}/llama-server")"
fi

# ---------------------------------------------------------------------------
# Step 3/4: Download clean SPARK-4Bit (Q8_0 trunk) + external DFly Q8_0 draft (+ mmproj)
# ---------------------------------------------------------------------------
write_state "download" 3 4 "running" "Downloading Hy4-flash-SPARK-4Bit.gguf (151.35 GiB), DFly (0.84 GiB), mmproj (0.97 GiB)"
DL_EXTRA_ARGS=(--offline-ok)
if [[ "${VERIFY_SHA}" == "1" ]]; then
  DL_EXTRA_ARGS+=(--verify-sha)
fi
python3 -u "${SCRIPT_DIR}/download_gguf.py" --url "${DFLY_URL}" --out "${DFLY_GGUF}" --workers 8 "${DL_EXTRA_ARGS[@]}"
if [[ "${ENABLE_VISION}" != "0" ]]; then
  python3 -u "${SCRIPT_DIR}/download_gguf.py" --url "${MMPROJ_URL}" --out "${MMPROJ_GGUF}" --workers 8 "${DL_EXTRA_ARGS[@]}"
fi
python3 -u "${SCRIPT_DIR}/download_gguf.py" --url "${MAIN_URL}" --out "${MAIN_GGUF}" --workers "${DL_WORKERS}" "${DL_EXTRA_ARGS[@]}"

# ---------------------------------------------------------------------------
# Step 4/4: Launch llama-server (tiered 8x256K KV + F16 SWA + Q8_0 Full + DFly)
# ---------------------------------------------------------------------------
write_state "start_server" 4 4 "running" "Launching llama-server and loading weights (expect 11-24 min with --load-mode none)"
stop_managed_server

# Preserve previous server log for post-mortem debugging.
if [[ -s "${SERVER_LOG}" ]]; then
  mv -f "${SERVER_LOG}" "${SERVER_LOG}.prev" || true
fi

# Persist active API key (0600) and runtime config signature so subsequent cells/shells can read it.
if [[ -n "${API_KEY}" ]]; then
  write_secret_file "${API_KEY_FILE}" "${API_KEY}"
  if [[ "${AUTO_GENERATED_KEY}" == "1" ]]; then
    echo "[bootstrap-hy4] Auto-generated strong API key and saved to ${API_KEY_FILE} (mode 0600)."
  else
    echo "[bootstrap-hy4] Persisted API key to ${API_KEY_FILE} (mode 0600)."
  fi
else
  rm -f "${API_KEY_FILE}"
fi
write_secret_file "${CONFIG_SIG_FILE}" "${WANTED_RUNTIME_SIG}"

VISION_ARGS=()
if [[ "${ENABLE_VISION}" != "0" ]]; then
  VISION_ARGS=(--mmproj "${MMPROJ_GGUF}")
  if [[ "${MMPROJ_OFFLOAD}" == "0" ]]; then
    VISION_ARGS+=(--no-mmproj-offload)
  else
    VISION_ARGS+=(--mmproj-offload)
  fi
fi

# Note: FA_576_WHOLE_TILES, CPY_F32_VEC, SPARK_MMQ_TILE=-1, SPARK_MMQ_COMPACT_GRID=1,
# SPARK_MMVQ_2W=1 are enabled by default on Blackwell in 02-hy4-flash-g4.patch.
# Do NOT combine GGML_CUDA_REUSE_Q8=1 with GGML_CUDA_GRAPH_OPT=1.
export CUDA_VISIBLE_DEVICES=0
export GGML_CUDA_MOE_CONCURRENT=1
export GGML_CUDA_GRAPH_OPT=1

# Pass API key via LLAMA_API_KEY env var rather than CLI argv (avoids leaking secret in ps).
if [[ -n "${API_KEY}" ]]; then
  export LLAMA_API_KEY="${API_KEY}"
else
  unset LLAMA_API_KEY || true
fi

T0="$(date +%s.%N)"
nohup "${BIN_DIR}/llama-server" \
  -m "${MAIN_GGUF}" \
  -md "${DFLY_GGUF}" \
  ${VISION_ARGS[@]+"${VISION_ARGS[@]}"} \
  -ngl 99 -ngld 99 \
  --fit off \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --load-mode none --lazy-mode off \
  -ot '^token_embd\.weight$=CUDA0,^per_layer_token_embd\.weight$=CPU,^blk\.42\..*=CPU' \
  -kvu -np "${N_PARALLEL}" --kv-unified-per-slot "${PER_SLOT_CTX}" \
  -b 4096 -ub 4096 \
  --spec-type draft-dfly --spec-draft-n-max "${DFLY_N_MAX}" \
  --cache-ram "${CACHE_RAM_MIB}" --no-cache-idle-slots \
  --jinja --metrics \
  --host "${HOST}" --port "${PORT}" \
  --alias "${ALIAS}" \
  > "${SERVER_LOG}" 2>&1 < /dev/null 9>&- &
SPID=$!
unset LLAMA_API_KEY || true
echo "${SPID}" > "${SERVER_PID_FILE}"

wait_for_ready_and_check_ram "${SPID}" "${T0}"
