#!/usr/bin/env python3
"""Zero-dependency end-to-end verification script for a running Hy4-Flash llama-server.

Checks (each isolated so one failure does not abort the remaining checks):
  1. GET  /health & /v1/models (model id + multimodal capability)
  2. Auth guard (when --api-key is given, unauthenticated /v1/models must return HTTP 401)
  3. POST /v1/chat/completions (no_think code generation + DFly speculative draft stats)
  4. POST /v1/chat/completions (reasoning_effort="low" thinking + answer check)
  5. POST /v1/responses        (OpenAI Responses API with reasoning.effort="low" + output text check)
  6. POST /v1/messages         (Anthropic Messages API using ONLY x-api-key header + text output check)
  7. Vision multimodal read + follow-up KV cache reuse (auto-skipped if model lacks multimodal capability)
  8. Optional concurrent throughput probe (--concurrency N, reports aggregate & per-slot average)

Usage:
  python3 scripts/verify.py --base-url http://127.0.0.1:8000 --api-key sk-xxx
  python3 scripts/verify.py --base-url http://127.0.0.1:8000 --skip-vision
  python3 scripts/verify.py --base-url http://127.0.0.1:8000 --concurrency 4
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from typing import Callable


def make_split_color_png(w: int = 320, h: int = 160) -> bytes:
    """Create an unambiguous left-RED / right-BLUE PNG using only Python stdlib."""
    raw = bytearray()
    half = w // 2
    red_row = b"\xe6\x18\x18" * half
    blue_row = b"\x18\x48\xe6" * (w - half)
    row_bytes = b"\x00" + red_row + blue_row
    for _ in range(h):
        raw.extend(row_bytes)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


class Client:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # Always bypass local HTTP proxies when talking to loopback / tunneled endpoints.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request_raw(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        extra_headers: dict[str, str] | None = None,
        send_bearer: bool = True,
    ) -> tuple[int, str, float]:
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {}
        if send_bearer and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        t0 = time.perf_counter()
        try:
            with self.opener.open(req, timeout=self.timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace"), time.perf_counter() - t0
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return exc.code, body, time.perf_counter() - t0

    def request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        extra_headers: dict[str, str] | None = None,
        send_bearer: bool = True,
    ) -> tuple[int, dict, float]:
        code, raw, dt = self.request_raw(
            method, path, payload=payload, extra_headers=extra_headers, send_bearer=send_bearer
        )
        if code < 200 or code >= 300:
            raise RuntimeError(f"{method} {path} -> HTTP {code}: {raw[:300]}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{method} {path} -> invalid JSON ({exc}): {raw[:200]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{method} {path} -> expected JSON object, got {type(parsed).__name__}")
        return code, parsed, dt


def extract_responses_text(body: dict) -> str:
    parts: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                txt = c.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
    return "".join(parts).strip()


def extract_anthropic_text(body: dict) -> str:
    parts: list[str] = []
    for block in body.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            txt = block.get("text")
            if isinstance(txt, str):
                parts.append(txt)
    return "".join(parts).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a running Hy4-Flash llama-server deployment.")
    ap.add_argument("--base-url", default=os.environ.get("HY4_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=os.environ.get("HY4_API_KEY", ""))
    ap.add_argument(
        "--api-key-file",
        default=os.environ.get("HY4_API_KEY_FILE", "/content/.hy4-g4/api_key"),
        help="Fallback file to read API key from when --api-key and HY4_API_KEY are empty",
    )
    ap.add_argument("--model", default=os.environ.get("HY4_MODEL_ALIAS", "hy4-flash"))
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--skip-vision", action="store_true", help="Skip multimodal image test (use if HY4_ENABLE_VISION=0)")
    ap.add_argument("--concurrency", type=int, default=0, help="Optional N-way concurrent throughput check (e.g. 4 or 8)")
    args = ap.parse_args()

    if not args.api_key and args.api_key_file and os.path.isfile(args.api_key_file):
        try:
            with open(args.api_key_file, "r", encoding="utf-8") as f:
                loaded = f.read().strip()
            if loaded:
                args.api_key = loaded
                print(f"[INFO] Loaded API key from {args.api_key_file}", flush=True)
        except OSError as exc:
            print(f"[WARN] Could not read {args.api_key_file}: {exc}", file=sys.stderr, flush=True)

    cli = Client(args.base_url, args.api_key, args.timeout)
    failures: list[str] = []
    passed = 0
    skipped = 0

    def record(name: str, ok: bool, detail: str) -> None:
        nonlocal passed
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}: {detail}", flush=True)
        if ok:
            passed += 1
        else:
            failures.append(f"{name}: {detail}")

    def record_skip(name: str, reason: str) -> None:
        nonlocal skipped
        skipped += 1
        print(f"[SKIP] {name}: {reason}", flush=True)

    def run_check(name: str, fn: Callable[[], tuple[bool, str]]) -> None:
        try:
            ok, detail = fn()
            record(name, ok, detail)
        except Exception as exc:
            record(name, False, f"{type(exc).__name__}: {exc}")

    # 1. /health
    def chk_health() -> tuple[bool, str]:
        st, body, dt = cli.request_json("GET", "/health", send_bearer=False)
        return st == 200 and body.get("status") == "ok", f"HTTP {st} body={body} ({dt*1000:.0f}ms)"

    run_check("health", chk_health)

    # 2. Auth guard (when --api-key is provided, unauthenticated /v1/models must be rejected with 401)
    if args.api_key:
        def chk_auth_guard() -> tuple[bool, str]:
            code, raw, _ = cli.request_raw("GET", "/v1/models", send_bearer=False)
            return code == 401, f"unauthenticated GET /v1/models -> HTTP {code} ({raw[:120]!r})"

        run_check("auth_guard", chk_auth_guard)
    else:
        record_skip("auth_guard", "no --api-key provided")

    # 3. /v1/models (also discovers capabilities)
    model_caps: list[str] | None = None

    def chk_models() -> tuple[bool, str]:
        nonlocal model_caps
        st, body, _ = cli.request_json("GET", "/v1/models")
        ids = [m.get("id") for m in body.get("data", []) if isinstance(m, dict)]
        caps: list[str] = []
        for m in body.get("models", []):
            if isinstance(m, dict) and m.get("name") == args.model:
                raw_caps = m.get("capabilities")
                if isinstance(raw_caps, list):
                    caps = [str(c) for c in raw_caps]
        model_caps = caps
        return st == 200 and args.model in ids, f"ids={ids} capabilities={caps}"

    run_check("models", chk_models)

    # 4. OpenAI Chat Completions (no_think) + DFly speculative stats
    def chk_chat_dfly() -> tuple[bool, str]:
        st, body, dt = cli.request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a concise Python function implementing binary search on a sorted list. Code only.",
                    }
                ],
                "max_tokens": 160,
                "temperature": 0.0,
                "seed": 42,
                "cache_prompt": False,
                "chat_template_kwargs": {"reasoning_effort": "no_think"},
            },
        )
        tim = body.get("timings") or {}
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        dn = tim.get("draft_n", 0)
        da = tim.get("draft_n_accepted", 0)
        tps = tim.get("predicted_per_second", 0.0)
        return (
            st == 200 and "def " in content and dn > 0,
            f"tg={tps:.1f} t/s, predicted={tim.get('predicted_n')}, draft={da}/{dn} "
            f"({(100.0*da/dn if dn else 0):.1f}%), wall={dt:.2f}s",
        )

    run_check("chat_no_think_dfly", chk_chat_dfly)

    # 5. Reasoning mode (low)
    def chk_reasoning_low() -> tuple[bool, str]:
        st, body, _ = cli.request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
                "max_tokens": 256,
                "temperature": 0.0,
                "reasoning_effort": "low",
            },
        )
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        has_rc = bool(msg.get("reasoning_content"))
        ans = (msg.get("content") or "").strip()
        tps = (body.get("timings") or {}).get("predicted_per_second", 0.0)
        return (
            st == 200 and has_rc and "391" in ans,
            f"has_reasoning={has_rc}, ans={ans[:60]!r}, tg={tps:.1f} t/s",
        )

    run_check("chat_reasoning_low", chk_reasoning_low)

    # 6. OpenAI /v1/responses (explicit reasoning.effort="low" + verify actual output text)
    def chk_responses() -> tuple[bool, str]:
        st, body, dt = cli.request_json(
            "POST",
            "/v1/responses",
            {
                "model": args.model,
                "input": "Reply with PONG only.",
                "reasoning": {"effort": "low"},
                "max_output_tokens": 128,
                "temperature": 0.0,
            },
        )
        out_text = extract_responses_text(body)
        return (
            st == 200 and body.get("status") == "completed" and "pong" in out_text.lower(),
            f"HTTP {st} status={body.get('status')!r} text={out_text[:60]!r} ({dt:.2f}s)",
        )

    run_check("openai_responses", chk_responses)

    # 7. Anthropic /v1/messages (send ONLY x-api-key, NO Authorization Bearer, and verify text output)
    def chk_anthropic() -> tuple[bool, str]:
        anth_headers = {"anthropic-version": "2023-06-01"}
        if args.api_key:
            anth_headers["x-api-key"] = args.api_key
        st, body, dt = cli.request_json(
            "POST",
            "/v1/messages",
            {
                "model": args.model,
                "max_tokens": 256,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": "Reply with PONG only."}],
            },
            extra_headers=anth_headers,
            send_bearer=False,
        )
        out_text = extract_anthropic_text(body)
        return (
            st == 200 and body.get("type") == "message" and "pong" in out_text.lower(),
            f"HTTP {st} type={body.get('type')!r} text={out_text[:60]!r} (auth=x-api-key-only, {dt:.2f}s)",
        )

    run_check("anthropic_messages", chk_anthropic)

    # 8. Vision multimodal + follow-up KV cache reuse
    vision_disabled_by_server = (model_caps is not None and len(model_caps) > 0 and "multimodal" not in model_caps)
    if args.skip_vision:
        record_skip("vision_first_turn", "--skip-vision specified")
        record_skip("vision_followup_kv_reuse", "--skip-vision specified")
    elif vision_disabled_by_server:
        record_skip("vision_first_turn", f"server capabilities={model_caps} (no multimodal)")
        record_skip("vision_followup_kv_reuse", f"server capabilities={model_caps} (no multimodal)")
    else:
        b64 = base64.b64encode(make_split_color_png()).decode("ascii")
        v_msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This image is split vertically into a left half and a right half. Name the two colors (left first, then right).",
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ]
        first_answer = ""

        def chk_vision_1() -> tuple[bool, str]:
            nonlocal first_answer
            st, d1, dt1 = cli.request_json(
                "POST",
                "/v1/chat/completions",
                {
                    "model": args.model,
                    "messages": v_msgs,
                    "max_tokens": 48,
                    "temperature": 0.0,
                    "cache_prompt": True,
                    "chat_template_kwargs": {"reasoning_effort": "no_think"},
                },
            )
            msg = (d1.get("choices") or [{}])[0].get("message") or {}
            a1 = (msg.get("content") or "").strip()
            first_answer = a1
            low1 = a1.lower()
            t1 = d1.get("timings") or {}
            return (
                st == 200 and "red" in low1 and "blue" in low1,
                f"wall={dt1:.2f}s, prompt_n={t1.get('prompt_n')}, tg={t1.get('predicted_per_second', 0):.1f} t/s, ans={a1[:80]!r}",
            )

        run_check("vision_first_turn", chk_vision_1)

        def chk_vision_2() -> tuple[bool, str]:
            v_msgs2 = v_msgs + [
                {"role": "assistant", "content": first_answer or "Left: Red, Right: Blue."},
                {"role": "user", "content": "Which color is on the right side? One word only."},
            ]
            st, d2, dt2 = cli.request_json(
                "POST",
                "/v1/chat/completions",
                {
                    "model": args.model,
                    "messages": v_msgs2,
                    "max_tokens": 16,
                    "temperature": 0.0,
                    "cache_prompt": True,
                    "chat_template_kwargs": {"reasoning_effort": "no_think"},
                },
            )
            msg = (d2.get("choices") or [{}])[0].get("message") or {}
            a2 = (msg.get("content") or "").strip()
            t2 = d2.get("timings") or {}
            return (
                st == 200 and "blue" in a2.lower() and t2.get("cache_n", 0) > 50,
                f"wall={dt2:.2f}s, cache_n={t2.get('cache_n')}, prompt_n={t2.get('prompt_n')}, ans={a2!r}",
            )

        run_check("vision_followup_kv_reuse", chk_vision_2)

    # 9. Optional N-way concurrent check
    if args.concurrency > 0:
        n = args.concurrency

        def chk_concurrency() -> tuple[bool, str]:
            def one(i: int):
                return cli.request_json(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": args.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Task #{i}: Write a clean Python class implementing a thread-safe bounded queue with comments.",
                            }
                        ],
                        "max_tokens": 160,
                        "temperature": 0.0,
                        "seed": 42 + i,
                        "cache_prompt": False,
                        "chat_template_kwargs": {"reasoning_effort": "no_think"},
                    },
                )

            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
                results = list(ex.map(one, range(n)))
            wall = max(time.perf_counter() - t0, 1e-6)
            tot_tok = sum((b.get("timings") or {}).get("predicted_n", 0) for _, b, _ in results)
            slot_tps = [
                float((b.get("timings") or {}).get("predicted_per_second", 0.0))
                for _, b, _ in results
            ]
            all_ok = all(s == 200 for s, _, _ in results)
            agg = tot_tok / wall
            slot_avg = sum(slot_tps) / len(slot_tps) if slot_tps else 0.0
            slot_min = min(slot_tps) if slot_tps else 0.0
            slot_max = max(slot_tps) if slot_tps else 0.0
            return (
                all_ok and tot_tok > 0,
                f"agg={agg:.1f} t/s ({tot_tok} tok in {wall:.2f}s), "
                f"per_slot_avg={slot_avg:.1f} t/s (range {slot_min:.1f}-{slot_max:.1f} t/s)",
            )

        run_check(f"concurrent_{n}x", chk_concurrency)

    if failures:
        print(f"\nSUMMARY: {passed} passed, {len(failures)} failed, {skipped} skipped", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"\nALL CHECKS PASSED ({passed} passed, {skipped} skipped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
