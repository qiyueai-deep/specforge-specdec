from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .decoding import autoregressive_generate, speculative_generate
from .training import load_artifacts


def make_handler(target, draft, tokenizer, *, max_concurrency: int = 2,
                 queue_timeout_seconds: float = 30.0):
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    if queue_timeout_seconds < 0:
        raise ValueError("queue_timeout_seconds must be >= 0")
    slots = threading.BoundedSemaphore(max_concurrency)
    metrics_lock = threading.Lock()
    metrics = {"active_requests": 0, "total_requests": 0,
               "rejected_requests": 0, "peak_active_requests": 0}

    def snapshot_metrics() -> dict:
        with metrics_lock:
            return dict(metrics)

    def enter_request() -> bool:
        acquired = slots.acquire(timeout=queue_timeout_seconds)
        with metrics_lock:
            if not acquired:
                metrics["rejected_requests"] += 1
                return False
            metrics["active_requests"] += 1
            metrics["total_requests"] += 1
            metrics["peak_active_requests"] = max(
                metrics["peak_active_requests"], metrics["active_requests"]
            )
        return True

    def leave_request() -> None:
        with metrics_lock:
            metrics["active_requests"] -= 1
        slots.release()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                service = snapshot_metrics()
                self._send(200, {
                    "status": "ok", "project": "SpecForge", "device": "cpu",
                    "target_parameters": target.parameter_count, "draft_parameters": draft.parameter_count,
                    "max_concurrency": max_concurrency,
                    "queue_timeout_seconds": queue_timeout_seconds,
                    **service,
                })
                return
            self._send(404, {"error": "Use GET /health or POST /generate"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/generate":
                self._send(404, {"error": "Use POST /generate"})
                return
            request_started = time.perf_counter()
            if not enter_request():
                body = {
                    "error": "server concurrency limit reached",
                    "max_concurrency": max_concurrency,
                    "queue_timeout_seconds": queue_timeout_seconds,
                }
                payload = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(429)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Retry-After", "1")
                self.end_headers()
                self.wfile.write(payload)
                return
            inference_started = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                prompt = str(payload.get("prompt", ""))
                if not prompt:
                    raise ValueError("prompt must not be empty")
                decoder = str(payload.get("decoder", "speculative"))
                mode = str(payload.get("mode", "greedy"))
                new_tokens = min(64, max(1, int(payload.get("max_new_tokens", 32))))
                prompt_ids = tokenizer.encode(prompt, add_bos=True)
                common = {
                    "mode": mode, "temperature": float(payload.get("temperature", 0.9)),
                    "top_k": int(payload.get("top_k", 20)), "seed": int(payload.get("seed", 7)),
                }
                if decoder == "autoregressive":
                    tokens, stats = autoregressive_generate(target, prompt_ids, new_tokens, **common)
                elif decoder == "speculative":
                    tokens, stats = speculative_generate(
                        target, draft, prompt_ids, new_tokens, int(payload.get("draft_k", 8)), **common
                    )
                else:
                    raise ValueError("decoder must be autoregressive or speculative")
                response_stats = stats.to_dict()
                response_stats["service_queue_seconds"] = max(0.0, inference_started - request_started)
                response_stats["service_wall_seconds"] = time.perf_counter() - request_started
                response_stats["service_active_requests"] = snapshot_metrics()["active_requests"]
                self._send(200, {"decoder": decoder, "text": tokenizer.decode(tokens), "stats": response_stats})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})
            except Exception as error:  # noqa: BLE001 - always answer instead of hanging the client
                self._send(500, {"error": f"internal generation error: {error}"})
            finally:
                leave_request()

        def log_message(self, fmt: str, *args) -> None:
            print(f"[SpecForge API] {self.address_string()} - {fmt % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SpecForge with the Python standard library")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrency", type=int, default=2,
                        help="Maximum simultaneous generation requests")
    parser.add_argument("--queue-timeout", type=float, default=30.0,
                        help="Seconds to wait for a free request slot; 0 rejects immediately")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="CPU operator threads per request; keep at 1 for request-level concurrency")
    args = parser.parse_args()
    if args.max_concurrency < 1 or args.queue_timeout < 0 or args.torch_threads < 1:
        parser.error("--max-concurrency and --torch-threads must be >= 1; --queue-timeout must be >= 0")
    target, draft, tokenizer, _ = load_artifacts(args.artifact_dir)
    torch.set_num_threads(args.torch_threads)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(target, draft, tokenizer, max_concurrency=args.max_concurrency,
                     queue_timeout_seconds=args.queue_timeout),
    )
    server.daemon_threads = True
    print(f"SpecForge API: http://{args.host}:{args.port}")
    print("GET /health | POST /generate")
    print(f"max_concurrency={args.max_concurrency}; queue_timeout={args.queue_timeout:.1f}s; "
          f"torch_threads={args.torch_threads}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nserver stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
