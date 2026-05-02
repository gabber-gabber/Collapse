#!/usr/bin/env python3
"""GPU RPC server for Occlum benchmark.

This server runs outside TEE and exposes a local TCP RPC endpoint.
The TEE side sends obfuscated matrices and receives GEMM results.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


LEN_FMT = "!Q"
LEN_BYTES = struct.calcsize(LEN_FMT)


def recvall(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    header = recvall(sock, LEN_BYTES)
    total = struct.unpack(LEN_FMT, header)[0]
    payload = recvall(sock, total)
    return pickle.loads(payload)


def send_msg(sock: socket.socket, obj: dict[str, Any]) -> None:
    payload = pickle.dumps(obj, protocol=5)
    sock.sendall(struct.pack(LEN_FMT, len(payload)))
    sock.sendall(payload)


def np_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    if dtype_name == "float64":
        return np.float64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def torch_dtype(dtype_name: str):
    if torch is None:
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def matmul_numpy(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    c = a @ b
    t_ms = (time.perf_counter() - t0) * 1000.0
    return c, t_ms


def matmul_torch(a: np.ndarray, b: np.ndarray, device: str, dtype_name: str) -> tuple[np.ndarray, float]:
    if torch is None:
        raise RuntimeError("PyTorch is not installed on server")

    tdtype = torch_dtype(dtype_name)
    a_t = torch.from_numpy(a).to(device=device, dtype=tdtype)
    b_t = torch.from_numpy(b).to(device=device, dtype=tdtype)

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        c_t = a_t @ b_t
        end.record()
        torch.cuda.synchronize()
        gpu_ms = float(start.elapsed_time(end))
    else:
        t0 = time.perf_counter()
        c_t = a_t @ b_t
        gpu_ms = (time.perf_counter() - t0) * 1000.0

    c = c_t.detach().cpu().numpy()
    return c, gpu_ms


def run_server(host: str, port: int, device: str) -> None:
    print(f"[server] listen on {host}:{port}, device={device}", flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(8)

        while True:
            conn, addr = srv.accept()
            print(f"[server] client connected: {addr}", flush=True)
            with conn:
                try:
                    while True:
                        req = recv_msg(conn)
                        cmd = req.get("cmd", "matmul")

                        if cmd == "ping":
                            send_msg(conn, {"ok": True, "msg": "pong"})
                            continue

                        if cmd == "shutdown":
                            send_msg(conn, {"ok": True, "msg": "bye"})
                            print("[server] shutdown requested", flush=True)
                            return

                        if cmd != "matmul":
                            send_msg(conn, {"ok": False, "error": f"unknown cmd={cmd}"})
                            continue

                        dtype_name = req["dtype"]
                        a_shape = tuple(req["a_shape"])
                        b_shape = tuple(req["b_shape"])
                        dt = np_dtype(dtype_name)

                        a = np.frombuffer(req["a_bytes"], dtype=dt).reshape(a_shape)
                        b = np.frombuffer(req["b_bytes"], dtype=dt).reshape(b_shape)

                        t0 = time.perf_counter()
                        if torch is None or device == "numpy":
                            c, gpu_ms = matmul_numpy(a, b)
                        else:
                            c, gpu_ms = matmul_torch(a, b, device=device, dtype_name=dtype_name)
                        total_ms = (time.perf_counter() - t0) * 1000.0

                        resp = {
                            "ok": True,
                            "gpu_ms": gpu_ms,
                            "server_total_ms": total_ms,
                            "dtype": dtype_name,
                            "c_shape": c.shape,
                            "c_bytes": c.tobytes(order="C"),
                        }
                        send_msg(conn, resp)
                except ConnectionError:
                    print(f"[server] client disconnected: {addr}", flush=True)
                except Exception as exc:  # pragma: no cover
                    send_msg(conn, {"ok": False, "error": repr(exc)})
                    print(f"[server] error: {exc!r}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU RPC server for Occlum benchmark")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cuda:0 / cpu / numpy",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_server(host=args.host, port=args.port, device=args.device)


if __name__ == "__main__":
    main()
