"""Measure the inference memory footprint and speed.

    python benchmark.py --weights weights/best.pt

Reports: parameter count, weight file size, MACs/FLOPs, activation memory,
process RSS before/after loading + running the model on CPU, MPS/CUDA allocated
memory, and latency / FPS at batch size 1.
"""
import argparse
import gc
import json
import os
import platform
import resource
import sys
import time

import psutil
import torch
import torch.nn as nn

from src.dataset import IMG_H, IMG_W
from src.model import TinyLaneSeg, count_params

MB = 1024 ** 2


def count_macs_and_activations(model, x):
    """Count conv MACs and the size of every intermediate output with forward hooks."""
    stats = {"macs": 0, "act_bytes": 0, "peak_single": 0}

    def hook(m, inp, out):
        nbytes = out.numel() * out.element_size()
        stats["act_bytes"] += nbytes
        stats["peak_single"] = max(stats["peak_single"], nbytes)
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            stats["macs"] += out.numel() * k

    hs = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return stats


def rss():
    return psutil.Process().memory_info().rss / MB


def time_it(model, x, sync, n=100, warmup=10):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        sync()
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    args = ap.parse_args()
    torch.set_num_threads(4)

    r = {"input": [1, 3, IMG_H, IMG_W], "torch": torch.__version__,
         "machine": f"{platform.system()} {platform.machine()} ({platform.processor() or 'cpu'})"}
    gc.collect()
    r["rss_baseline_mb"] = rss()

    model = TinyLaneSeg().eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    r["params"] = count_params(model)
    r["param_mb_fp32"] = r["params"] * 4 / MB
    r["weight_file_mb"] = os.path.getsize(args.weights) / MB
    x = torch.randn(1, 3, IMG_H, IMG_W)
    r["rss_after_load_mb"] = rss()

    s = count_macs_and_activations(model, x)
    r["gmacs"] = s["macs"] / 1e9
    r["gflops"] = 2 * s["macs"] / 1e9
    r["activations_total_mb"] = s["act_bytes"] / MB
    r["largest_activation_mb"] = s["peak_single"] / MB

    r["cpu_latency_ms"] = time_it(model, x, lambda: None)
    r["cpu_fps"] = 1000 / r["cpu_latency_ms"]
    r["rss_after_cpu_inference_mb"] = rss()
    r["rss_inference_overhead_mb"] = r["rss_after_cpu_inference_mb"] - r["rss_baseline_mb"]
    # ru_maxrss is bytes on macOS, KiB on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r["peak_rss_mb"] = peak / MB if sys.platform == "darwin" else peak / 1024

    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        torch.mps.empty_cache()
        m = model.to(dev)
        xd = x.to(dev)
        base = torch.mps.current_allocated_memory()
        r["mps_latency_ms"] = time_it(m, xd, torch.mps.synchronize)
        r["mps_fps"] = 1000 / r["mps_latency_ms"]
        r["mps_allocated_mb"] = (torch.mps.current_allocated_memory()) / MB
        r["mps_driver_allocated_mb"] = torch.mps.driver_allocated_memory() / MB
        r["mps_model_mb"] = base / MB
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
        m = model.to(dev)
        xd = x.to(dev)
        torch.cuda.reset_peak_memory_stats()
        r["cuda_latency_ms"] = time_it(m, xd, torch.cuda.synchronize)
        r["cuda_fps"] = 1000 / r["cuda_latency_ms"]
        r["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / MB

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(r, f, indent=1)
    for k, v in r.items():
        print(f"{k:<28}{v:.3f}" if isinstance(v, float) else f"{k:<28}{v}")


if __name__ == "__main__":
    main()
