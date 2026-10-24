#!/usr/bin/env python3
"""Bandwidth benchmark for the aclrt + AIV kv_exchange ops (MLA geometry).

Measures effective H2D/D2H bandwidth of the sgl-kernel-npu AIV kernel with
the production-like MLA pool layout: page-first host (registered) and
layer-first device, k/v split pools (kv_lora_rank vs qk_rope_head_dim).

Usage:
    python3 benchmark/bench_kv_exchange_aclrt.py                     # default MLA geometry
    python3 benchmark/bench_kv_exchange_aclrt.py --layers 8 --pages 256
    SGLANG_KV_EXCHANGE_BLOCK_DIM=32 python3 benchmark/bench_kv_exchange_aclrt.py
"""
import argparse
import time

import torch

from sgl_kernel_npu.kvcacheio import (
    register_host_memory,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_per_layer_direct_pf_lf,
    unregister_host_memory,
)

# MLA geometry mirrors sglang pool_host/mla.py page_first_kv_split
PAGE_SIZE = 128
K_WIDTH = 656   # kv_lora_rank
V_WIDTH = 64    # qk_rope_head_dim
ESZ = torch.bfloat16.itemsize


def _page_aligned_indices(pages: int, device: str):
    """Page-granular bijection: slot p maps dev_pages[p] <-> host_pages[p]."""
    offset = torch.arange(PAGE_SIZE, dtype=torch.int64).repeat(pages)
    dev_pages = torch.randperm(pages, dtype=torch.int64)
    host_pages = torch.randperm(pages, dtype=torch.int64)
    dev_idx = ((dev_pages * PAGE_SIZE).repeat_interleave(PAGE_SIZE) + offset).to(device)
    host_idx = ((host_pages * PAGE_SIZE).repeat_interleave(PAGE_SIZE) + offset).to(device)
    return dev_idx, host_idx


def _pooled_bytes(layers: int, pages: int, width: int) -> int:
    return layers * pages * PAGE_SIZE * width * ESZ


def bench_h2d_per_layer(layers: int, pages: int, iters: int) -> None:
    torch.npu.set_device(0)
    host_k = torch.zeros((pages, layers, PAGE_SIZE, 1, K_WIDTH), dtype=torch.bfloat16).pin_memory()
    host_v = torch.zeros((pages, layers, PAGE_SIZE, 1, V_WIDTH), dtype=torch.bfloat16).pin_memory()
    dev_k = torch.zeros((layers, pages, PAGE_SIZE, 1, K_WIDTH), dtype=torch.bfloat16, device="npu")
    dev_v = torch.zeros((layers, pages, PAGE_SIZE, 1, V_WIDTH), dtype=torch.bfloat16, device="npu")
    dev_idx, host_idx = _page_aligned_indices(pages, "npu")

    mk = register_host_memory(host_k)
    mv = register_host_memory(host_v)
    try:
        for _ in range(3):  # warmup
            for layer in range(layers):
                transfer_kv_per_layer_direct_pf_lf(
                    [host_k, host_v], [dev_k, dev_v], host_idx, dev_idx, layer, PAGE_SIZE,
                    host_mappings=[mk, mv])
        torch.npu.synchronize()

        bytes_moved = layers * _pooled_bytes(1, pages, K_WIDTH)
        bytes_moved += layers * _pooled_bytes(1, pages, V_WIDTH)
        start = time.perf_counter()
        for _ in range(iters):
            for layer in range(layers):
                transfer_kv_per_layer_direct_pf_lf(
                    [host_k, host_v], [dev_k, dev_v], host_idx, dev_idx, layer, PAGE_SIZE,
                    host_mappings=[mk, mv])
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start

        gbps = bytes_moved * iters / elapsed / 1e9
        print(f"H2D per-layer : {gbps:8.2f} GB/s  "
              f"({layers} layers x {pages} pages, {bytes_moved * iters / 1e9:.1f} GB in {elapsed * 1e3:.1f} ms)")
    finally:
        unregister_host_memory(mk)
        unregister_host_memory(mv)


def bench_d2h_all_layer(layers: int, pages: int, iters: int) -> None:
    torch.npu.set_device(0)
    host_k = torch.zeros((pages, layers, PAGE_SIZE, 1, K_WIDTH), dtype=torch.bfloat16).pin_memory()
    host_v = torch.zeros((pages, layers, PAGE_SIZE, 1, V_WIDTH), dtype=torch.bfloat16).pin_memory()
    dev_k = torch.randn((layers, pages, PAGE_SIZE, 1, K_WIDTH), device="npu").to(torch.bfloat16)
    dev_v = torch.randn((layers, pages, PAGE_SIZE, 1, V_WIDTH), device="npu").to(torch.bfloat16)
    dev_idx, host_idx = _page_aligned_indices(pages, "npu")

    mk = register_host_memory(host_k)
    mv = register_host_memory(host_v)
    try:
        for _ in range(3):  # warmup
            transfer_kv_all_layer_direct_lf_pf(
                [dev_k, dev_v], [host_k, host_v], dev_idx, host_idx, PAGE_SIZE,
                host_mappings=[mk, mv])
        torch.npu.synchronize()

        bytes_moved = _pooled_bytes(layers, pages, K_WIDTH) + _pooled_bytes(layers, pages, V_WIDTH)
        start = time.perf_counter()
        for _ in range(iters):
            transfer_kv_all_layer_direct_lf_pf(
                [dev_k, dev_v], [host_k, host_v], dev_idx, host_idx, PAGE_SIZE,
                host_mappings=[mk, mv])
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start

        gbps = bytes_moved * iters / elapsed / 1e9
        print(f"D2H all-layer : {gbps:8.2f} GB/s  "
              f"({layers} layers x {pages} pages, {bytes_moved * iters / 1e9:.1f} GB in {elapsed * 1e3:.1f} ms)")
    finally:
        unregister_host_memory(mk)
        unregister_host_memory(mv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=16, help="device pool layers (default 16)")
    parser.add_argument("--pages", type=int, default=128, help="pages per transfer (default 128)")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations (default 20)")
    args = parser.parse_args()

    import os

    total_mib = (args.pages * PAGE_SIZE * (K_WIDTH + V_WIDTH) * ESZ) / 1024 / 1024
    print(f"MLA geometry: page_size={PAGE_SIZE} k={K_WIDTH} v={V_WIDTH} | "
          f"layers={args.layers} pages={args.pages} ({total_mib:.1f} MiB per layer-slab) iters={args.iters}")
    print(f"block_dim env SGLANG_KV_EXCHANGE_BLOCK_DIM = "
          f"{os.environ.get('SGLANG_KV_EXCHANGE_BLOCK_DIM', 'default (arch: A5=64/A3=48)')}")
    bench_h2d_per_layer(args.layers, args.pages, args.iters)
    bench_d2h_all_layer(args.layers, args.pages, args.iters)


if __name__ == "__main__":
    main()
