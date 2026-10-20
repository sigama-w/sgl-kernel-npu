"""Single-operator benchmark for the HiCache AIV kv_exchange kernel.

Compares host memory backed by hugepages vs regular 4KB pages (the DMA
efficiency question behind the "1G hugepage configuration optimization"
requirement), by running the same sparse page transfer with
SGLANG_HICACHE_HUGEPAGE_MODE pinned to different values:

    python bench_kv_exchange.py                       # auto(1G->2M->4K) vs off
    python bench_kv_exchange.py --modes 1g,off        # strict 1G vs 4KB
    python bench_kv_exchange.py --modes 2m,off

Pool layouts mirror the MLA page_first_kv_split host pool:
  host   [page, layer, page_size, 1, width]  (page-first)
  device [layer, page, page_size, 1, width]  (layer-first)
"""

import argparse
import os
import time

import torch

import sgl_kernel_npu  # noqa: F401
from sgl_kernel_npu.hicache_offload import (
    HOST_LAYOUT_PAGE_FIRST,
    KV_EXCHANGE_DIRECTION_D2H,
    KV_EXCHANGE_DIRECTION_H2D,
    build_kv_exchange_component,
    build_kv_exchange_meta,
    free_hugepage_all,
    kv_exchange,
)

BLOCK_DIM = 64  # Ascend950: cores 62/63 participate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-pages", type=int, default=1024, help="L2 pool size in pages")
    p.add_argument("--pages", type=int, default=256, help="pages transferred per iteration (sparse hit set)")
    p.add_argument("--page-size", type=int, default=128, help="tokens per page")
    p.add_argument("--width", type=int, default=576, help="elements per token per component (k_lora+rope)")
    p.add_argument("--layers", type=int, default=2, help="KV layers covered by the transfer")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--modes", type=str, default="auto,off", help="comma list: auto / 1g / 2m / off")
    p.add_argument("--directions", type=str, default="h2d,d2h")
    p.add_argument(
        "--start-dims",
        type=str,
        default="",
        help=(
            "comma list of KV_EXCHANGE_START_DIM overrides (active cores = "
            "block_dim - start_dim), e.g. '62,60,56,48,32,0'; empty = default 62"
        ),
    )
    return p.parse_args()


def make_host_buffer(shape, dtype):
    from sgl_kernel_npu.hicache_offload import create_hugepage_tensor

    tensor, host_ptr, dev_ptr = create_hugepage_tensor(shape, dtype)
    return tensor, host_ptr, dev_ptr


def bench_direction(direction, indices_cpu, args, dtype, layers, width, start_dim=None):
    """Returns (seconds_per_iter, bytes_moved_per_iter)."""
    device = "npu"
    pool_pages, pages, page_size = args.pool_pages, args.pages, args.page_size

    # Both pools cover the full pool; the indices select which pages move
    # (same as the sglang host pool).
    host_shape = (pool_pages, layers, page_size, 1, width)
    device_shape = (layers, pool_pages, page_size, 1, width)

    host_k, host_k_ptr, host_k_dev = make_host_buffer(host_shape, dtype)
    host_v, host_v_ptr, host_v_dev = make_host_buffer(host_shape, dtype)
    device_k = torch.zeros(device_shape, dtype=dtype, device=device)
    device_v = torch.zeros(device_shape, dtype=dtype, device=device)

    device_indices = indices_cpu.to(device=device, dtype=torch.int64)
    host_indices = device_indices.clone()

    components = []
    for buf, dev in ((host_k, host_k_dev), (host_v, host_v_dev)):
        components.append(
            build_kv_exchange_component(
                device_base=device_k.data_ptr() if buf is host_k else device_v.data_ptr(),
                host_base=dev,
                device_layer_pitch=(device_k if buf is host_k else device_v).stride(0) * dtype.itemsize,
                device_page_pitch=(device_k if buf is host_k else device_v).stride(1) * dtype.itemsize,
                host_page_pitch=buf.stride(0) * dtype.itemsize,
                host_layer_pitch=buf.stride(1) * dtype.itemsize,
                width=buf.stride(2) * dtype.itemsize * page_size,
                layer_lo=0,
                layer_hi=layers,
            )
        )

    vals = build_kv_exchange_meta(
        num_pages=pages,
        page_size=page_size,
        direction=direction,
        device_indices_ptr=device_indices.data_ptr(),
        host_indices_ptr=host_indices.data_ptr(),
        host_layout_mode=HOST_LAYOUT_PAGE_FIRST,
        components=components,
        start_dim=start_dim,
    )
    meta = torch.tensor(vals, dtype=torch.int64, device=device)

    def run_once():
        kv_exchange(meta, block_dim=BLOCK_DIM)

    for _ in range(args.warmup):
        run_once()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        run_once()
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    bytes_moved = (
        pages * page_size * layers * 2 * width * dtype.itemsize  # k + v, bf16 bytes
    )
    free_hugepage_all()
    return elapsed / args.iters, bytes_moved


def run_mode(mode, direction, indices_cpu, args, dtype, start_dim=None):
    os.environ["SGLANG_HICACHE_HUGEPAGE_MODE"] = mode
    sec, nbytes = bench_direction(direction, indices_cpu, args, dtype, args.layers, args.width, start_dim)
    gbps = nbytes / sec / 1e9
    active = BLOCK_DIM - start_dim if start_dim is not None else BLOCK_DIM - 62
    cores = f" cores={active:<2}" if start_dim is not None else ""
    print(
        f"  mode={mode:<5} dir={'H2D' if direction == 1 else 'D2H'}{cores}  "
        f"moved={nbytes / 1e6:.1f}MB  {sec * 1e3:.3f} ms/iter  {gbps:.2f} GB/s"
    )
    return gbps


def main():
    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.npu.set_device(0)

    if args.pages > args.pool_pages:
        raise SystemExit("--pages must be <= --pool-pages")

    # Sparse ascending hit set: simulates radix-cache hits (sorted like the
    # sglang host pool transfer path).
    gen = torch.Generator().manual_seed(0)
    page_ids = torch.randperm(args.pool_pages, generator=gen)[: args.pages].sort().values
    indices_cpu = (page_ids * args.page_size).repeat_interleave(args.page_size).to(torch.int64)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    dirs = [KV_EXCHANGE_DIRECTION_H2D if d == "h2d" else KV_EXCHANGE_DIRECTION_D2H
            for d in args.directions.split(",") if d.strip()]
    if args.start_dims.strip():
        start_dims = [int(s) for s in args.start_dims.split(",") if s.strip()]
    else:
        start_dims = [None]

    print(
        f"kv_exchange bench: pool={args.pool_pages} pages, transfer={args.pages} pages, "
        f"page_size={args.page_size}, width={args.width}, layers={args.layers}, dtype={args.dtype}, "
        f"iters={args.iters}, block_dim={BLOCK_DIM}"
    )
    results = {}
    for sd in start_dims:
        for mode in modes:
            for direction in dirs:
                results[(sd, mode, direction)] = run_mode(mode, direction, indices_cpu, args, dtype, sd)

    print("\n==== hugepage A/B summary (GB/s) ====")
    for sd in start_dims:
        tag = f"start_dim={sd} (cores={BLOCK_DIM - sd})" if sd is not None else "default (cores=2)"
        print(f"  ---- {tag} ----")
        for direction in dirs:
            cells = "  ".join(f"{m}: {results[(sd, m, direction)]:8.2f}" for m in modes)
            print(f"  {'H2D' if direction == 1 else 'D2H'}: {cells}")
            base = results.get((sd, "off", direction))
            if base:
                for m in modes:
                    if m != "off":
                        print(f"    {m} vs off: {results[(sd, m, direction)] / base:.2f}x")


if __name__ == "__main__":
    main()
