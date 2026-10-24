# Self-contained correctness test for the AICPU kv_exchange backend.
#
# Unlike the migrated AIV path (test_kv_exchange.py), this test does NOT need the
# hugepage host allocator. The copy runs inside the AICPU device kernel launched
# by torch.ops.npu.kv_exchange_aicpu.
#
# IMPORTANT: the AICPU stars subsystem cannot dereference host memory through a
# registration DVA (rtsHostRegister channel is AIV-only on current CANN); the
# working mechanism -- the one mf's aicpu mode uses -- is passing host VIRTUAL
# addresses (HVA) directly: the kernel reads host indices / host cache via HVA
# and device indices / device cache via device pointers. So meta carries HVAs
# for all host-side fields and no aclrt_host_register call is needed here.
#
# Prerequisites (one-time, on the NPU device):
#   1. Rebuild sgl-kernel-npu so sgl_kernel_npu.so contains kv_exchange_aicpu.
#   2. bash scripts/build_kv_exchange_aicpu_run.sh install
#      (or export SGL_KV_EXCHANGE_AICPU_JSON=/path/to/libcann_kv_exchange_kernel.json)
#
# Run (from sgl-kernel-npu root, PYTHONPATH must point at the freshly built package):
#   export PYTHONPATH="$PWD/python/sgl_kernel_npu:$PYTHONPATH"
#   pytest test_kv_exchange_aicpu.py -v
#
# meta layout (44 int64, must match kv_exchange_aicpu_kernel.cc):
#   [0]=num_components [1]=num_indices [2]=page_size [3]=direction(1=H2D,2=D2H)
#   [4]=device_indices ptr [5]=host_indices HVA [6]=host_layout_mode(0=pf,1=lf)
#   per component c at 8+9c: dev_base, host_base(HVA), dev_layer_pitch,
#   dev_page_pitch, host_page_pitch, host_layer_pitch, width, layer_lo, layer_hi

import pytest
import torch

try:
    import torch_npu  # noqa: F401

    HAS_NPU = True
except ImportError:
    HAS_NPU = False

try:
    import sgl_kernel_npu  # noqa: F401

    HAS_OPS = hasattr(torch.ops.npu, "kv_exchange_aicpu") and hasattr(torch.ops.npu, "aclrt_host_register")
except ImportError:
    HAS_OPS = False

DTYPE = torch.bfloat16
ELEM_SIZE = 2  # bytes per bf16 element (torch.dtype has no element_size attr)
ALIGN = 2 * 1024 * 1024  # register at a 2MB-aligned host address
H2D, D2H = 1, 2
PAGE_FIRST, LAYER_FIRST = 0, 1


def alloc_host_cache(shape):
    """Allocate a PINNED (page-locked) CPU buffer so its pages have a stable
    physical backing while the AICPU kernel dereferences the HVA, and return
    the bf16 tensor view plus the holder keeping it alive."""
    return torch.empty(shape, dtype=DTYPE, pin_memory=True), None


def build_meta(num_components, num_indices, page_size, direction, layout,
               device_idx_ptr, host_idx_ptr, comp_geoms):
    """comp_geoms: list of dicts with dev_base, host_base, dev_layer_pitch,
    dev_page_pitch, host_page_pitch, host_layer_pitch, width, layer_lo, layer_hi."""
    meta = [0] * 44
    meta[0] = num_components
    meta[1] = num_indices
    meta[2] = page_size
    meta[3] = direction
    meta[4] = device_idx_ptr
    meta[5] = host_idx_ptr
    meta[6] = layout
    for c, g in enumerate(comp_geoms):
        base = 8 + 9 * c
        meta[base + 0] = g["dev_base"]
        meta[base + 1] = g["host_base"]
        meta[base + 2] = g["dev_layer_pitch"]
        meta[base + 3] = g["dev_page_pitch"]
        meta[base + 4] = g["host_page_pitch"]
        meta[base + 5] = g["host_layer_pitch"]
        meta[base + 6] = g["width"]
        meta[base + 7] = g["layer_lo"]
        meta[base + 8] = g["layer_hi"]
    return torch.tensor(meta, dtype=torch.int64)


@pytest.mark.skipif(not (HAS_NPU and HAS_OPS), reason="torch_npu or kv_exchange_aicpu op not available")
@pytest.mark.parametrize("direction", [H2D, D2H])
@pytest.mark.parametrize("layout", [PAGE_FIRST, LAYER_FIRST])
@pytest.mark.parametrize("page_size", [1, 16])
def test_kv_exchange_aicpu(direction, layout, page_size):
    device = "npu"
    num_layers, head_dim = 6, 576
    num_tokens_dev, num_tokens_host, num_indices = 128, 96, 40

    # Geometry: one component per K/V tensor, device cache is layer-major
    # [layers, tokens, head_dim]; host cache layout depends on host_layout_mode.
    token_bytes = page_size * head_dim * ELEM_SIZE  # width per page row
    width = token_bytes
    dev_layer_pitch = num_tokens_dev * head_dim * ELEM_SIZE
    dev_page_pitch = head_dim * ELEM_SIZE
    if layout == PAGE_FIRST:
        host_page_pitch = num_layers * head_dim * ELEM_SIZE
        host_layer_pitch = head_dim * ELEM_SIZE
        host_shape = (num_tokens_host, num_layers, head_dim)
    else:
        host_page_pitch = head_dim * ELEM_SIZE
        host_layer_pitch = num_tokens_host * head_dim * ELEM_SIZE
        host_shape = (num_layers, num_tokens_host, head_dim)
    dev_shape = (num_layers, num_tokens_dev, head_dim)

    # Indices: distinct token ids. Device indices live on device (read via
    # device pointer); host indices live in pinned host memory (read via HVA).
    dev_tok_ids = torch.randperm(num_tokens_dev, device=device)[:num_indices].contiguous()
    host_tok_ids = torch.randperm(num_tokens_host).pin_memory().contiguous()

    results = []
    host_holders = []
    for _ in range(2):  # K and V
        dev_cache = torch.zeros(dev_shape, dtype=DTYPE, device=device)
        host_buf, holder = alloc_host_cache(host_shape)
        if direction == H2D:
            host_buf.copy_(torch.randint(1, 100, host_shape).to(DTYPE))
        else:
            dev_cache = torch.randn(dev_shape, dtype=DTYPE, device=device).to(DTYPE)
        hva = host_buf.data_ptr()
        host_holders.append((host_buf, holder, hva))
        results.append((dev_cache, host_buf))

    try:
        geoms = [
            {
                "dev_base": dev_cache.data_ptr(),
                "host_base": hva,
                "dev_layer_pitch": dev_layer_pitch,
                "dev_page_pitch": dev_page_pitch,
                "host_page_pitch": host_page_pitch,
                "host_layer_pitch": host_layer_pitch,
                "width": width,
                "layer_lo": 0,
                "layer_hi": num_layers,
            }
            for dev_cache, (_, _, hva) in zip([r[0] for r in results], host_holders)
        ]
        meta = build_meta(2, num_indices, page_size, direction, layout,
                          dev_tok_ids.data_ptr(), host_tok_ids.data_ptr(), geoms)

        # Keep every buffer alive across the async launch + sync.
        keep_alive = [meta, dev_tok_ids, host_tok_ids] + [b for b, _, _ in host_holders] + [r[0] for r in results]
        torch.ops.npu.kv_exchange_aicpu(meta)
        torch.npu.synchronize()

        dev_tok_cpu = dev_tok_ids.cpu()
        host_tok_cpu = host_tok_ids.cpu()
        for dev_cache, host_buf in results:
            dev_flat = dev_cache.reshape(num_layers, num_tokens_dev, head_dim)
            if direction == H2D:
                expected = host_buf.cpu().reshape(host_shape)
                for layer in range(num_layers):
                    got = dev_flat[layer, dev_tok_cpu.long()]
                    want = _gather_host(expected, layout, num_layers, host_tok_cpu, page_size, head_dim)[layer]
                    torch.testing.assert_close(got.cpu(), want, msg=f"H2D mismatch layer={layer}")
            else:
                got_host = host_buf.cpu().reshape(host_shape)
                for layer in range(num_layers):
                    got = _gather_host(got_host, layout, num_layers, host_tok_cpu, page_size, head_dim)[layer]
                    want = dev_flat[layer, dev_tok_cpu.long()].cpu()
                    torch.testing.assert_close(got, want, msg=f"D2H mismatch layer={layer}")
        print(f"precision OK (direction={direction}, layout={layout}, page_size={page_size})")
    finally:
        pass  # pinned buffers are freed by GC; no registration to undo


PROBE_MAGIC = 0x41435030  # "ACP0", must match kProbeMagicBase in the kernel

PROBE_STAGES = [
    (0, None, "S0 kernel running / param+meta readable"),
    (1, 9, "S1 read device_indices (device memory)"),
    (2, 10, "S2 read device_base byte (device data)"),
    (3, 11, "S3 read host_indices (host HVA)"),
    (4, 12, "S4 read host_base byte (host data)"),
    (5, None, "S5 all probes completed"),
]


@pytest.mark.skipif(not (HAS_NPU and HAS_OPS), reason="torch_npu or kv_exchange_aicpu op not available")
def test_aicpu_probe():
    """Diagnostic: which address spaces can the AICPU stars sandbox touch?

    Run alone on a fresh process (it may crash the device on host accesses):
        pytest test_kv_exchange_aicpu.py::test_aicpu_probe -v -s
    """
    device = "npu"
    status = torch.zeros(16, dtype=torch.int32, device=device)
    dev_idx = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)
    host_idx = torch.tensor([5, 6, 7, 8], dtype=torch.int64).pin_memory()
    dev_cache = torch.full((64,), 1.0, dtype=DTYPE, device=device)
    host_buf = torch.full((64,), 1.0, dtype=DTYPE).pin_memory()

    meta = [0] * 44
    meta[0] = 1  # num_components
    meta[1] = 4  # num_indices (echoed back at status[8])
    meta[2] = 1  # page_size
    meta[3] = 0  # direction 0 = PROBE
    meta[4] = dev_idx.data_ptr()
    meta[5] = host_idx.data_ptr()
    meta[6] = 0
    meta[7] = status.data_ptr()
    meta[8] = dev_cache.data_ptr()
    meta[9] = host_buf.data_ptr()
    meta_t = torch.tensor(meta, dtype=torch.int64)

    err = None
    try:
        torch.ops.npu.kv_exchange_aicpu(meta_t)
        torch.npu.synchronize()
    except Exception as e:  # noqa: BLE001
        err = e

    st = status.cpu().tolist()
    print(f"\nprobe raw status: {[hex(v) if v else 0 for v in st[:13]]}")
    for slot, comp_slot, name in PROBE_STAGES:
        ok = st[slot] == PROBE_MAGIC + slot and (comp_slot is None or st[comp_slot] != 0)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}")
    if err is not None:
        print(f"  launch/sync raised: {err}")
    if st[0] == 0:
        print("  ==> status[0] never written: the kernel never ran or never saw a valid param "
              "(args staging / rtsHostRegister layer is broken)")
    print("  ==> interpretation: the FAILed stage right after the last OK stage is the "
          "address class the AICPU sandbox cannot dereference")


def _gather_host(host_buf, layout, num_layers, tok_ids, page_size, head_dim):
    """Return per-layer [num_indices, head_dim] slices from a host cache tensor."""
    tok_ids = tok_ids.long()
    if layout == PAGE_FIRST:
        # [num_tokens, layers, head_dim] -> per layer, per token
        return [host_buf[tok_ids, layer, :] for layer in range(num_layers)]
    # LAYER_FIRST: [layers, num_tokens, head_dim]
    return [host_buf[layer, tok_ids, :] for layer in range(num_layers)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
