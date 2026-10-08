import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import get_arch, is_hip


@triton.jit
def _reduce_extreme_kernel(X, Y, BLOCK: tl.constexpr, USE_MAX: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + offsets)
    result = tl.max(values, axis=0) if USE_MAX else tl.min(values, axis=0)
    tl.store(Y, result)


@pytest.mark.parametrize(
    "use_max, sign, expected",
    [(True, -1.0, -1.0), (False, 1.0, 1.0)],
)
def test_amd_reduction_boundary_identity(use_max, sign, expected, device):
    if not is_hip():
        pytest.skip("AMDGPU reduction regression")

    warp_size = triton.runtime.driver.active.get_current_target().warp_size
    if warp_size not in (32, 64):
        pytest.skip(f"unsupported AMD wave size: {warp_size}")

    values = sign * torch.arange(1, warp_size + 1, device=device, dtype=torch.float32)
    result = torch.empty((1, ), device=device, dtype=torch.float32)
    _reduce_extreme_kernel[(1, )](values, result, warp_size, use_max, num_warps=1)
    assert result.item() == expected


@triton.jit
def _reduce_two_32_lane_groups(X, Y):
    rows = tl.arange(0, 2)
    columns = tl.arange(0, 32)
    offsets = rows[:, None] * 32 + columns[None, :]
    values = tl.load(X + offsets)
    sums = tl.sum(values, axis=1)
    tl.store(Y + rows, sums)


@triton.jit
def _reduce_interleaved_32_lanes(X, Y):
    rows = tl.arange(0, 32)
    columns = tl.arange(0, 2)
    offsets = rows[:, None] * 2 + columns[None, :]
    values = tl.load(X + offsets)
    sums = tl.sum(values, axis=0)
    tl.store(Y + columns, sums)


@pytest.mark.parametrize("layout_case", ["two_lane_groups", "interleave_2"])
def test_amd_wave64_partial_warp_sum(layout_case, device):
    if not is_hip():
        pytest.skip("AMDGPU reduction regression")

    target = triton.runtime.driver.active.get_current_target()
    if target.warp_size != 64:
        pytest.skip(f"requires wave64, got wave{target.warp_size}")

    if layout_case == "two_lane_groups":
        values = torch.cat((torch.ones((1, 32)), torch.full((1, 32), 2.0)), dim=0).to(device=device)
        kernel = _reduce_two_32_lane_groups
    else:
        values = torch.stack((torch.ones((32, )), torch.full((32, ), 2.0)), dim=1).to(device=device)
        kernel = _reduce_interleaved_32_lanes

    result = torch.empty((2, ), device=device, dtype=torch.float32)
    kernel[(1, )](values, result, num_warps=1)
    torch.testing.assert_close(result, torch.tensor([32.0, 64.0], device=device), rtol=0, atol=0)


@triton.jit
def _atomic_add_masked_pairs(
    Output,
    Values,
    N: tl.constexpr,
    BASE: tl.constexpr,
    EVEN: tl.constexpr,
    ODD: tl.constexpr,
):
    offsets = tl.max_contiguous(tl.arange(0, N), 2)
    values = tl.load(Values + offsets)
    mask = tl.where((offsets & 1) == 0, EVEN, ODD)
    tl.atomic_add(Output + BASE + offsets, values, mask=mask, sem="relaxed")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask_pair", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("base_offset", [0, 1])
def test_amd_half_atomic_mask_pairs(dtype, mask_pair, base_offset, device):
    if not is_hip():
        pytest.skip("AMDGPU atomic-mask regression")

    arch = get_arch()
    if dtype is torch.bfloat16 and not arch.startswith(("gfx90a", "gfx94")):
        pytest.skip(f"BF16 atomic add is only covered on CDNA targets: {arch}")

    n = 256
    even, odd = mask_pair
    mask = torch.tensor(mask_pair, device=device, dtype=torch.bool).repeat(n // 2)
    values = torch.arange(1, n + 1, device=device, dtype=torch.float32).to(dtype)
    output = torch.zeros((n + base_offset, ), device=device, dtype=dtype)

    _atomic_add_masked_pairs[(1, )](output, values, n, base_offset, even, odd, num_warps=4)

    expected = torch.zeros_like(output)
    expected[base_offset:base_offset + n] = values * mask.to(dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
