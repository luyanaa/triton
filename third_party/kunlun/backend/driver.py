"""Triton driver for the Kunlun K200 (KL1 / XPU1).

Triton's backend discovery imports this module on every ``import triton``, so
nothing here may touch the vendor runtime at import time: on a host without a
K200 card, dlopen'ing libxpurt can wedge the machine.  All of that is lazy,
inside :mod:`triton.backends.kunlun.jitc`.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import struct
import time

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

XPU1_ARCH = 1
WARP_SIZE = 8

# Nothing is staged through XPU1 shared memory: the reference emitter lowers
# every tensor to a local-memory array, so `metadata.shared` is always 0.  The
# value below only has to be an honest upper bound on LM.
LM_BYTES = 48 * 1024

# Device images are addressed by an integer key: Triton compares the launch
# `function` value on every launch, so the key must hash in O(1) rather than
# over the image bytes.  The buffers own the code and must outlive every
# launch, and identical images (same kernel recompiled after a cache eviction)
# share one buffer.
_IMAGES = {}  # key -> (address, length)
_IMAGE_BUFFERS = {}  # key -> ctypes buffer that keeps `address` valid
_IMAGE_KEYS = {}  # sha256(image) -> key
_NEXT_IMAGE = 0

# xpu_launch_argument_set() copies the whole parameter block into a 512-byte
# slot and rejects a larger one with 0xFFFFFD3D, so the launcher checks the
# size itself and fails with a message that names the cause.
MAX_PARAM_BYTES = 512


def _torch_kunlun():
    """``torch.kunlun`` when torch_kunlun is importable, else None."""
    try:
        import torch
        if not hasattr(torch, "kunlun"):
            import torch_kunlun  # noqa: F401
        return getattr(torch, "kunlun", None)
    except Exception:
        return None


def _sync():
    lib = _torch_kunlun()
    if lib is not None:
        lib.synchronize()


def _native_stream(handle: int) -> int:
    """Fallback for torch_kunlun builds without ``current_stream_native``.

    Reads the first field of ``StreamHandle{ XPUStream stream; ... }``.  The
    layout is that of the C++ struct, so prefer the native getter when the
    extension provides it.
    """
    if handle == 0:
        return 0
    return int(ctypes.c_void_p.from_address(handle).value or 0)


def _pointer(arg) -> int:
    if arg is None:
        return 0
    if isinstance(arg, int):
        return arg
    data_ptr = getattr(arg, "data_ptr", None)
    if data_ptr is None:
        raise TypeError(f"cannot convert {type(arg).__name__} to a device pointer")
    return int(data_ptr())


def _arg_kind(ty: str) -> str:
    if ty.startswith("*"):
        return "ptr"
    if ty == "fp64":
        return "f64"
    if ty == "fp32":
        return "f32"
    if ty == "fp16":
        return "f16"
    if ty in ("i64", "u64"):
        return "i64"
    # i1/i8/i16/i32/u8/u16/u32 all occupy a padded 4-byte slot.
    return "i32"


class KunlunUtils:

    def load_binary(self, name, kernel, shared, device):
        # libxpujitc wants the compiled image as bytes at launch time, and the
        # Triton caches keep it on disk rather than in memory, so park a copy
        # here under an integer key that is unique per distinct image.
        global _NEXT_IMAGE
        digest = hashlib.sha256(kernel).digest()
        key = _IMAGE_KEYS.get(digest)
        if key is None:
            buf = ctypes.create_string_buffer(kernel, len(kernel))
            key = _NEXT_IMAGE
            _NEXT_IMAGE += 1
            _IMAGES[key] = (ctypes.addressof(buf), len(kernel))
            _IMAGE_BUFFERS[key] = buf
            _IMAGE_KEYS[digest] = key
        # The image is already compiled and loaded by the JIT: no separate
        # module handle, and XPU-C gives no register/spill report.
        return (_IMAGE_BUFFERS[key], key, 0, 0)

    def get_device_properties(self, device):
        return {"max_shared_mem": LM_BYTES}


class KunlunLauncher:

    def __init__(self, src, metadata):
        from triton.backends.kunlun import jitc

        self._jitc = jitc
        self.kinds = [_arg_kind(ty) for _, ty in src.signature.items()]
        self.ncluster = int(metadata.cluster_dims[0])
        # One Triton warp is one XPU core, so num_warps is the number of cores
        # the launch may use.  Distributed mode requires all 16 (checked in
        # KunlunOptions); serial mode runs on fewer.
        self.ncore = int(metadata.num_warps)
        self.kernel_hash = int(metadata.kernel_hash)
        self.mangled = metadata.mangled
        self.use_cdnn = bool(metadata.use_cdnn)

    def __call__(self, gridX, gridY, gridZ, stream, function, packed_metadata, launch_metadata,
                 enter_hook, exit_hook, *args):
        if enter_hook is not None:
            enter_hook(launch_metadata)

        image = _IMAGES.get(function)
        if image is None:
            raise RuntimeError(f"no Kunlun device image registered for launch key {function}")
        if len(args) != len(self.kinds):
            raise ValueError(f"kernel expects {len(self.kinds)} arguments, got {len(args)}")
        address, length = image

        grid = (int(gridX), int(gridY), int(gridZ))
        params = bytearray()
        for kind, arg in zip(self.kinds, args):
            if kind == "ptr":
                ptr = _pointer(arg)
                if ptr >> 32:
                    # K200 device addresses are 32 bits wide, so a host pointer
                    # (or an allocation past 4 GB) would be silently truncated
                    # into an address the kernel can still dereference.
                    raise RuntimeError(f"pointer 0x{ptr:x} does not fit the 32-bit K200 "
                                       "device address space; is this a host pointer?")
                params += struct.pack("<II", ptr, 0)
            elif kind == "i64":
                params += struct.pack("<Q", int(arg) & 0xFFFFFFFFFFFFFFFF)
            elif kind == "f64":
                params += struct.pack("<d", float(arg))
            elif kind == "f32":
                params += struct.pack("<f", float(arg))
            elif kind == "f16":
                params += struct.pack("<e", float(arg)) + b"\0\0"
            else:
                params += struct.pack("<I", int(arg) & 0xFFFFFFFF)
        # The emitter derives the persistent-loop bounds from these.
        params += struct.pack("<4i", grid[0] * grid[1] * grid[2], *grid)
        if len(params) > MAX_PARAM_BYTES:
            raise RuntimeError(f"kernel parameters take {len(params)} bytes, but the Kunlun "
                               f"runtime accepts at most {MAX_PARAM_BYTES}")

        jitc = self._jitc
        jitc.launch(address, length, self.kernel_hash, self.ncluster, self.ncore,
                    int(stream or 0), bytes(params), self.use_cdnn, self.mangled)

        if exit_hook is not None:
            exit_hook(launch_metadata)


class _TimingEvent:
    # XPURT has no elapsed-time query, so do_bench-style timings fall back to
    # draining the stream and reading the wall clock.
    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing
        self.t = 0.0

    def record(self):
        _sync()
        self.t = time.perf_counter()

    def elapsed_time(self, end):
        return (end.t - self.t) * 1e3


class _DeviceInterface:
    Event = _TimingEvent

    @staticmethod
    def synchronize():
        _sync()


def _bench(fn, warmup=25.0, rep=100.0, grad_to_none=None, quantiles=None, return_mode="mean",
           **kwargs):
    """Wall-clock replacement for triton.testing.do_bench."""
    fn()
    _sync()
    start = time.perf_counter()
    fn()
    _sync()
    estimate = max(time.perf_counter() - start, 1e-4)

    for _ in range(max(1, int(warmup / 1e3 / estimate))):
        fn()
    times = []
    for _ in range(max(1, int(rep / 1e3 / estimate))):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        start = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - start) * 1e3)
    times.sort()

    if quantiles:
        return [times[min(len(times) - 1, int(q * (len(times) - 1)))] for q in quantiles]
    if return_mode == "min":
        return times[0]
    if return_mode == "max":
        return times[-1]
    return sum(times) / len(times)


class KunlunDriver(DriverBase):

    def __init__(self):
        self.utils = KunlunUtils()
        self.launcher_cls = KunlunLauncher

    @classmethod
    def is_active(cls):
        # Never probe the runtime here: opening /dev/xpu* is safe, dlopen'ing
        # the vendor libraries on a cardless host is not.
        return bool(glob.glob("/dev/xpu[0-9]*")) or os.environ.get("KUNLUN_DRIVER_ACTIVE") == "1"

    @classmethod
    def get_current_target(cls):
        return GPUTarget("kunlun", XPU1_ARCH, WARP_SIZE)

    def get_benchmarker(self):
        return _bench

    def get_device_interface(self):
        return _DeviceInterface

    def get_current_device(self):
        lib = _torch_kunlun()
        return 0 if lib is None else int(lib.current_device())

    def set_current_device(self, device):
        lib = _torch_kunlun()
        if lib is not None:
            lib.set_device(int(device))

    def get_current_stream(self, device):
        lib = _torch_kunlun()
        if lib is None:
            return 0
        # Prefer the getter that returns XPUStream directly; reading the first
        # field of torch_kunlun's StreamHandle is the fallback for older builds.
        native = getattr(lib, "current_stream_native", None)
        if native is not None:
            return int(native(int(device)))
        return _native_stream(int(lib.current_stream(int(device)).stream_id))
