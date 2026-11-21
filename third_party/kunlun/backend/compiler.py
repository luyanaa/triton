from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget

# Bumped whenever the generated kernel signature, the emitted code semantics or
# the launcher's parameter buffer changes, so stale entries from the Triton
# cache are not reused.
EMITTER_ABI = 5

# XPU1 local memory per core, minus what the runtime keeps for the kernel's
# spill/scratch area.  Every tensor the reference emitter materialises is a
# full LM array, so this is a hard limit on block size.
DEFAULT_LM_BUDGET = 48 * 1024

EXEC_MODES = ("distributed", "serial")

# The C++ emitter prefixes every failure that exists only because distributed
# mode slices each tensor across the 16 cores of a cluster. Those kernels are
# still correct, just not distributable, so the backend retries them in serial
# mode; a limitation of the emitter carries no marker and stays an error.
DISTRIBUTED_LAYOUT_MARK = "distributed layout:"


@dataclass(frozen=True)
class KunlunOptions:
    num_warps: int = 16
    num_ctas: int = 1
    num_stages: int = 1
    cluster_dims: tuple = (4, 1, 1)
    debug: bool = False
    backend_name: str = "kunlun"
    # "distributed" spreads one program id over the 16 cores of a cluster;
    # "serial" pins it to one core with every tensor fully materialised, which
    # is the golden reference the distributed code is checked against.
    exec_mode: str = "distributed"
    # When distributed mode cannot slice a kernel's layout (a block that does
    # not divide by 16, a broadcast across cores, a reduction that keeps its
    # shape), compile it in serial mode with a warning instead of failing.
    serial_fallback: bool = True

    def __post_init__(self):
        if self.num_warps <= 0 or self.num_warps & (self.num_warps - 1):
            raise ValueError(f"num_warps must be a power of two, got {self.num_warps}")
        if not 1 <= self.num_warps <= 16:
            raise ValueError(f"a K200 cluster has 16 cores, got num_warps={self.num_warps}")
        if self.num_ctas != 1:
            raise ValueError("K200 has no cluster groups; num_ctas must be 1")
        if self.num_stages != 1:
            raise ValueError("XPU1 has no async copy, so num_stages must be 1")
        if tuple(self.cluster_dims) != (4, 1, 1):
            raise ValueError(f"K200 exposes 4 clusters per partition, got {self.cluster_dims}")
        if self.exec_mode not in EXEC_MODES:
            raise ValueError(f"exec_mode must be one of {EXEC_MODES}, got {self.exec_mode!r}")
        if self.exec_mode == "distributed" and self.num_warps != 16:
            # The emitter hands every one of the 16 cores of a K200 cluster a
            # 1/16 slice, so a launch with fewer cores would leave parts of
            # every tensor unprocessed.
            raise ValueError("distributed mode needs all 16 cores of a K200 cluster, so "
                             f"num_warps must be 16; got {self.num_warps} "
                             "(use exec_mode=\"serial\" for smaller launches)")
        object.__setattr__(self, "cluster_dims", tuple(self.cluster_dims))

    def hash(self):
        key = "_".join(f"{name}-{val}" for name, val in sorted(self.__dict__.items()))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KunlunBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "kunlun"

    def __init__(self, target: GPUTarget):
        super().__init__(target)
        self.arch = target.arch
        self.binary_ext = "bin"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in KunlunOptions.__dataclass_fields__ if k in opts}
        return KunlunOptions(**args)

    def load_dialects(self, ctx):
        # triton.compile() has already loaded triton/arith/math/scf for TTIR; the
        # reference emitter emits C, so no extra dialect is involved.
        pass

    def get_codegen_implementation(self) -> Dict[str, Any]:
        return {
            "convert_custom_types": lambda dtype, *args, **kwargs: (None, False),
            "min_dot_size": lambda lhs, rhs: (1, 1, 1),
        }

    def get_module_map(self):
        return {}

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.kernel_hash,
            metadata.mangled,
            metadata.use_cdnn,
        )

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # The inliner has to run: the emitter accepts exactly one tt.func, and
        # @triton.jit helpers otherwise stay behind tt.call.
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_xpuc(src, metadata, opt):
        from triton._C.libtriton import kunlun
        budget = int(os.environ.get("KUNLUN_LM_BYTES", DEFAULT_LM_BUDGET))
        mode = opt.exec_mode
        try:
            source, entry = kunlun.translate_to_xpuc(src, budget, mode == "distributed")
        except RuntimeError as err:
            if mode != "distributed" or not opt.serial_fallback or \
                    DISTRIBUTED_LAYOUT_MARK not in str(err):
                raise
            warnings.warn(
                f"kunlun: this kernel's layout cannot be sliced across the 16 cores of a "
                f"cluster ({err}), so it was compiled with exec_mode='serial': one whole "
                f"program id per core, 1/16 of the throughput and 16x the local memory per "
                f"core. Pass serial_fallback=False to get an error instead.", RuntimeWarning,
                stacklevel=2)
            try:
                source, entry = kunlun.translate_to_xpuc(src, budget, False)
            except RuntimeError as retry_err:
                raise RuntimeError(
                    f"kunlun: distributed mode cannot slice this kernel ({err}), and the "
                    f"serial fallback failed too: {retry_err}") from retry_err
            mode = "serial"
        metadata["entry"] = entry
        metadata["exec_mode"] = mode
        # CompiledKernel requires both: `name` labels the kernel, and `shared`
        # is checked against get_device_properties() before the first launch.
        metadata["name"] = entry
        metadata["shared"] = 0
        return source

    @staticmethod
    def make_bin(src, metadata, opt):
        from triton.backends.kunlun import jitc
        image = jitc.compile_source(src, metadata["entry"])
        metadata["kernel_hash"] = image.hash
        metadata["mangled"] = image.mangled
        metadata["use_cdnn"] = image.is_cdnn
        return image.code

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["xpuc"] = lambda src, metadata: self.make_xpuc(src, metadata, options)
        stages["bin"] = lambda src, metadata: self.make_bin(src, metadata, options)

    def hash(self):
        return f"kunlun-{self.arch}-{EMITTER_ABI}"
