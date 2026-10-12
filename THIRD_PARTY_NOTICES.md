# Third-party notices

## triton-turing

This repository is a Triton 3.2 fork with pre-Ampere (sm70/sm75) NVIDIA work.
Part of that work — the software pipeline without `cp.async`, the int4 Tensor
Core path, the pre-Ampere MMA element-type gates and the bf16 conversions that
avoid sm80-only PTX — is derived from **triton-turing**
(<https://github.com/Chennesxu/triton-turing>).

triton-turing is distributed under the MIT license and carries the same
copyright notice as upstream Triton:

```
Copyright 2018-2020 Philippe Tillet
Copyright 2020-2022 OpenAI
```

That notice and the permission notice are included in this repository's
[`LICENSE`](LICENSE) — triton-turing added no copyright line of its own, so the
LICENSE file here satisfies the MIT notice requirement for the borrowed
portions as well. This file records *provenance*: which parts of this tree came
from where, so a reader can follow them back to the original.

### What is derived

| Area | Files here | Reference in triton-turing |
|---|---|---|
| Synchronous-copy software pipeline (`ld.global` → `st.shared`, one `bar.sync` per k-tile, shared-memory clamp, pipeline-depth reporting) | `lib/Dialect/TritonGPU/Transforms/Pipeliner/MatmulLoopPipeline.cpp`, `PipeliningUtility.cpp`, `lib/Dialect/TritonGPU/Transforms/Prefetch.cpp`, `include/triton/Dialect/TritonGPU/IR/Dialect.h`, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp`, `third_party/nvidia/backend/compiler.py` | the sync-copy branch of `Pipeliner/LowerLoops.cpp` and `createSyncCopy`/`placeSyncBarriers`, the latency clamp in `Pipeliner/AssignLatencies.cpp`, the prefetch guards in `Prefetch.cpp` |
| int4 Tensor Core MMA (`m8n8k32`): packed-int32 staging, the dot-operand relabel, the MMAv2 entry, `min_dot_size`, the tutorial | `include/triton/Dialect/Triton/IR/TritonOps.td`, `lib/Dialect/Triton/IR/Ops.cpp`, `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp`, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ReinterpretAsInt4OpToLLVM.cpp`, `python/triton/language/{core,semantic}.py`, `python/src/ir.cc`, `python/tutorials/12-turing-integer-matmul.py` | commit `72ef6d7` (“int4 Tensor Core MMA end-to-end (Turing sm75)”) and the tutorial `python/tutorials/12-turing-integer-matmul.py` |
| Pre-Ampere MMA element-type gates (`getMMAVersionSafe` filter, Turing register-shortcut gate, prefetch width guard) | `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DecomposeUnsupportedConversions.cpp`, `lib/Dialect/TritonGPU/Transforms/Prefetch.cpp`, `lib/Analysis/Utility.cpp` | commits `d512594`, `28b0f97`-era work, and the `canPromoteSplitDot`/`sameElementWidth` conditions in their `Prefetch.cpp` |
| bf16 conversions below sm80 (integer shift upcast, RTZ/RTNE downcast, NaN guard) | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp` | commit `e57048e` (“lower f32->bf16 RTZ without cvt.rz.bf16.f32”) |
| Turing autotune configuration spaces | `python/tutorials/03-matrix-multiplication.py`, `06-fused-attention.py`, `08-grouped-gemm.py` | the `get_turing_*_config*` spaces and their measured rationale |

Comments in the files above that explain the reasoning (barrier invariants,
why a store is marked `ttg.pipeline_ring_store`, why bf16 needs a NaN guard, why
int4 must stay out of shared memory) follow triton-turing's write-ups of the
same behaviours.

### Verification

The sm70/sm75 paths here were validated on this fork's own terms: lit tests
(`test/TritonGPU/accelerate-matmul-sm75*.mlir`,
`test/Conversion/*sm75*.mlir`) and end-to-end checks on a T4 (sm75) — INT8 and
INT4 GEMM bit-exact against an int32 reference, bf16 running on the FMA path,
and the pipeline engaging with identical results at `num_stages` 1/2/3. See the
commit message for the numbers.
