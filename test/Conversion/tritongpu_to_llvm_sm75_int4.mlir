// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s

// Turing (sm75) int4 Tensor Core: s4 x s4 -> s32 via m8n8k32.
// versionMinor = 1 selects the Turing path, kWidth = 8 packs 8 int4 values per
// 32-bit register, and instrShape = [16, 8] is the fixed MMAv2 tile.
//
// i4 cannot go through ttg.local_alloc (the shared-memory element width must be
// a multiple of 8 bits), so real i4 data flow keeps the operand packed as int32
// in shared memory and reinterprets it to i4 at the dot-operand register level
// (see accelerate-matmul-sm75-int4.mlir). This test feeds i4 dot operands
// directly, to isolate the MMA lowering.

#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dot_a = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot_b = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_int4
  tt.func @convert_dot_int4(%a: tensor<16x64xi4, #dot_a>, %b: tensor<64x16xi4, #dot_b>) {
    %cst = arith.constant dense<0> : tensor<16x16xi32, #mma>
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32
    %d = tt.dot %a, %b, %cst : tensor<16x64xi4, #dot_a> * tensor<64x16xi4, #dot_b> -> tensor<16x16xi32, #mma>
    tt.return
  }
}
