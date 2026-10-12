// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

// Turing (sm75) int4 Tensor Core: end-to-end layout assignment for a packed
// int4 dot. The frontend emits `tt.reinterpret_as_int4(%packed_i32)` feeding a
// tt.dot whose operands carry a blocked dot-operand layout. BlockedToMMA must
// NOT give the i4 tensor a dot-operand layout through a convert_layout on i4:
// that routes through shared memory, and an i4 local_alloc is illegal (the
// shared element width must be a multiple of 8 bits).
//
// convertDotOperandForMMA intercepts instead: it applies the dot-operand layout
// to the int32 SOURCE (kWidth = 1, legal for 32-bit) and re-creates the
// reinterpret on top, so the i4 tensor only ever exists in dot-operand form
// (kWidth = 8). This test pins that invariant: an int32 convert_layout and an
// i4 reinterpret at the dot-operand level, an i4 dot fed to an nvidia_mma
// result, and no i4 local_alloc / i4 convert_layout.

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked6 = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>

// CHECK: #mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 1, {{.*}}instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.target" = "cuda:75", "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @int4_dot
  tt.func public @int4_dot(%a_packed: tensor<16x8xi32, #blocked>, %b_packed: tensor<8x8xi32, #blocked>) -> tensor<16x8xi32, #blocked6> {
    %cst = arith.constant dense<0> : tensor<16x8xi32, #blocked6>

    // A: the int32 source gets the dot-operand layout (kWidth = 1), then the
    // reinterpret relabels it to an i4 dot operand (kWidth = 8) -- no i4
    // convert_layout.
    // CHECK: triton_gpu.convert_layout %{{.*}} : tensor<16x8xi32, #blocked> -> tensor<16x8xi32, #triton_gpu.dot_op<{{.*}}kWidth = 1}>>
    // CHECK: tt.reinterpret_as_int4 %{{.*}} {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.dot_op<{{.*}}kWidth = 1}>> -> tensor<16x64xi4, #triton_gpu.dot_op<{{.*}}kWidth = 8}>>
    %a = tt.reinterpret_as_int4 %a_packed {axis = 1 : i32} : tensor<16x8xi32, #blocked> -> tensor<16x64xi4, #blocked5>

    // B: same on opIdx = 1.
    // CHECK: triton_gpu.convert_layout %{{.*}} : tensor<8x8xi32, #blocked> -> tensor<8x8xi32, #triton_gpu.dot_op<{{.*}}kWidth = 1}>>
    // CHECK: tt.reinterpret_as_int4 %{{.*}} {axis = 0 : i32} : tensor<8x8xi32, #triton_gpu.dot_op<{{.*}}kWidth = 1}>> -> tensor<64x8xi4, #triton_gpu.dot_op<{{.*}}kWidth = 8}>>
    %b = tt.reinterpret_as_int4 %b_packed {axis = 0 : i32} : tensor<8x8xi32, #blocked> -> tensor<64x8xi4, #blocked>

    %a_dot = triton_gpu.convert_layout %a : tensor<16x64xi4, #blocked5> -> tensor<16x64xi4, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked6}>>
    %b_dot = triton_gpu.convert_layout %b : tensor<64x8xi4, #blocked> -> tensor<64x8xi4, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked6}>>

    // The i4 dot operands (kWidth = 8) feed an nvidia_mma result.
    // CHECK: tt.dot %{{.*}} : tensor<16x64xi4, {{.*}}kWidth = 8}>> * tensor<64x8xi4, {{.*}}kWidth = 8}>> -> tensor<16x8xi32, #mma>
    %d = tt.dot %a_dot, %b_dot, %cst, inputPrecision = tf32 : tensor<16x64xi4, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked6}>> * tensor<64x8xi4, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked6}>> -> tensor<16x8xi32, #blocked6>

    // The i4 tensor never routes through shared memory.
    // CHECK-NOT: triton_gpu.local_alloc
    tt.return %d : tensor<16x8xi32, #blocked6>
  }
}
