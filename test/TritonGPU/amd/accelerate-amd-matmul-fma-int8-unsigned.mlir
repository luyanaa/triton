// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx942 matrix-instruction-size=0' -verify-diagnostics

#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @unsigned_i8_dot(
      %arg0: tensor<2x64xui8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x64xui8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) {
    %acc = arith.constant dense<16777217> : tensor<2x64xi32, #blocked>
    // expected-error @+1 {{INT8 dot with INT32 accumulator has no exact integer lowering on this target; FP32 legalization is lossy}}
    %0 = tt.dot %arg0, %arg1, %acc : tensor<2x64xui8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xui8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<2x64xi32, #blocked>
    tt.return
  }
}
