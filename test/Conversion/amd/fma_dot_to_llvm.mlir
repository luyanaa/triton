// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"triton_gpu.target" = "hip:gfx942", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: v_dot_i8
  // CHECK-COUNT-4: llvm.call_intrinsic "llvm.amdgcn.sdot4"
  tt.func public @v_dot_i8(
      %arg0: tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<16x16xi32, #blocked>) {
    %cst = arith.constant dense<0> : tensor<16x16xi32, #blocked>
    %0 = tt.dot %arg0, %arg1, %cst, inputPrecision = ieee : tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"triton_gpu.target" = "hip:gfx942", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: v_dot_fp16
  // CHECK-COUNT-8: llvm.call_intrinsic "llvm.amdgcn.fdot2"
  tt.func public @v_dot_fp16(
      %arg0: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<16x16xf32, #blocked>) {
    %cst = arith.constant dense<0.0> : tensor<16x16xf32, #blocked>
    %0 = tt.dot %arg0, %arg1, %cst, inputPrecision = ieee : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}
// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#dot = #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>
module attributes {"triton_gpu.target" = "hip:gfx942", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @fma_shared_k_tail
  // CHECK-NOT: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<2xf32>
  // CHECK-NOT: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
  tt.func public @fma_shared_k_tail(
      %arg0: !tt.memdesc<1x10xf32, #shared, #triton_gpu.shared_memory>)
      -> tensor<1x10xf32, #dot> {
    %0 = triton_gpu.local_load %arg0 : !tt.memdesc<1x10xf32, #shared, #triton_gpu.shared_memory> -> tensor<1x10xf32, #dot>
    tt.return %0 : tensor<1x10xf32, #dot>
  }
}
