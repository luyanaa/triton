// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=GFX9
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1010 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=RDNA

#blocked64 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.target" = "hip:gfx942", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // GFX9-LABEL: reduce_add_wave64
  // GFX9: with 322,
  // GFX9: with 323,
  tt.func @reduce_add_wave64(%arg0: tensor<64xf32, #blocked64>) -> f32 {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<64xf32, #blocked64>) -> f32
    tt.return %0 : f32
  }

  // GFX9-LABEL: reduce_max_wave64
  // GFX9-NOT: with 322,
  // GFX9-NOT: with 323,
  // GFX9: rocdl.ds_swizzle
  tt.func @reduce_max_wave64(%arg0: tensor<64xf32, #blocked64>) -> f32 {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %max = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %max : f32
    }) : (tensor<64xf32, #blocked64>) -> f32
    tt.return %0 : f32
  }
  // GFX9-LABEL: reduce_min_wave64
  // GFX9-NOT: with 322,
  // GFX9-NOT: with 323,
  // GFX9: rocdl.ds_swizzle
  tt.func @reduce_min_wave64(%arg0: tensor<64xf32, #blocked64>) -> f32 {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %min = arith.minimumf %lhs, %rhs : f32
      tt.reduce.return %min : f32
    }) : (tensor<64xf32, #blocked64>) -> f32
    tt.return %0 : f32
  }

// -----

#blocked32 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.target" = "hip:gfx1010", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // RDNA-LABEL: reduce_add_wave32
  // RDNA-NOT: with 322,
  // RDNA-NOT: with 323,
  // RDNA: rocdl.ds_swizzle
  tt.func @reduce_add_wave32(%arg0: tensor<32xf32, #blocked32>) -> f32 {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<32xf32, #blocked32>) -> f32
    tt.return %0 : f32
}
