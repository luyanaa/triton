// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

// Turing (sm75) accepts fewer element types than Ampere, and supportMMA() only
// carries Ampere's and later's rules. A dot whose type Turing has no `mma.sync`
// form for must not keep an MMA layout: MMAv2 has no instruction-table entry
// for it and aborts (`mmaInstructions.at(mmaType)`), so the dot has to fall
// back to FMA instead. This pins the gate for every element type that reaches
// it, plus Ampere staying untouched.

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.target" = "cuda:75", "triton_gpu.threads-per-warp" = 32 : i32} {
  // int8 is the Turing s8 form (m8n8k16): it keeps an MMA layout, and the
  // encoding must say versionMinor 1 -- that is what makes isTuring() true and
  // routes codegen to the m8n8k16 instruction.
  // CHECK-LABEL: @i8_dot_sm75
  // CHECK: tt.dot {{.*}} -> tensor<16x8xi32, #mma>
  tt.func public @i8_dot_sm75(%a: tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
                              %b: tensor<16x8xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>)
      -> tensor<16x8xi32, #blocked> {
    %cst = arith.constant dense<0> : tensor<16x8xi32, #blocked>
    %d = tt.dot %a, %b, %cst : tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> *
                                tensor<16x8xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> ->
                                tensor<16x8xi32, #blocked>
    tt.return %d : tensor<16x8xi32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.target" = "cuda:75", "triton_gpu.threads-per-warp" = 32 : i32} {
  // f16 shares Ampere's m16n8k8 fragment map, so the operand packing is valid.
  // CHECK-LABEL: @f16_dot_sm75
  // CHECK: tt.dot {{.*}} -> tensor<16x8xf32, #mma>
  tt.func public @f16_dot_sm75(%a: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
                               %b: tensor<16x8xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>)
      -> tensor<16x8xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x8xf32, #blocked>
    %d = tt.dot %a, %b, %cst : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> *
                                tensor<16x8xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> ->
                                tensor<16x8xf32, #blocked>
    tt.return %d : tensor<16x8xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.target" = "cuda:75", "triton_gpu.threads-per-warp" = 32 : i32} {
  // bf16 has no Turing MMA form. The dot keeps a blocked result and
  // decomposeMixedModeDotOp widens the operands to f32, i.e. the FMA path.
  // CHECK-LABEL: @bf16_dot_sm75
  // CHECK: arith.extf {{.*}} : tensor<16x16xbf16, {{.*}}> to tensor<16x16xf32
  // CHECK: tt.dot {{.*}} tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> {{.*}} -> tensor<16x8xf32, #blocked>
  tt.func public @bf16_dot_sm75(%a: tensor<16x16xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
                                %b: tensor<16x8xbf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>)
      -> tensor<16x8xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x8xf32, #blocked>
    %d = tt.dot %a, %b, %cst, inputPrecision = tf32 : tensor<16x16xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> *
                                                      tensor<16x8xbf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> ->
                                                      tensor<16x8xf32, #blocked>
    tt.return %d : tensor<16x8xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.target" = "cuda:75", "triton_gpu.threads-per-warp" = 32 : i32} {
  // tf32 is the default input precision of tl.dot, but mma...tf32 is sm80+.
  // supportMMA() admits it for version 2, so without the gate this dot would
  // carry an MMA layout and abort in MMAv2.
  // CHECK-LABEL: @f32_dot_sm75
  // CHECK: tt.dot {{.*}} -> tensor<16x8xf32, #blocked>
  tt.func public @f32_dot_sm75(%a: tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
                               %b: tensor<16x8xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>)
      -> tensor<16x8xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x8xf32, #blocked>
    %d = tt.dot %a, %b, %cst, inputPrecision = tf32 : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> *
                                                      tensor<16x8xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> ->
                                                      tensor<16x8xf32, #blocked>
    tt.return %d : tensor<16x8xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.target" = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  // Ampere has a native bf16 MMA (m16n8k16): the gate is scoped to pre-Ampere
  // targets and must not touch it.
  // CHECK-LABEL: @bf16_dot_sm80
  // CHECK: tt.dot {{.*}} -> tensor<16x8xf32, #mma>
  tt.func public @bf16_dot_sm80(%a: tensor<16x16xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
                                %b: tensor<16x8xbf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>)
      -> tensor<16x8xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x8xf32, #blocked>
    %d = tt.dot %a, %b, %cst : tensor<16x16xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> *
                                tensor<16x8xbf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> ->
                                tensor<16x8xf32, #blocked>
    tt.return %d : tensor<16x8xf32, #blocked>
  }
}
