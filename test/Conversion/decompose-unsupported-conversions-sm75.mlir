// RUN: triton-opt %s -split-input-file --decompose-unsupported-nvidia-conversions | FileCheck %s

// Turing's s8/s4 dot operands must not take the mma -> dot_operand register
// shortcut. The dot-operand layout is shared with Ampere (kWidth =
// 32/bitwidth), so the mma layout looks like a register-permutation away from
// it, but the hardware packing is not: Turing's m8n8k16 / m8n8k32 read the
// K-adjacent elements of each 32-bit word differently from Ampere's m16n8k32.
// Relabeling the accumulator registers therefore mispacks the operand (this is
// the "generated INT8 dot operand is garbage on sm75" failure). Declining the
// shortcut here sends the conversion through shared memory instead, where
// SharedToDotOperandMMAv2 implements the Turing packing.
//
// f16 is the exception: both targets use m16n8k8, so their fragments really do
// line up and the register shortcut stays valid.

#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot_a_i8 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.target" = "cuda:75", "triton_gpu.threads-per-warp" = 32 : i32} {
  // The A operand is decomposed into mma -> blocked -> dot_operand. The second
  // conversion stays a convert_layout rather than becoming local_alloc: the
  // pass's blocked -> dot_operand step is skipped for isAmpere() targets
  // (Turing included) because the codegen lowers that pair through its own
  // shared-memory scratch.
  // CHECK-LABEL: tt.func @turing_i8_mma_to_a_operand
  // CHECK: %[[TMP:.*]] = triton_gpu.convert_layout %arg0 : tensor<16x32xi8, #mma> -> tensor<16x32xi8, #[[BLK:[a-zA-Z0-9_]+]]>
  // CHECK: triton_gpu.convert_layout %[[TMP]] : tensor<16x32xi8, #[[BLK]]> -> tensor<16x32xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
  tt.func @turing_i8_mma_to_a_operand(%arg0: tensor<16x32xi8, #mma>) -> tensor<16x32xi8, #dot_a_i8> {
    %cvt = triton_gpu.convert_layout %arg0 : tensor<16x32xi8, #mma> -> tensor<16x32xi8, #dot_a_i8>
    tt.return %cvt : tensor<16x32xi8, #dot_a_i8>
  }
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot_b_i8 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.target" = "cuda:75", "triton_gpu.threads-per-warp" = 32 : i32} {
  // The B operand has the same problem and nothing else covers it: generated B
  // operands are common in attention backward (dS feeds dQ and dK), and a
  // peephole that only knows how to materialize opIdx 0 would miss them.
  // CHECK-LABEL: tt.func @turing_i8_mma_to_b_operand
  // CHECK: %[[TMP:.*]] = triton_gpu.convert_layout %arg0 : tensor<32x16xi8, #mma> -> tensor<32x16xi8, #[[BLK:[a-zA-Z0-9_]+]]>
  // CHECK: triton_gpu.convert_layout %[[TMP]] : tensor<32x16xi8, #[[BLK]]> -> tensor<32x16xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
  tt.func @turing_i8_mma_to_b_operand(%arg0: tensor<32x16xi8, #mma>) -> tensor<32x16xi8, #dot_b_i8> {
    %cvt = triton_gpu.convert_layout %arg0 : tensor<32x16xi8, #mma> -> tensor<32x16xi8, #dot_b_i8>
    tt.return %cvt : tensor<32x16xi8, #dot_b_i8>
  }
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot_a_f16 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.target" = "cuda:75", "triton_gpu.threads-per-warp" = 32 : i32} {
  // f16 keeps the shortcut: a single conversion, no blocked hop.
  // CHECK-LABEL: tt.func @turing_f16_mma_to_a_operand
  // CHECK: triton_gpu.convert_layout %arg0 : tensor<16x32xf16, #mma> -> tensor<16x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  tt.func @turing_f16_mma_to_a_operand(%arg0: tensor<16x32xf16, #mma>) -> tensor<16x32xf16, #dot_a_f16> {
    %cvt = triton_gpu.convert_layout %arg0 : tensor<16x32xf16, #mma> -> tensor<16x32xf16, #dot_a_f16>
    tt.return %cvt : tensor<16x32xf16, #dot_a_f16>
  }
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot_a_i8 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.target" = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  // Ampere's m16n8k32 fragment is what the dot-operand layout models, so int8
  // keeps the register shortcut there. Pinned so the Turing rule cannot spread
  // to sm80.
  // CHECK-LABEL: tt.func @ampere_i8_mma_to_a_operand
  // CHECK: triton_gpu.convert_layout %arg0 : tensor<16x32xi8, #mma> -> tensor<16x32xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
  tt.func @ampere_i8_mma_to_a_operand(%arg0: tensor<16x32xi8, #mma>) -> tensor<16x32xi8, #dot_a_i8> {
    %cvt = triton_gpu.convert_layout %arg0 : tensor<16x32xi8, #mma> -> tensor<16x32xi8, #dot_a_i8>
    tt.return %cvt : tensor<16x32xi8, #dot_a_i8>
  }
}
