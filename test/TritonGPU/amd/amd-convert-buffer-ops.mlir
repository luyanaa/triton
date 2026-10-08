// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: simple
    tt.func @simple(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 :i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    // CHECK: %[[offset:.*]] = arith.addi
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // CHECK: buffer_load %arg0[%[[offset]]]
    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    // CHECK: buffer_load %arg1[%[[offset]]]
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    // CHECK: %[[data:.*]] = arith.addf
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // CHECK: buffer_store %[[data]], %arg2[%[[offset]]]
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: assume_positive_offset
  tt.func @assume_positive_offset(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) ->  tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %sub = arith.subi %1, %c128_i32 : i32
    %cmp = arith.cmpi sgt, %sub, %c0_i32 : i32
    llvm.intr.assume %cmp : i1
    %2 = tt.splat %sub : i32 -> tensor<1024xi32, #blocked>
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[offset:.*]] = arith.addi
    %4 = arith.addi %2, %3 : tensor<1024xi32, #blocked>
    // CHECK: %[[scalar_ptr:.*]] = tt.addptr %arg0
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: buffer_load %[[scalar_ptr]][%[[offset]]]
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %10 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32}  {
  // CHECK-LABEL: offset_64_bits
  tt.func @offset_64_bits(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<1024xf32, #blocked> {
    %c1024_i32 = arith.constant 1024 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %sub = arith.subi %1, %c128_i32 : i32
    %2 = tt.splat %sub : i32 -> tensor<1024xi32, #blocked>
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ext2 = arith.extsi %2 : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
    %ext3 = arith.extsi %3 : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
    %4 = arith.addi %ext2, %ext3 : tensor<1024xi64, #blocked>
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi64, #blocked>
    // CHECK: tt.load
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %10 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32}  {
  // CHECK-LABEL: offset_64_bits_narrow
  tt.func public @offset_64_bits_narrow(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) -> tensor<1024xf32, #blocked> {
    %c1024_i32 = arith.constant 1024 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.splat %1: i32 -> tensor<1024xi32, #blocked>
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ext2 = arith.extsi %2 : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
    %ext3 = arith.extsi %3 : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
    %4 = arith.addi %ext2, %ext3 : tensor<1024xi64, #blocked>
    // CHECK: %[[scalar_ptr:.*]] = tt.addptr %arg0
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[offset_32_bit:.*]] = arith.trunci
    %narrow4 = arith.trunci %4 : tensor<1024xi64, #blocked> to tensor <1024xi32, #blocked>
    %9 = tt.addptr %8, %narrow4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: buffer_load %[[scalar_ptr]][%[[offset_32_bit]]]
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %10 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32}  {
  // CHECK-LABEL: non_canonical_ptr
  tt.func @non_canonical_ptr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: tensor<1024xi32, #blocked>) -> tensor<1024xf32, #blocked>{
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %arg1: tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: tt.load
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %10 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.target" = "hip:gfx940", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: async_copy_target_attr
  // CHECK: amdgpu.buffer_load_to_local %arg0[%[[OFF:.*]]] mask = %[[MASK:.*]] other = %[[OTHER:.*]] into %[[DST:.*]]
  // CHECK: triton_gpu.async_wait
  tt.func @async_copy_target_attr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %offset = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %src = tt.addptr %ptrs, %offset : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    %mask = arith.constant dense<false> : tensor<32xi1, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<32xf32, #blocked>
    %dst = triton_gpu.local_alloc : () -> !tt.memdesc<32xf32, #shared, #triton_gpu.shared_memory, mutable>
    %token = triton_gpu.async_copy_global_to_local %src, %dst mask %mask other %other : tensor<32x!tt.ptr<f32>, #blocked> -> !tt.memdesc<32xf32, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.target" = "hip:gfx1030", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: async_copy_rdna_excluded
  // CHECK: triton_gpu.async_copy_global_to_local %[[SRC:.*]], %[[DST:.*]] mask %[[MASK:.*]] other %[[OTHER:.*]]
  tt.func @async_copy_rdna_excluded(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %offset = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %src = tt.addptr %ptrs, %offset : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %mask = arith.constant dense<false> : tensor<128xi1, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked>
    %dst = triton_gpu.local_alloc : () -> !tt.memdesc<128xf32, #shared, #triton_gpu.shared_memory, mutable>
    %token = triton_gpu.async_copy_global_to_local %src, %dst mask %mask other %other : tensor<128x!tt.ptr<f32>, #blocked> -> !tt.memdesc<128xf32, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #triton_gpu.shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.target" = "hip:gfx940", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: stream_load_to_local
  // CHECK: amdgpu.buffer_load_to_local {{.*}} {OpIdx = #amdgpu.OpIdx<0>}
  // CHECK: triton_gpu.async_wait
  tt.func @stream_load_to_local(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %offset = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ptrs = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
    %src = tt.addptr %ptrs, %offset : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %dst = triton_gpu.local_alloc : () -> !tt.memdesc<128xf16, #shared, #triton_gpu.shared_memory, mutable>
    %loaded = tt.load %src {OpIdx = #amdgpu.OpIdx<0>} : tensor<128x!tt.ptr<f16>, #blocked>
    triton_gpu.local_store %loaded, %dst : tensor<128xf16, #blocked> -> !tt.memdesc<128xf16, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.target" = "hip:gfx940", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: async_copy_preserves_eviction
  // CHECK: triton_gpu.async_copy_global_to_local
  // CHECK-NOT: amdgpu.buffer_load_to_local
  tt.func @async_copy_preserves_eviction(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %offset = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %src = tt.addptr %ptrs, %offset : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %dst = triton_gpu.local_alloc : () -> !tt.memdesc<128xf32, #shared, #triton_gpu.shared_memory, mutable>
    %token = triton_gpu.async_copy_global_to_local %src, %dst evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>, #blocked> -> !tt.memdesc<128xf32, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #triton_gpu.shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.target" = "hip:gfx940", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: stream_load_preserves_volatile
  // CHECK: tt.load {{.*}}isVolatile = true
  // CHECK: triton_gpu.local_store
  // CHECK-NOT: amdgpu.buffer_load_to_local
  tt.func @stream_load_preserves_volatile(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %offset = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ptrs = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
    %src = tt.addptr %ptrs, %offset : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %dst = triton_gpu.local_alloc : () -> !tt.memdesc<128xf16, #shared, #triton_gpu.shared_memory, mutable>
    %loaded = tt.load %src {isVolatile = true} : tensor<128x!tt.ptr<f16>, #blocked>
    triton_gpu.local_store %loaded, %dst : tensor<128xf16, #blocked> -> !tt.memdesc<128xf16, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #triton_gpu.shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.target" = "hip:gfx900", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: stream_load_to_local_gcn5
  // CHECK: amdgpu.buffer_load_to_local
  // CHECK: triton_gpu.async_wait
  tt.func @stream_load_to_local_gcn5(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %offset = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ptrs = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
    %src = tt.addptr %ptrs, %offset : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %dst = triton_gpu.local_alloc : () -> !tt.memdesc<128xf16, #shared, #triton_gpu.shared_memory, mutable>
    %loaded = tt.load %src : tensor<128x!tt.ptr<f16>, #blocked>
    triton_gpu.local_store %loaded, %dst : tensor<128xf16, #blocked> -> !tt.memdesc<128xf16, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
  // CHECK-LABEL: async_copy_stays_global_load_lds_only
  // CHECK: triton_gpu.async_copy_global_to_local
  // CHECK-NOT: amdgpu.buffer_load_to_local
  tt.func @async_copy_stays_global_load_lds_only(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %offset = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ptrs = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
    %src = tt.addptr %ptrs, %offset : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %dst = triton_gpu.local_alloc : () -> !tt.memdesc<128xf16, #shared, #triton_gpu.shared_memory, mutable>
    %token = triton_gpu.async_copy_global_to_local %src, %dst : tensor<128x!tt.ptr<f16>, #blocked> -> !tt.memdesc<128xf16, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #triton_gpu.shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.target" = "hip:gfx906", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: stream_load_to_local_vega20
  // CHECK: amdgpu.buffer_load_to_local
  // CHECK: triton_gpu.async_wait
  tt.func @stream_load_to_local_vega20(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %offset = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ptrs = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
    %src = tt.addptr %ptrs, %offset : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %dst = triton_gpu.local_alloc : () -> !tt.memdesc<128xf16, #shared, #triton_gpu.shared_memory, mutable>
    %loaded = tt.load %src : tensor<128x!tt.ptr<f16>, #blocked>
    triton_gpu.local_store %loaded, %dst : tensor<128xf16, #blocked> -> !tt.memdesc<128xf16, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}
