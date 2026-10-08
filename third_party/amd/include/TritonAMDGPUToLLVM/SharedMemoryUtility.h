#ifndef TRITON_AMDGPU_TO_LLVM_SHARED_MEMORY_UTILITY_H
#define TRITON_AMDGPU_TO_LLVM_SHARED_MEMORY_UTILITY_H

#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::LLVM::AMD {

// Return true when the source-to-shared layout writes contiguous LDS chunks
// for the requested vector width.
bool canCoalesceWriteIntoSharedMemory(RankedTensorType srcTy,
                                      triton::MemDescType dstTy,
                                      unsigned vectorSize);

} // namespace mlir::LLVM::AMD

#endif
