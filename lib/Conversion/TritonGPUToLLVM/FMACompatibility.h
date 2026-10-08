#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_FMA_COMPATIBILITY_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_FMA_COMPATIBILITY_H

#include <cstddef>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu::fma_compat {

/// Expand a matrix shape to batch x M x N form.
template <typename T>
SmallVector<T> expandMatrixShapeWithBatch(ArrayRef<T> shape) {
  assert(shape.size() == 2 || shape.size() == 3);
  if (shape.size() == 3)
    return SmallVector<T>(shape);
  return {T{1}, shape[0], shape[1]};
}

/// Expand a matrix order, keeping the batch dimension slowest-changing.
inline SmallVector<unsigned>
expandMatrixOrderWithBatch(ArrayRef<unsigned> order) {
  assert(order.size() == 2 || order.size() == 3);
  if (order.size() == 3)
    return SmallVector<unsigned>(order);
  SmallVector<unsigned> expanded(3, 0);
  for (unsigned i = 0; i < order.size(); ++i)
    expanded[i] = order[i] + 1;
  return expanded;
}

inline SmallVector<unsigned> delinearizeIndex(unsigned linear,
                                              ArrayRef<unsigned> shape,
                                              ArrayRef<unsigned> order) {
  assert(order.size() == shape.size());
  SmallVector<unsigned> multiDim(shape.size());
  for (unsigned dim : order) {
    multiDim[dim] = linear % shape[dim];
    linear /= shape[dim];
  }
  assert(linear == 0);
  return multiDim;
}

inline std::size_t linearizeIndex(ArrayRef<unsigned> multiDim,
                                  ArrayRef<unsigned> shape,
                                  ArrayRef<unsigned> order) {
  assert(multiDim.size() == shape.size() && shape.size() == order.size());
  std::size_t linear = 0;
  for (std::size_t i = order.size(); i > 0; --i) {
    unsigned dim = order[i - 1];
    linear = linear * shape[dim] + multiDim[dim];
  }
  return linear;
}

} // namespace mlir::triton::gpu::fma_compat

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_FMA_COMPATIBILITY_H
