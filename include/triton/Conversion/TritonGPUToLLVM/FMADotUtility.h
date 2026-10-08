#ifndef TRITON_CONVERSION_FMA_DOT_UTILITY_H
#define TRITON_CONVERSION_FMA_DOT_UTILITY_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::gpu {

/// Abstract interface for multiplying two vectors and accumulating into c.
/// Backends can replace the scalar products with native vector-dot
/// instructions while sharing the layout traversal below.
class FMAVectorMultiplier {
public:
  virtual Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                                Value c) = 0;
  virtual ~FMAVectorMultiplier() = default;
};

/// Convert a blocked DotOp using a backend-provided vector multiplier.
LogicalResult parametricConvertFMADot(DotOp op, DotOp::Adaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      FMAVectorMultiplier &multiplier);

} // namespace mlir::triton::gpu

#endif // TRITON_CONVERSION_FMA_DOT_UTILITY_H
