#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {
class GenericFMAVectorMultiplier : public FMAVectorMultiplier {
  OpBuilder &builder;
  Location loc;

public:
  GenericFMAVectorMultiplier(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    assert(a.size() == b.size());
    Value accumulator = c;
    for (auto [aElement, bElement] : llvm::zip(a, b))
      accumulator = builder.create<LLVM::FMulAddOp>(loc, aElement, bElement,
                                                    accumulator);
    return accumulator;
  }
};
} // namespace

LogicalResult convertFMADot(DotOp op, DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  GenericFMAVectorMultiplier multiplier(rewriter, op.getLoc());
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}
