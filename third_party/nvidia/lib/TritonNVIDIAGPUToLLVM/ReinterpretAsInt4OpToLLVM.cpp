// Derived from triton-turing (MIT) -- see THIRD_PARTY_NOTICES.md.
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

// Lower tt.reinterpret_as_int4 to a pure register relabel. The operand is an
// int32 dot-operand (kWidth = 1) whose per-thread struct holds N x i32; the
// result is an int4 dot-operand (kWidth = 8) whose per-thread struct holds
// 8 * N x i4 -- the same bits, unpacked. Each i32 bitcasts to a vector of 8 i4
// (LSB-first) and its 8 elements are emitted in order, which is exactly the
// order getValuesFromDotOperandLayoutStruct() re-packs them in (numElemsPerVec
// = 32 / bitwidth = 8, and i4 with kWidth = 8 is not a large-K layout). No data
// moves between threads or registers.
class ReinterpretAsInt4OpPattern
    : public ConvertOpToLLVMPattern<ReinterpretAsInt4Op> {
public:
  ReinterpretAsInt4OpPattern(LLVMTypeConverter &typeConverter,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern<ReinterpretAsInt4Op>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(ReinterpretAsInt4Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i4Ty = rewriter.getIntegerType(4);
    auto i4x8Ty = vec_ty(i4Ty, 8);

    auto packed = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> results;
    results.reserve(packed.size() * 8);
    for (Value word : packed) {
      Value vec = bitcast(word, i4x8Ty);
      for (int i = 0; i < 8; ++i)
        results.push_back(extract_element(i4Ty, vec, i32_val(i)));
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateReinterpretAsInt4OpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ReinterpretAsInt4OpPattern>(typeConverter, benefit);
}
