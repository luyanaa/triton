#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {

struct DotIntrinsic {
  unsigned vectorSize;
  Type resultElementType;
  StringRef name;
  SmallVector<Value> extraArgs;
};

StringRef getAMDArch(ModuleOp module) {
  auto target = module->getAttrOfType<StringAttr>("triton_gpu.target");
  if (!target)
    return {};
  StringRef value = target.getValue();
  if (value.starts_with("hip:"))
    return value.drop_front(4);
  return value;
}

class AMDFMAVectorMultiplier final : public FMAVectorMultiplier {
  ConversionPatternRewriter &rewriter;
  Location loc;
  DotIntrinsic intrinsic;

  DotIntrinsic chooseIntrinsic(DotOp op) {
    auto aType = cast<RankedTensorType>(op.getA().getType());
    auto bType = cast<RankedTensorType>(op.getB().getType());
    auto dType = cast<RankedTensorType>(op.getD().getType());
    Type aElementType = aType.getElementType();
    Type bElementType = bType.getElementType();
    Type dElementType = dType.getElementType();
    assert(aElementType == bElementType &&
           "AMD FMA dot expects matching input element types");

    ModuleOp module = op->getParentOfType<ModuleOp>();
    StringRef arch = getAMDArch(module);
    if (AMD::supportsVDot(arch)) {
      if (aElementType.isF16() && dElementType.isF32())
        return {2, f32_ty, "llvm.amdgcn.fdot2", {i1_val(false)}};
      if (aElementType.isInteger(8) && !aElementType.isUnsignedInteger() &&
          dElementType.isInteger(32))
        return {4, i32_ty, "llvm.amdgcn.sdot4", {i1_val(false)}};
    }

    // AccelerateBlocked legalizes all other AMD FMA dots to a common floating
    // point type before conversion. Keep this scalar path as the fallback for
    // targets without vector-dot support.
    assert(aElementType == dElementType &&
           (aElementType.isF16() || aElementType.isF32()) &&
           "unsupported AMD FMA dot element types");
    return {1, aElementType,
            aElementType.isF16() ? "llvm.fmuladd.f16" : "llvm.fmuladd.f32",
            {}};
  }

  Value packOperand(ArrayRef<Value> values, unsigned firstElement,
                    unsigned vectorSize) {
    if (vectorSize == 1)
      return values[firstElement];

    Type elementType = values[firstElement].getType();
    Type vectorType = LLVM::getFixedVectorType(elementType, vectorSize);
    Value packed = rewriter.create<LLVM::UndefOp>(loc, vectorType);
    for (unsigned element = 0; element < vectorSize; ++element)
      packed = rewriter.create<LLVM::InsertElementOp>(
          loc, vectorType, packed, values[firstElement + element],
          i32_val(element));
    if (elementType.isInteger(8))
      packed = bitcast(packed, i32_ty);
    return packed;
  }

  Value generateDot(Value a, Value b, Value c) {
    SmallVector<Value> args{a, b, c};
    args.append(intrinsic.extraArgs.begin(), intrinsic.extraArgs.end());
    return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic.name,
                                           intrinsic.resultElementType, args)
        .getResult(0);
  }

public:
  AMDFMAVectorMultiplier(ConversionPatternRewriter &rewriter, DotOp op)
      : rewriter(rewriter), loc(op.getLoc()), intrinsic(chooseIntrinsic(op)) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    assert(a.size() == b.size());
    assert(intrinsic.vectorSize == 1 ||
           a.size() % intrinsic.vectorSize == 0);
    Value accumulator = c;
    for (unsigned k = 0; k < a.size(); k += intrinsic.vectorSize)
      accumulator = generateDot(packOperand(a, k, intrinsic.vectorSize),
                                packOperand(b, k, intrinsic.vectorSize),
                                accumulator);
    return accumulator;
  }
};

} // namespace

namespace mlir::triton::AMD {

LogicalResult convertAMDFMADot(DotOp op, DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter) {
  AMDFMAVectorMultiplier multiplier(rewriter, op);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}

} // namespace mlir::triton::AMD
