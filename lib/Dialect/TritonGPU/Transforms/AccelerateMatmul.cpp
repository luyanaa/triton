// The pre-Ampere MMA element-type gate, the int4 operand interception and the
// bf16 mixed-mode promotion are derived from triton-turing (MIT) -- see
// THIRD_PARTY_NOTICES.md.
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {
namespace gpu {

namespace {

static bool isFp8Type(Type t) {
  return t.isFloat8E5M2() || t.isFloat8E4M3FN() || t.isFloat8E4M3FNUZ() ||
         t.isFloat8E5M2FNUZ();
}

// Element types the pre-Ampere `mma.sync` instruction set has a form for:
// f16 (m16n8k8), s8 (m8n8k16) and s4 (m8n8k32). fp8 is admitted because
// decomposeMixedModeDotOp promotes it to f16 below sm89, which is supported.
//
// supportMMA() carries Ampere's and later's rules only, so on Turing it admits
// bf16, tf32 and f64 dots as well. Those keep an MMA layout and then reach
// MMAv2, whose Turing instruction table has no row for them: getMmaType() maps
// bf16 to FP32_BF16_BF16_F32 and an f32 tf32 dot to FP32_TF32_TF32_FP32, and
// `mmaInstructions.at(mmaType)` aborts on the missing key. Refusing the layout
// here sends those dots to FMA, which is what a target without a tensor-core
// form would do.
static bool isSupportedByTuringMMA(Type aElemTy, Type bElemTy) {
  if (isFp8Type(aElemTy) && isFp8Type(bElemTy))
    return true;
  if (aElemTy != bElemTy)
    return false;
  return aElemTy.isF16() || aElemTy.isInteger(8) || aElemTy.isInteger(4);
}

// Get the highest version supported for the hardware and the dot.
static int getMMAVersionSafe(int computeCapability, DotOp op) {
  // List supported mma version in order of preference.
  SmallVector<int> versionsSupported;
  if (computeCapability < 75) {
    versionsSupported = {1};
  } else if (computeCapability < 90) {
    versionsSupported = {2};
  } else if (computeCapability < 100) {
    versionsSupported = {3, 2};
  } else {
    assert(false && "computeCapability not supported");
  }
  for (int baseVersion : versionsSupported) {
    if (supportMMA(op, baseVersion)) {
      // Pre-Ampere targets have a smaller instruction set than supportMMA()
      // encodes; the version-2 path below is the Turing one when
      // computeCapability == 75 (see createMMAEncodingForDot).
      if (computeCapability < 80 && baseVersion >= 2 &&
          !isSupportedByTuringMMA(op.getA().getType().getElementType(),
                                  op.getB().getType().getElementType()))
        continue;
      return baseVersion;
    }
    if (baseVersion == 3)
      op.emitRemark() << "Warning: can't use MMA V3 for the dot op";
  }
  return 0;
}

SmallVector<unsigned> warpsPerTileV2(DotOp dotOp, const ArrayRef<int64_t> shape,
                                     int numWarps) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};

  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion() &&
           !isa<TransOp>(op);
  };
  auto slices = multiRootGetSlice(dotOp, {filter}, {filter});
  bool hasChainedDot = false;
  for (Operation *op : slices) {
    if (isa<DotOp>(op) && (op != dotOp)) {
      auto chainedDot = cast<DotOp>(op);
      auto resTy = chainedDot.getResult().getType();
      if (resTy.getRank() != rank) {
        continue;
      }
      if (auto mmaEncoding =
              dyn_cast<NvidiaMmaEncodingAttr>(resTy.getEncoding())) {
        return getWarpsPerCTA(mmaEncoding);
      }
      hasChainedDot = true;
    }
  }
  if (hasChainedDot) {
    if (shape[0] >= shape[1]) {
      return {(unsigned)numWarps, 1};
    } else {
      return {1, (unsigned)numWarps};
    }
  }

  SmallVector<unsigned> ret(rank, 1);
  SmallVector<int64_t> shapePerWarp(rank, 1);
  shapePerWarp[rank - 1] = 8;
  shapePerWarp[rank - 2] = 16;
  // TODO (@daadaada): double-check.
  // original logic in
  // https://github.com/triton-lang/triton/blob/master/lib/codegen/analysis/layout.cc#L252
  // seems buggy for shape = [32, 16] ?
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

SmallVector<unsigned, 2>
warpsPerTileV3(DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
               const SmallVector<unsigned, 3> &instrShape) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getResult(), &slices);
  // Contains a chained dot. We prefer to assign warps to one axis
  // to facilitate use cases like flash attention, allowing reductions within
  // the same warp.
  if (llvm::find_if(slices, [](Operation *op) {
        return op->hasTrait<OpTrait::DotLike>();
      }) != slices.end())
    return {(unsigned)numWarps, 1};

  // For MMAv3, the smallest indivisible unit of warp shape is (4, 1).
  SmallVector<unsigned, 2> ret = {4, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, instrShape[1]};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] > shapePerWarp[0] * ret[0]) {
      ret[0] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

// Give a dot operand its dot-operand layout, at the bitwidth the instruction
// actually consumes.
//
// int4 is special: the operand is `tt.reinterpret_as_int4(%packed_i32)` and the
// i4 tensor cannot take a dot-operand layout directly. That would need a
// blocked -> dot_op convert_layout on i4, which routes through shared memory,
// and an i4 local_alloc is illegal (the shared element width must be a multiple
// of 8 bits). Instead apply the dot-operand layout to the int32 SOURCE
// (kWidth = 1, legal for 32 bits) and re-create the reinterpret on top, so the
// i4 tensor only ever exists in dot-operand form (kWidth = 8). The reinterpret
// is a pure register relabel: 4x i32 == 32x i4 == 128 bits per thread.
static Value convertDotOperandForMMA(Value v, int opIdx, int bitwidth,
                                     RankedTensorType newRetType,
                                     mlir::PatternRewriter &rewriter) {
  // The operand may be wrapped in convert_layout ops inserted while giving the
  // reinterpret result a dot-operand encoding; look through them to the
  // reinterpret.
  Value reintVal = v;
  while (auto cvt = reintVal.getDefiningOp<ConvertLayoutOp>())
    reintVal = cvt.getSrc();
  if (auto reint = reintVal.getDefiningOp<triton::ReinterpretAsInt4Op>()) {
    Value packed = reint.getSrc();
    auto packedType = cast<RankedTensorType>(packed.getType());
    auto i32Type = rewriter.getIntegerType(32);
    auto i4Type = rewriter.getIntegerType(4);
    // int32 dot-operand encoding (kWidth derives to 1).
    auto i32Encoding = DotOperandEncodingAttr::get(
        v.getContext(), opIdx, newRetType.getEncoding(), i32Type);
    auto i32DotType = RankedTensorType::get(
        packedType.getShape(), packedType.getElementType(), i32Encoding);
    Value i32Dot =
        rewriter.create<ConvertLayoutOp>(packed.getLoc(), i32DotType, packed);
    // int4 dot-operand encoding (kWidth derives to 8); the shape is the
    // unpacked i4 shape the reinterpret result carries.
    auto i4Encoding = DotOperandEncodingAttr::get(
        v.getContext(), opIdx, newRetType.getEncoding(), i4Type);
    auto reintTy = cast<RankedTensorType>(reint.getType());
    auto i4DotType = RankedTensorType::get(reintTy.getShape(),
                                          reintTy.getElementType(), i4Encoding);
    return rewriter.create<triton::ReinterpretAsInt4Op>(
        reint.getLoc(), i4DotType, i32Dot, reint.getAxis());
  }
  auto minType = bitwidth > 0 ? rewriter.getIntegerType(bitwidth) : v.getType();
  auto vType = cast<RankedTensorType>(v.getType());
  auto newVEncoding = DotOperandEncodingAttr::get(
      v.getContext(), opIdx, newRetType.getEncoding(),
      bitwidth > 0 ? minType : vType.getElementType());
  auto newVType = RankedTensorType::get(vType.getShape(),
                                        vType.getElementType(), newVEncoding);
  return rewriter.create<ConvertLayoutOp>(v.getLoc(), newVType, v);
}

// Returns a shared memory allocation that can be used by a dotMMA op for the
// given value.
static Value getSharedMemoryMMAOperand(Value v, mlir::PatternRewriter &rewriter,
                                       int opIdx, bool allowTranspose) {
  OpBuilder::InsertionGuard g(rewriter);
  Value arg = v;
  if (auto cvtOp = v.getDefiningOp<ConvertLayoutOp>())
    arg = cvtOp.getSrc();
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto newOrder = getOrder(argType.getEncoding());

  // If the MMA op doesn't support transpose pick the layout expected by the MMA
  // op.
  if (!allowTranspose) {
    if (opIdx == 1) {
      newOrder = {0, 1};
    } else {
      newOrder = {1, 0};
    }
  }

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CTALayout = getCTALayout(argType.getEncoding());
  auto newLayout =
      SharedEncodingAttr::get(argType.getContext(), argType.getShape(),
                              newOrder, CTALayout, argType.getElementType());
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return rewriter.create<LocalAllocOp>(arg.getLoc(), newType, arg);
}

class BlockedToMMA : public mlir::OpRewritePattern<DotOp> {
  int computeCapability;
  mutable int mmaV1Counter{}; // used to generate ID for MMAv1 encoding
  mutable llvm::DenseMap<Operation *, unsigned> dotOpInstNs;

  static bool bwdFilter(Operation *op) {
    return op->getNumOperands() == 1 &&
           (isa<FpToFpOp, BitcastOp, ConvertLayoutOp>(op) ||
            isPureUnaryInlineAsm(op) ||
            op->getDialect()->getTypeID() ==
                mlir::TypeID::get<arith::ArithDialect>());
  }

  // Finds the first different bitwidth in the chain of shape-preserving
  // unary ops that x depends on.
  // There are two primary scenarios:
  // (1) Upcasting: A sequence such as loading an fp16, followed by arithmetic
  // operations, then bitcasting to fp32, and finally computing in fp32.
  // (2) Downcasting: This might involve loading an fp32, performing arithmetic
  // operations, bitcasting to fp16, and finally computing in fp16.
  // In the upcasting scenario, element reordering converts the original
  // elements distribution to the order of higher precision primitives. As a
  // result, kwidth can be the bitwidth of the lower precision primitive.
  // Conversely, in the downcasting scenario, no reordering is performed,
  // making it directly use the lower precision primitive.
  static int computeOrigBitWidth(Value x) {
    int finalBitWidth = getElementTypeOrSelf(x).getIntOrFloatBitWidth();
    int origBitWidth = finalBitWidth;
    SetVector<Operation *> slice;
    mlir::BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = bwdFilter;
    getBackwardSlice(x, &slice, opt);
    for (auto op : slice) {
      if (Value arg = op->getOperand(0))
        if (auto argTy = dyn_cast<RankedTensorType>(arg.getType())) {
          auto argBitWidth = argTy.getElementType().getIntOrFloatBitWidth();
          if (argBitWidth != origBitWidth) {
            origBitWidth = std::min<int>(origBitWidth, argBitWidth);
            break;
          }
        }
    }
    return origBitWidth;
  }

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability)
      : OpRewritePattern<DotOp>(context), computeCapability(computeCapability) {
  }

  static SmallVector<unsigned, 3>
  getWarpsPerTile(DotOp dotOp, const ArrayRef<int64_t> shape, int version,
                  int numWarps, const SmallVector<unsigned, 3> &instrShape) {
    switch (version) {
    case 2:
      return warpsPerTileV2(dotOp, shape, numWarps);
    case 3:
      return warpsPerTileV3(dotOp, shape, numWarps, instrShape);
    default:
      assert(false && "not supported version");
      return {0, 0};
    }
  }

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    // TODO: Check data-types and SM compatibility
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    auto mod = dotOp->getParentOfType<mlir::ModuleOp>();
    int numWarps = TritonGPUDialect::getNumWarps(mod);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());

    int versionMajor = getMMAVersionSafe(computeCapability, dotOp);
    if (!(versionMajor >= 1 && versionMajor <= 3))
      return failure();

    auto instrShape = mmaVersionToInstrShape(
        versionMajor, retShapePerCTA, dotOp.getA().getType().getElementType(),
        numWarps);
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = dotOp.getA().getType();
    auto oldBType = dotOp.getB().getType();

    NvidiaMmaEncodingAttr mmaEnc;
    if (versionMajor == 1) {
      SetVector<Operation *> aBwdSlices, bBwdSlices;
      auto isCvt = [](Operation *op) { return isa<ConvertLayoutOp>(op); };
      mlir::BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      opt.filter = isCvt;
      getBackwardSlice(a, &aBwdSlices, opt);
      getBackwardSlice(b, &bBwdSlices, opt);
      // get the source of the first conversion found in slices
      auto getCvtArgOrder = [](Operation *op) {
        return mlir::cast<BlockedEncodingAttr>(
                   cast<ConvertLayoutOp>(op).getSrc().getType().getEncoding())
            .getOrder();
      };
      bool isARow = true;
      bool isBRow = true;
      Operation *aOp = a.getDefiningOp();
      Operation *bOp = b.getDefiningOp();
      if (!aBwdSlices.empty())
        aOp = aBwdSlices[0];
      if (!bBwdSlices.empty())
        bOp = bBwdSlices[0];
      if (aOp)
        isARow = getCvtArgOrder(aOp)[0] == 1;
      if (bOp)
        isBRow = getCvtArgOrder(bOp)[0] == 1;

      mmaEnc = NvidiaMmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor, numWarps, CTALayout,
          instrShape, oldAType.getShape(), oldBType.getShape(), retShapePerCTA,
          isARow, isBRow, mmaV1Counter++);
    } else {
      assert(versionMajor == 2 || versionMajor == 3);
      int versionMinor = computeCapability == 75 ? 1 : 0;
      auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                          numWarps, instrShape);
      mmaEnc = NvidiaMmaEncodingAttr::get(oldRetType.getContext(), versionMajor,
                                          versionMinor, warpsPerTile, CTALayout,
                                          instrShape);
    }
    PatternRewriterWithAsyncTaskIds taskIdRewriter(rewriter, dotOp);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);
    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc =
        rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

    Operation *newDot = nullptr;
    if (versionMajor == 3) {
      auto eltType = dotOp.getA().getType().getElementType();
      // In MMAV3 tranpose is only supported for f16 and bf16.
      bool allowTranspose = eltType.isF16() || eltType.isBF16();
      a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose);
      b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose);
      newDot = taskIdRewriter.create<triton::nvidia_gpu::WarpGroupDotOp>(
          dotOp.getLoc(), newRetType, a, b, newAcc, nullptr,
          dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc(), false);
    } else {
      // convert operands
      int minBitwidth =
          std::min(computeOrigBitWidth(a), computeOrigBitWidth(b));
      a = convertDotOperandForMMA(a, /*opIdx=*/0, minBitwidth, newRetType,
                                  rewriter);
      b = convertDotOperandForMMA(b, /*opIdx=*/1, minBitwidth, newRetType,
                                  rewriter);
      newDot = taskIdRewriter.create<DotOp>(dotOp.getLoc(), newRetType, a, b,
                                            newAcc, dotOp.getInputPrecision(),
                                            dotOp.getMaxNumImpreciseAcc());
    }
    // convert dot instruction
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType,
                                                 newDot->getResult(0));
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  auto tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  Type operandElType =
      cast<RankedTensorType>(operand.getType()).getElementType();
  if (operandElType == promotedType)
    return operand;
  if (isFp8Type(operandElType))
    return builder.create<FpToFpOp>(loc, tensorPromotedType, operand);
  // bf16 -> f16 is not an extf (same width) and the NVIDIA lowering tables have
  // no direct entry, so it goes through f32 like semantic.py routes every
  // bf16 <-> f16 cast. Widening conversions are plain extf: a `fp_to_fp` here
  // would have to name a rounding mode and the lowering rejects the pairs it
  // has no entry for (bf16 -> f32 among them).
  if (operandElType.isBF16() && promotedType.isF16()) {
    Type f32TensorType = cast<RankedTensorType>(operand.getType())
                             .cloneWith(std::nullopt, builder.getF32Type());
    Value asF32 = builder.create<arith::ExtFOp>(loc, f32TensorType, operand);
    return builder.create<FpToFpOp>(
        loc, tensorPromotedType, asF32,
        RoundingModeAttr::get(builder.getContext(), RoundingMode::RTNE));
  }
  return builder.create<arith::ExtFOp>(loc, tensorPromotedType, operand);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod, int computeCapability) {
  mod.walk([=](DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    NvidiaMmaEncodingAttr mmaLayout =
        dyn_cast<NvidiaMmaEncodingAttr>(D.getType().getEncoding());
    if (mmaLayout) {
      bool isNativeFP8 = AElType.isFloat8E5M2() || AElType.isFloat8E4M3FN();
      // promote operands for sm < 89 since fp8 mma is not natively supported
      // promote operands for sm >= 90 when mma is not v3
      if (!isNativeFP8 ||
          (isNativeFP8 && (computeCapability == 89 || mmaLayout.isHopper())))
        return;
      promoteType = builder.getF16Type();
    } else {
      // FMA case.
      Type AElType = dotOp.getA().getType().getElementType();
      Type DElType = D.getType().getElementType();
      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

class ScaledBlockedToMMAv2
    : public mlir::OpRewritePattern<triton::DotScaledOp> {
  int computeCapability;

public:
  ScaledBlockedToMMAv2(mlir::MLIRContext *context, int computeCapability)
      : mlir::OpRewritePattern<triton::DotScaledOp>(context),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotScaledOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability >= 100)
      return failure();

    auto oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();
    auto ctx = dotOp.getContext();

    // Check that rhs scale is null
    assert(dotOp.getRhsScale() == nullptr && "rhs scale NYI");

    // operands
    auto a = dotOp.getLhs();
    auto b = dotOp.getRhs();
    auto scale = dotOp.getLhsScale();
    auto aType = dotOp.getLhsType();
    auto bType = dotOp.getRhsType();

    assert((aType == ScaleDotElemType::E4M3 ||
            aType == ScaleDotElemType::E5M2 ||
            aType == ScaleDotElemType::E2M1) &&
           "NYI: lhs supports fp4 or fp8");
    assert(bType == ScaleDotElemType::E4M3 || bType == ScaleDotElemType::E5M2 ||
           bType == ScaleDotElemType::BF16 && "NYI: rhs supports fp8 and bf16");

    // TODO run accelerate matmul on A and B first to choose their layouts
    // Set return type
    auto versionMajor = 2;
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    auto mod = dotOp->getParentOfType<mlir::ModuleOp>();
    unsigned numWarps = TritonGPUDialect::getNumWarps(mod);
    auto instrShape = mmaVersionToInstrShape(versionMajor, retShapePerCTA,
                                             rewriter.getBF16Type(), numWarps);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());
    // TODO Use warpsPerTileV2
    SmallVector<unsigned> warpsPerCTA = {numWarps, 1};
    auto mmaEnc = NvidiaMmaEncodingAttr::get(ctx, /*versionMajor=*/versionMajor,
                                             /*versionMinor=*/0, warpsPerCTA,
                                             CTALayout, instrShape);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc =
        rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

    auto toMMABf16 =
        [&newRetType, &rewriter,
         &ctx](TypedValue<RankedTensorType> v, int idx,
               ScaleDotElemType type) -> TypedValue<RankedTensorType> {
      auto vType = v.getType();
      if (type == ScaleDotElemType::E2M1) {
        // A bit too dynamically typed...
        // perhaps return ints in both cases?

        auto retEnc = dyn_cast<NvidiaMmaEncodingAttr>(newRetType.getEncoding());
        auto newVEncoding = DotOperandEncodingAttr::get(
            ctx, idx, newRetType.getEncoding(), /*kWidth=*/4);
        auto newVType = RankedTensorType::get(
            vType.getShape(), vType.getElementType(), newVEncoding);
        return rewriter.create<ConvertLayoutOp>(v.getLoc(), newVType, v);
      } else {
        assert(type == ScaleDotElemType::E5M2 ||
               type == ScaleDotElemType::E4M3 ||
               type == ScaleDotElemType::BF16);
        auto newVEncoding = DotOperandEncodingAttr::get(
            ctx, idx, newRetType.getEncoding(), /*kWidth=*/8);
        auto newVType = RankedTensorType::get(
            vType.getShape(), vType.getElementType(), newVEncoding);
        v = rewriter.create<ConvertLayoutOp>(v.getLoc(), newVType, v);

        if (type == ScaleDotElemType::BF16) {
          return v;
        } else {
          // Convert to bf16
          auto vTypeBf16 = RankedTensorType::get(
              vType.getShape(), rewriter.getBF16Type(), newVEncoding);
          return rewriter.create<FpToFpOp>(v.getLoc(), vTypeBf16, v);
        }
      }
    };
    a = toMMABf16(a, 0, aType);
    b = toMMABf16(b, 1, bType);

    // [Note: A trick to avoid warp shuffles in the lowering]
    // FIXME: Implement this when we can set general layouts on a tensor

    // For bf16, we have 4 threads per row
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16816-a-f16
    // and each of them needs to get every scale in that row.
    // It turns out that the layout for the output of type bf16 gives us exactly
    // this layout when the number of mxfp vectors is equal to two (K = 64)
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16816-c
    // This can be generalised to other K with linear layouts, but the general
    // layout cannot cannot be represented with the predefined layouts :(
    // With this trick, we could do the full lowering here and remove the
    // UpcastMXFPOp altogether

    assert(instrShape == ArrayRef<unsigned>({16, 8}) ||
           instrShape == ArrayRef<unsigned>({1, 16, 8}));
    auto shapeTileA = std::array<unsigned, 2>{instrShape[0], instrShape[0]};
    // Necessary choice to leave all the scales of the tile in that given warp
    auto threadsPerWarp =
        SmallVector<unsigned>{shapeTileA[0], 32 / shapeTileA[0]};

    auto newScaleEncoding = triton::gpu::BlockedEncodingAttr::get(
        ctx, {1, 1}, threadsPerWarp, warpsPerCTA, {1, 0}, CTALayout);

    auto newScaleDotElemType = RankedTensorType::get(
        scale.getType().getShape(), scale.getType().getElementType(),
        newScaleEncoding);
    scale = rewriter.create<ConvertLayoutOp>(scale.getLoc(),
                                             newScaleDotElemType, scale);

    auto scaledA = rewriter.create<triton::gpu::UpcastMXFPOp>(
        dotOp.getLoc(), a, scale, dotOp.getLhsType());

    // convert dot instruction
    auto newDot =
        rewriter.create<DotOp>(dotOp.getLoc(), newRetType, scaledA, b, newAcc);
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType, newDot);
    return success();
  }
};

#define GEN_PASS_DEF_TRITONGPUACCELERATEMATMUL
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUAccelerateMatmulPass
    : public impl::TritonGPUAccelerateMatmulBase<
          TritonGPUAccelerateMatmulPass> {
public:
  using impl::TritonGPUAccelerateMatmulBase<
      TritonGPUAccelerateMatmulPass>::TritonGPUAccelerateMatmulBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    auto computeCapability = getNVIDIAComputeCapability(m);

    mlir::RewritePatternSet patterns(context);
    patterns.add<BlockedToMMA, ScaledBlockedToMMAv2>(context,
                                                     computeCapability);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    // Now that we have picked the mma type, decompose dot that are not natively
    // supported.
    decomposeMixedModeDotOp(m, computeCapability);
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
