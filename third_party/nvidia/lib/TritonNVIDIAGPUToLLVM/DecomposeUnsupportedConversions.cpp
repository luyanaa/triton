#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DECOMPOSEUNSUPPORTEDNVIDIACONVERSIONS
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

// Loading from Hopper shared memory layout to dot operand is not supported. We
// need to break it down and use a different shared layout. This would mostly
// happen when TMAs are used with MMAV2 and will cause poor performance.
class DecomposeLocalLoadToDotOperand
    : public OpRewritePattern<triton::gpu::LocalLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::LocalLoadOp op,
                                PatternRewriter &rewriter) const override {

    auto dstDotOp = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
        op.getType().getEncoding());
    MemDescType srcType = op.getSrc().getType();
    auto sharedEncoding = cast<SharedEncodingAttr>(srcType.getEncoding());
    if (!dstDotOp || !sharedEncoding.getHasLeadingOffset())
      return failure();
    RankedTensorType type = op.getType();
    auto parentEnc = dstDotOp.getParent();
    int numWarps = triton::gpu::getNumWarpsPerCTA(parentEnc);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    int numCTAs = triton::gpu::getNumCTAs(parentEnc);
    auto blockEncoding = getDefaultBlockedEncoding(
        op.getContext(), type.getShape(), numWarps, threadsPerWarp, numCTAs);
    auto tmpType = RankedTensorType::get(type.getShape(), type.getElementType(),
                                         blockEncoding);
    Value load =
        rewriter.create<LocalLoadOp>(op.getLoc(), tmpType, op.getSrc());
    auto newSharedDescTy = triton::MemDescType::get(
        type.getShape(), type.getElementType(),
        triton::gpu::SharedEncodingAttr::get(
            op.getContext(), dstDotOp, type.getShape(),
            triton::gpu::getOrder(parentEnc),
            triton::gpu::getCTALayout(parentEnc), type.getElementType()),
        srcType.getMemorySpace());
    auto tmp = rewriter.create<triton::gpu::LocalAllocOp>(
        op.getLoc(), newSharedDescTy, load);
    auto newConvert =
        rewriter.create<triton::gpu::LocalLoadOp>(op.getLoc(), type, tmp);
    rewriter.replaceOp(op, newConvert);
    return success();
  }
};

struct DecomposeUnsupportedConversions
    : public mlir::triton::impl::DecomposeUnsupportedNVIDIAConversionsBase<
          DecomposeUnsupportedConversions> {
  void runOnOperation() override {
    // FIXME [Dot LL]
    // Remove the decomposeTensorCoreToDotLayoutConversion class entirely after
    // we have enabled the new layout conversion for all the cases.
    auto nvidiaShortCutFn = [&](RankedTensorType srcTy,
                                RankedTensorType dstTy) {
      auto nvidiaMma = dyn_cast<NvidiaMmaEncodingAttr>(srcTy.getEncoding());
      // Turing reports isAmpere() (versionMajor == 2) but does not share
      // Ampere's operand packing. The f16 fragments do (both targets use
      // m16n8k8), so a register relabel is exact there. The s8/s4 ones do not:
      // Turing's m8n8k16 / m8n8k32 read 4 / 8 K-adjacent elements out of each
      // 32-bit word where Ampere's m16n8k32 reads a different set, and the
      // dot-operand layout is shared between the two targets (kWidth =
      // 32/bitwidth), so nothing upstream objects to the relabel. It then
      // mispacks the operand, which has to go through shared memory instead,
      // where SharedToDotOperandMMAv2 performs the Turing packing. Every other
      // Turing value that carries an MMA layout on its way to a dot operand is
      // a 32-bit accumulator holding packed s8/s4, so f16 is the only element
      // type that may take the shortcut.
      if (nvidiaMma && nvidiaMma.isTuring() && !srcTy.getElementType().isF16())
        return false;
      // Supported mma to dot conversion
      if (nvidiaMma && nvidiaMma.isAmpere())
        return true;
      // No need to decompose if shared memory is not needed
      return matchMmaV3AndDotOperandLayout(srcTy, dstTy) ||
             cvtReordersRegisters(srcTy, dstTy);
    };
    ModuleOp mod = getOperation();
    triton::gpu::decomposeSplatOpToSharedLayoutConversion(mod);
    triton::gpu::decomposeTensorCoreToDotLayoutConversion(mod,
                                                          nvidiaShortCutFn);
    triton::gpu::decomposeBlockedToDotLayoutConversion(mod);

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeLocalLoadToDotOperand>(&getContext());
    if (mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::triton::NVIDIA {

std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedConversionsPass() {
  return std::make_unique<DecomposeUnsupportedConversions>();
}

} // namespace mlir::triton::NVIDIA
