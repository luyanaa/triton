#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <deque>
#include <optional>

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#define DEBUG_TYPE "tritonamdgpu-convert-buffer-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {
bool verifyNonNegativeByAssumption(Value expr,
                                   const DenseSet<Value> &assumptions) {
  for (Value assume : assumptions) {
    LDBG("Assumption:" << assume);
    if (auto cmpOp = assume.getDefiningOp<arith::CmpIOp>()) {
      bool isGreaterThan = (cmpOp.getPredicate() == arith::CmpIPredicate::sge ||
                            cmpOp.getPredicate() == arith::CmpIPredicate::sgt);
      APInt cst;
      if (isGreaterThan && (cmpOp.getLhs() == expr) &&
          matchPattern(cmpOp.getRhs(), m_ConstantInt(&cst))) {
        return cst.isNonNegative();
      }
    }
  }
  return false;
}

bool verifyNonNegativeExpr(Value expr, const DenseSet<Value> &assumptions) {

  // Check if the expression is contained in any assumption
  if (verifyNonNegativeByAssumption(expr, assumptions)) {
    LDBG("Non negative by assumption");
    return true;
  }

  // Recurse if the operation is defined
  Operation *op = expr.getDefiningOp();
  if (!op)
    return false;

  bool nonNegative =
      llvm::TypeSwitch<Operation *, bool>(expr.getDefiningOp())
          .Case<triton::BroadcastOp>([&](auto broadcastOp) {
            return verifyNonNegativeExpr(broadcastOp.getSrc(), assumptions);
          })
          .Case<triton::ExpandDimsOp>([&](auto expandOp) {
            return verifyNonNegativeExpr(expandOp.getSrc(), assumptions);
          })
          .Case<triton::SplatOp>([&](auto splatOp) {
            return verifyNonNegativeExpr(splatOp.getSrc(), assumptions);
          })
          .Case<triton::MakeRangeOp>([&](auto makeRangeOp) {
            return makeRangeOp.getStart() >= 0 && makeRangeOp.getEnd() >= 0;
          })
          .Case<arith::ConstantIntOp>(
              [&](auto constIntOp) { return constIntOp.value() >= 0; })
          .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
            Value val = constOp.getResult();
            DenseIntElementsAttr constVal;
            if (matchPattern(val, m_Constant(&constVal)) && constVal.isSplat())
              return constVal.getSplatValue<APInt>().isNonNegative();
            return false;
          })
          .Case<triton::GetProgramIdOp>([&](auto pidOp) { return true; })
          .Case<arith::MaxSIOp>([&](auto maxOp) {
            // max(a,b) >= 0 iff a>=0 || b>=0
            bool nnLhs = verifyNonNegativeExpr(maxOp.getLhs(), assumptions);
            bool nnRhs = verifyNonNegativeExpr(maxOp.getRhs(), assumptions);
            return nnLhs || nnRhs;
          })
          .Case<arith::RemSIOp>([&](auto remsiOp) {
            // a % b >= 0 iff a>=0
            return verifyNonNegativeExpr(remsiOp.getLhs(), assumptions);
          })
          .Case<arith::TruncIOp, arith::ExtSIOp>([&](Operation *unaryOp) {
            // a = OP b >= 0 iff b >= 0
            return verifyNonNegativeExpr(unaryOp->getOperand(0), assumptions);
          })
          .Case<arith::AddIOp, arith::MinSIOp, arith::MulIOp, arith::DivSIOp>(
              // Generally speaking, a OP b >= 0  iff  a >= 0 && b >= 0 when
              // OP != sub
              [&](Operation *binOp) {
                bool nnLhs =
                    verifyNonNegativeExpr(binOp->getOperand(0), assumptions);
                bool nnRhs =
                    verifyNonNegativeExpr(binOp->getOperand(1), assumptions);
                return nnLhs && nnRhs;
              })
          .Default([&](Operation *op) {
            // Conservatively assume that the expression is negative
            return false;
          });
  return nonNegative;
}

// Quick analysis on the Triton IR to decide if we can safely use
// buffer operations
bool canUseBufferOps(Value ptr, const DenseSet<Value> &assumptions) {
  // 1. Check if the pointer is uniform: i.e., if it comes from a uniform
  // pointer(splatted) and non-uniform offset addition

  LDBG("Buffer op checks for: " << ptr);
  auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp)
    return false;

  auto maybeSplatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
  if (!maybeSplatOp)
    return false;
  LDBG("Pattern matched");

  // 2. Check if the offset is a 32-bit tensor
  Value offset = addPtrOp.getOffset();
  if (cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth() != 32)
    return false;
  LDBG("32 bit offset");

  // 3. Check if the offset is non-negative
  if (!verifyNonNegativeExpr(offset, assumptions))
    return false;

  LDBG("Non-negative");
  return true;
}
} // namespace

static Value getBlockStride(Value offset) {
  if (auto add = offset.getDefiningOp<arith::AddIOp>()) {
    for (Value operand : add.getOperands()) {
      if (auto broadcast = operand.getDefiningOp<triton::BroadcastOp>()) {
        if (auto mul = broadcast.getSrc().getDefiningOp<arith::MulIOp>()) {
          for (Value mulOperand : mul.getOperands())
            if (auto splat = mulOperand.getDefiningOp<triton::SplatOp>())
              return splat.getSrc();
        }
      }
    }
  }
  return {};
}

static unsigned getBufferVectorSize(Value ptr, Value offset,
                                    tt::ModuleAxisInfoAnalysis &axisInfo) {
  auto offsetTy = cast<RankedTensorType>(offset.getType());
  auto ptrTy = offsetTy.cloneWith(std::nullopt, ptr.getType());
  auto order = ttg::getOrder(ptrTy.getEncoding());
  auto uniqueContig =
      ttg::getUniqueContigPerThread(ptrTy.getEncoding(), ptrTy.getShape());
  unsigned contiguity = uniqueContig[order[0]];
  auto *ptrAxisInfo = axisInfo.getAxisInfo(ptr);
  unsigned elementBytes =
      std::max<unsigned>(tt::getPointeeBitWidth(ptr.getType()) / 8, 1);
  unsigned alignment =
      std::max<int64_t>(ptrAxisInfo->getDivisibility(0) / elementBytes, 1);
  contiguity = std::min<unsigned>(alignment, contiguity);
  return std::min<unsigned>(128 / tt::getPointeeBitWidth(ptr.getType()),
                            contiguity);
}

struct ConvertStreamLoadToBufferLoad
    : public mlir::OpRewritePattern<triton::gpu::LocalStoreOp> {
  ConvertStreamLoadToBufferLoad(MLIRContext *context,
                                DenseSet<Value> &assumptions,
                                tt::ModuleAxisInfoAnalysis &axisInfo,
                                bool enabled)
      : OpRewritePattern(context), assumptions(assumptions), axisInfo(axisInfo),
        enabled(enabled) {}

  LogicalResult matchAndRewrite(triton::gpu::LocalStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (!enabled)
      return failure();
    auto load = op.getSrc().getDefiningOp<triton::LoadOp>();
    if (!load || !load->getResult(0).hasOneUse() ||
        !load.getBoundaryCheck().empty() || load.getPaddingAttr() ||
        load.getEvict() != triton::EvictionPolicy::NORMAL ||
        load.getIsVolatile() || !canUseBufferOps(load.getPtr(), assumptions))
      return failure();
    auto addPtr = load.getPtr().getDefiningOp<triton::AddPtrOp>();
    auto splat = addPtr.getPtr().getDefiningOp<triton::SplatOp>();
    if (!splat)
      return failure();

    unsigned vec =
        getBufferVectorSize(splat.getSrc(), addPtr.getOffset(), axisInfo);
    if (load.getMask())
      vec = std::min(vec, axisInfo.getMaskAlignment(load.getMask()));
    auto srcTy = cast<RankedTensorType>(load.getPtr().getType());
    auto dstTy = op.getDst().getType();
    if (!LLVM::AMD::canCoalesceWriteIntoSharedMemory(srcTy, dstTy, vec))
      return failure();
    int loadBits = vec * tt::getPointeeBitWidth(srcTy);
    if (!llvm::is_contained({8, 16, 32}, loadBits))
      return failure();

    auto bufferLoad = rewriter.create<triton::amdgpu::BufferLoadToLocalOp>(
        load.getLoc(), ttg::AsyncTokenType::get(load.getContext()), op.getDst(),
        splat.getSrc(), addPtr.getOffset(), load.getMask(), load.getOther(),
        getBlockStride(addPtr.getOffset()), load.getCache());
    if (auto opIdxAttr = load->getAttrOfType<triton::amdgpu::OpIdxAttr>(
            triton::amdgpu::OpIdxAttr::getMnemonic()))
      bufferLoad->setAttr(triton::amdgpu::OpIdxAttr::getMnemonic(), opIdxAttr);
    rewriter.create<triton::gpu::AsyncWaitOp>(load.getLoc(),
                                              bufferLoad.getToken(), 0);
    rewriter.eraseOp(op);
    if (load->use_empty())
      rewriter.eraseOp(load);
    return success();
  }

private:
  DenseSet<Value> &assumptions;
  tt::ModuleAxisInfoAnalysis &axisInfo;
  bool enabled;
};

struct ConvertAsyncCopyToBufferLoad
    : public mlir::OpRewritePattern<triton::gpu::AsyncCopyGlobalToLocalOp> {
  ConvertAsyncCopyToBufferLoad(MLIRContext *context,
                               DenseSet<Value> &assumptions,
                               tt::ModuleAxisInfoAnalysis &axisInfo,
                               bool enabled)
      : OpRewritePattern(context), assumptions(assumptions), axisInfo(axisInfo),
        enabled(enabled) {}

  LogicalResult matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op,
                                PatternRewriter &rewriter) const override {
    if (!enabled || op.getEvict() != triton::EvictionPolicy::NORMAL ||
        op.getIsVolatile() || !canUseBufferOps(op.getSrc(), assumptions))
      return failure();
    auto addPtr = op.getSrc().getDefiningOp<triton::AddPtrOp>();
    auto splat = addPtr.getPtr().getDefiningOp<triton::SplatOp>();
    if (!splat)
      return failure();
    unsigned vec =
        getBufferVectorSize(splat.getSrc(), addPtr.getOffset(), axisInfo);
    if (op.getMask())
      vec = std::min(vec, axisInfo.getMaskAlignment(op.getMask()));
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = op.getResult().getType();
    if (!LLVM::AMD::canCoalesceWriteIntoSharedMemory(srcTy, dstTy, vec))
      return failure();
    int loadBits = vec * tt::getPointeeBitWidth(srcTy);
    if (!llvm::is_contained({8, 16, 32}, loadBits))
      return failure();
    Value maybeMask = op.getMask();
    Value maybeOther = op.getOther();
    Value blockStride = getBlockStride(addPtr.getOffset());
    auto bufferLoadOp = rewriter.create<triton::amdgpu::BufferLoadToLocalOp>(
        op.getLoc(), op.getType(), op.getResult(), splat.getSrc(),
        addPtr.getOffset(), maybeMask, maybeOther, blockStride, op.getCache());
    if (auto opIdxAttr = op->getAttrOfType<triton::amdgpu::OpIdxAttr>(
            triton::amdgpu::OpIdxAttr::getMnemonic()))
      bufferLoadOp->setAttr(triton::amdgpu::OpIdxAttr::getMnemonic(),
                            opIdxAttr);
    if (op.getToken().use_empty())
      rewriter.create<triton::gpu::AsyncWaitOp>(op.getLoc(),
                                                bufferLoadOp.getToken(), 0);
    rewriter.replaceOp(op, bufferLoadOp);
    return success();
  }

private:
  DenseSet<Value> &assumptions;
  tt::ModuleAxisInfoAnalysis &axisInfo;
  bool enabled;
};

struct ConvertTritonLoadToBufferLoad
    : public mlir::OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonLoadToBufferLoad(mlir::MLIRContext *context,
                                DenseSet<Value> &assumptions)
      : mlir::OpRewritePattern<triton::LoadOp>(context),
        assumptions(assumptions) {}

  mlir::LogicalResult
  matchAndRewrite(triton::LoadOp op, PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (op.getCache() != triton::CacheModifier::NONE)
      return failure();
    if (op->getResult(0).hasOneUse() &&
        isa<triton::gpu::LocalStoreOp>(*op->getResult(0).user_begin()))
      return failure();

    if (canUseBufferOps(ptr, assumptions)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeOther = op.getOther();
      Value maybeMask = op.getMask();

      auto bufferLoadOp = rewriter.create<triton::amdgpu::BufferLoadOp>(
          op->getLoc(), op.getType(), basePtr, tensorOffset, maybeMask,
          maybeOther);

      // Propagate `OpIdxAttr` if the currently processed `tt.LoadOp` was
      // labeled it. The attribute needs to be preserved for custom instruction
      // scheduling.
      if (auto opIdxAttr = op->getAttrOfType<triton::amdgpu::OpIdxAttr>(
              triton::amdgpu::OpIdxAttr::getMnemonic())) {
        bufferLoadOp->setAttr(triton::amdgpu::OpIdxAttr::getMnemonic(),
                              opIdxAttr);
      }
      rewriter.replaceOp(op, bufferLoadOp);

      return success();
    }
    LDBG("Failed to convert: " << op);
    return failure();
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
};

struct ConvertTritonStoreToBufferStore
    : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonStoreToBufferStore(mlir::MLIRContext *context,
                                  DenseSet<Value> &assumptions)
      : mlir::OpRewritePattern<triton::StoreOp>(context),
        assumptions(assumptions) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (op.getCache() != triton::CacheModifier::NONE)
      return failure();

    if (canUseBufferOps(ptr, assumptions)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeMask = op.getMask();
      rewriter.replaceOpWithNewOp<triton::amdgpu::BufferStoreOp>(
          op, op.getValue(), basePtr, tensorOffset, maybeMask);
      return success();
    }
    LDBG("Failed to convert: " << op);
    return failure();
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
};

class TritonAMDGPUConvertToBufferOpsPass
    : public TritonAMDGPUConvertToBufferOpsBase<
          TritonAMDGPUConvertToBufferOpsPass> {

public:
  TritonAMDGPUConvertToBufferOpsPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();
    // Collect assumptions in the function
    DenseSet<Value> assumptions;
    m.walk([&](LLVM::AssumeOp op) {
      if (op->getOperand(0).getDefiningOp<arith::CmpIOp>())
        assumptions.insert(op->getOperand(0));
    });
    LDBG("Number of assumptions found: " << assumptions.size());

    bool enableBufferLoadToLocal = false;
    bool enableGlobalLoadLDS = false;
    if (auto targetAttr =
            m->getAttrOfType<StringAttr>(triton::AttrTargetName)) {
      StringRef arch = targetAttr.getValue();
      arch.consume_front("hip:");
      auto family = triton::AMD::deduceISAFamily(arch);
      enableBufferLoadToLocal = llvm::is_contained(
          {triton::AMD::ISAFamily::GCN5, triton::AMD::ISAFamily::VEGA20,
           triton::AMD::ISAFamily::CDNA1, triton::AMD::ISAFamily::CDNA2,
           triton::AMD::ISAFamily::CDNA3},
          family);
      enableGlobalLoadLDS = llvm::is_contained({triton::AMD::ISAFamily::CDNA1,
                                                triton::AMD::ISAFamily::CDNA2,
                                                triton::AMD::ISAFamily::CDNA3},
                                               family);
    }
    tt::ModuleAxisInfoAnalysis axisInfo(m);
    patterns.add<ConvertAsyncCopyToBufferLoad>(context, assumptions, axisInfo,
                                               enableGlobalLoadLDS);
    patterns.add<ConvertStreamLoadToBufferLoad>(context, assumptions, axisInfo,
                                                enableBufferLoadToLocal);
    patterns.add<ConvertTritonLoadToBufferLoad>(context, assumptions);
    patterns.add<ConvertTritonStoreToBufferStore>(context, assumptions);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUConvertToBufferOpsPass() {
  return std::make_unique<TritonAMDGPUConvertToBufferOpsPass>();
}
