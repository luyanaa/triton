#include "SchedInstructions.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUINSERTINSTRUCTIONSCHEDHINTS
#define GEN_PASS_DEF_TRITONAMDGPULOWERINSTRUCTIONSCHEDHINTS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

#undef DEBUG_TYPE
#define DEBUG_TYPE "lower-insert-instruction-sched-hints"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

// TODO: The following passes/algorithms are applicable only for a single
// `tt.dot` op in a `scf.for` block -i.e., a single schedule hint op per block.
// Note, we need to relax this assumption in the future and extend the current
// implementation.

namespace mlir::triton {

template <typename OpType, typename Callback>
static void walkSchedHintsInSameLoop(OpType op, Callback &&callback) {
  auto loopOp = op->template getParentOfType<scf::ForOp>();
  if (loopOp) {
    loopOp->walk([&](triton::amdgpu::InstructionSchedHint schedHint) {
      if (schedHint->getParentOfType<scf::ForOp>() == loopOp)
        callback(schedHint);
    });
    return;
  }

  if (auto loopId = op->template getAttrOfType<IntegerAttr>(
          triton::amdgpu::InstructionSchedLoopIdAttrName)) {
    if (auto moduleOp = op->template getParentOfType<ModuleOp>()) {
      moduleOp->walk([&](triton::amdgpu::InstructionSchedHint schedHint) {
        auto hintLoopId = schedHint->getAttrOfType<IntegerAttr>(
            triton::amdgpu::InstructionSchedLoopIdAttrName);
        if (hintLoopId == loopId)
          callback(schedHint);
      });
      return;
    }
  }

  // Some pipelines lower scf.for before converting AMD operations. In that
  // case, use the loop ID attached before lowering, or fall back to same-block
  // hints when no ID is available.
  for (Operation &blockOp : *op->getBlock()) {
    if (auto schedHint =
            dyn_cast<triton::amdgpu::InstructionSchedHint>(&blockOp))
      callback(schedHint);
  }
}
void setNumGeneratedMMAs(DotOp op, size_t mmaCount, unsigned m, unsigned n,
                         unsigned k, Type elementType) {
  auto *ctx = op->getContext();
  auto mmaType = RankedTensorType::get({m, n, k}, elementType);
  auto counterAttr =
      triton::amdgpu::InstCounterAttr::get(ctx, mmaCount, mmaType);

  walkSchedHintsInSameLoop(op,
                           [&](triton::amdgpu::InstructionSchedHint schedHint) {
                             schedHint.setNumMMAsAttr(counterAttr);
                           });
}

template <typename LoadOpType>
void setNumGeneratedGlobalLoads(LoadOpType op, size_t globalLoadsCount,
                                Type type) {
  MLIRContext *ctx = op->getContext();
  auto counterAttr =
      triton::amdgpu::InstCounterAttr::get(ctx, globalLoadsCount, type);

  walkSchedHintsInSameLoop(
      op, [&](triton::amdgpu::InstructionSchedHint schedHint) {
        if (auto opIdxAttr =
                op->template getAttrOfType<triton::amdgpu::OpIdxAttr>(
                    triton::amdgpu::OpIdxAttr::getMnemonic())) {
          const bool isBufferLoadOp =
              std::is_same_v<LoadOpType, triton::amdgpu::BufferLoadOp> ||
              std::is_same_v<LoadOpType, triton::amdgpu::BufferLoadToLocalOp>;
          if (opIdxAttr.getValue() == 0) {
            schedHint.setNumGlobalLoadsAAttr(counterAttr);
            schedHint.setIsBufferLoadsAEnabled(isBufferLoadOp);
          } else {
            schedHint.setNumGlobalLoadsBAttr(counterAttr);
            schedHint.setIsBufferLoadsBEnabled(isBufferLoadOp);
          }
        }
      });
}
template void setNumGeneratedGlobalLoads(triton::amdgpu::BufferLoadOp op,
                                         size_t globalLoadsCount, Type type);
template void setNumGeneratedGlobalLoads(triton::amdgpu::BufferLoadToLocalOp op,
                                         size_t globalLoadsCount, Type type);
template void setNumGeneratedGlobalLoads(triton::LoadOp op,
                                         size_t globalLoadsCount, Type type);

void setNumGeneratedDsReads(gpu::LocalLoadOp op, size_t dsReadsCount,
                            Type type) {
  auto *ctx = op->getContext();
  auto counterAttr =
      triton::amdgpu::InstCounterAttr::get(ctx, dsReadsCount, type);

  walkSchedHintsInSameLoop(
      op, [&](triton::amdgpu::InstructionSchedHint schedHint) {
        Value dst = op.getResult();
        auto dstTensorTy = cast<RankedTensorType>(dst.getType());
        auto dotOperandLayout =
            cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());
        const size_t opIdx = dotOperandLayout.getOpIdx();
        assert(opIdx < 2);
        if (opIdx == 0)
          schedHint.setNumDsReadsAAttr(counterAttr);
        else
          schedHint.setNumDsReadsBAttr(counterAttr);
      });
}

void storeOpConversionCallback(triton::gpu::LocalStoreOp op,
                               size_t localStoreOpCount, Type type) {
  MLIRContext *ctx = op->getContext();
  auto counterAttr =
      triton::amdgpu::InstCounterAttr::get(ctx, localStoreOpCount, type);

  walkSchedHintsInSameLoop(
      op, [&](triton::amdgpu::InstructionSchedHint schedHint) {
        if (auto opIdxAttr = op->getAttrOfType<triton::amdgpu::OpIdxAttr>(
                triton::amdgpu::OpIdxAttr::getMnemonic())) {
          assert(opIdxAttr.getValue() < 2);
          if (opIdxAttr.getValue() == 0)
            schedHint.setNumDsWritesAAttr(counterAttr);
          else
            schedHint.setNumDsWritesBAttr(counterAttr);
        }
      });
}

triton::DotOp getSingleDotOpIfExists(scf::ForOp forOp) {
  triton::DotOp dotOp = nullptr;
  size_t dotCounter = 0;
  forOp->walk([&](triton::DotOp op) {
    // Each loop is visited separately, so ignore dots owned by nested loops.
    if (op->getParentOfType<scf::ForOp>() != forOp)
      return;
    dotOp = op;
    ++dotCounter;
  });

  return (dotCounter == 1) ? dotOp : nullptr;
}
} // namespace mlir::triton

namespace {
bool isGfx942(ModuleOp moduleOp) {
  auto target = moduleOp->getAttrOfType<StringAttr>("triton_gpu.target");
  if (!target)
    return false;

  StringRef arch = target.getValue();
  if (arch.starts_with("hip:"))
    arch = arch.drop_front(4);
  return arch == "gfx942";
}

// Create an intrinsic to control how different instruction kinds should
// interleave for better ILP.
void createSchedGroupBarrier(PatternRewriter &rewriter, Location loc,
                             mlir::amdgpu::sched_barrier_opt_enum maskValue,
                             int sizeValue, int groupIdValue) {
  IntegerAttr mask =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(maskValue));
  IntegerAttr size =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(sizeValue));
  IntegerAttr groupId =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(groupIdValue));
  rewriter.create<ROCDL::SchedGroupBarrier>(loc, mask, size, groupId);
}

// Insert intrinsic that controls the types of instructions that may be
// allowed to cross the intrinsic during instruction scheduling.
Operation *createSchedBarrier(PatternRewriter &rewriter, Location loc,
                              mlir::amdgpu::sched_barrier_opt_enum maskValue) {
  IntegerAttr mask =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(maskValue));
  return rewriter.create<ROCDL::SchedBarrier>(loc, mask);
}

// Insert an experimental intrinsic for instruction group level parallelism.
// The intrinsic takes a value that specifies the strategy.
Operation *createIglpOpt(PatternRewriter &rewriter, Location loc, int value) {
  IntegerAttr iglpValue =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(value));
  return rewriter.create<ROCDL::IglpOpt>(loc, iglpValue);
}

struct InstructionSchedHintsRewriter
    : public OpRewritePattern<triton::amdgpu::InstructionSchedHint> {

  InstructionSchedHintsRewriter(MLIRContext *ctx, int32_t numStages,
                                std::string variant)
      : OpRewritePattern(ctx), numStages(numStages) {
    std::transform(variant.begin(), variant.end(), variant.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    this->schedulingType = llvm::StringSwitch<SchedulingType>(variant)
                               .Case("default", SchedulingType::NONE)
                               .Case("iglp0", SchedulingType::IGLP0)
                               .Case("iglp1", SchedulingType::IGLP1)
                               .Case("ck_v3", SchedulingType::CK_V3)
                               .Default(SchedulingType::UNKNOWN);

    if (this->numStages < 2 && this->schedulingType == SchedulingType::CK_V3) {
      this->schedulingType = SchedulingType::NONE;
      LDBG("ignoring CK_V3 instruction scheduling because num_stages < 2");
    }
  }

  enum class SchedulingType : uint32_t {
    NONE = 0,
    IGLP0,
    IGLP1,
    CK_V3,
    UNKNOWN
  };

  // This is the implementation of the CK's V3 pipelining (see
  // see ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp).
  // This scheduling requires 1x register and 1x LDS buffers combined with the
  // local (LDS to registers) and global (HBM to registers) data prefetching.
  // see:
  // include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.h
  LogicalResult
  createCKV3Schedule(PatternRewriter &rewriter, Location loc,
                     triton::amdgpu::InstructionSchedHint schedHint) const {

    const uint32_t numDsReadInstA = schedHint.getNumDsReadsA().getValue();
    const uint32_t numDsReadInstB = schedHint.getNumDsReadsB().getValue();

    const uint32_t numDsWriteInstA = schedHint.getNumDsWritesA().getValue();
    const uint32_t numDsWriteInstB = schedHint.getNumDsWritesB().getValue();

    const uint32_t numBufferLoadInstA =
        schedHint.getNumGlobalLoadsA().getValue();
    const uint32_t numBufferLoadInstB =
        schedHint.getNumGlobalLoadsB().getValue();

    if (numBufferLoadInstA == 0 || numBufferLoadInstB == 0) {
      schedHint.emitError("buffer load counts for tiles A and B must be "
                          "nonzero");
      return failure();
    }

    const uint32_t numMfmaInst = schedHint.getNumMMAs().getValue();

    auto mfmaType =
        dyn_cast<RankedTensorType>(schedHint.getNumMMAs().getType());
    auto dsReadsAType =
        dyn_cast<VectorType>(schedHint.getNumDsReadsA().getType());
    auto dsReadsBType =
        dyn_cast<VectorType>(schedHint.getNumDsReadsB().getType());
    if (!mfmaType || mfmaType.getRank() != 3 ||
        (numDsReadInstA > 0 &&
         (!dsReadsAType || dsReadsAType.getRank() != 1)) ||
        (numDsReadInstB > 0 &&
         (!dsReadsBType || dsReadsBType.getRank() != 1))) {
      schedHint.emitError("CK_V3 scheduling requires initialized MFMA and "
                          "DS-read type counters");
      return failure();
    }
    const uint32_t nPerXDL = mfmaType.getShape()[1];
    const uint32_t mfmaCycle = nPerXDL == 16 ? 16 : 32;
    const uint32_t dsReadAIssueCycle =
        dsReadsAType && dsReadsAType.getShape()[0] == 16 ? 8 : 4;
    const uint32_t dsReadBIssueCycle =
        dsReadsBType && dsReadsBType.getShape()[0] == 16 ? 8 : 4;

    const auto dsReadAMfmaRate =
        (mfmaCycle - 4 + 2 * dsReadAIssueCycle - 1) / (2 * dsReadAIssueCycle);
    const auto dsReadBMfmaRate =
        (mfmaCycle - 4 + 2 * dsReadBIssueCycle - 1) / (2 * dsReadBIssueCycle);

    const auto numDsreadAMfma =
        (numDsReadInstA + dsReadAMfmaRate - 1) / dsReadAMfmaRate;
    const auto numDsreadBMfma =
        (numDsReadInstB + dsReadBMfmaRate - 1) / dsReadBMfmaRate;

    const uint64_t numDsreadMfma = numDsreadAMfma + numDsreadBMfma;
    if (numDsreadMfma > numMfmaInst) {
      schedHint.emitError("DS-read schedule exceeds the generated MFMA count");
      return failure();
    }
    const auto numMfmaStage1 = numMfmaInst - numDsreadMfma;
    const uint64_t numBufferLoads = numBufferLoadInstA + numBufferLoadInstB;
    if (numMfmaStage1 % numBufferLoads != 0 ||
        numDsWriteInstA % numBufferLoadInstA != 0 ||
        numDsWriteInstB % numBufferLoadInstB != 0) {
      schedHint.emitError("CK_V3 schedule requires evenly distributed loads, "
                          "DS writes, and MFMAs");
      return failure();
    }
    const auto numMfmaPerIssue = numMfmaStage1 / numBufferLoads;
    const auto numDswritePerIssueA = numDsWriteInstA / numBufferLoadInstA;
    const auto numDswritePerIssueB = numDsWriteInstB / numBufferLoadInstB;
    if (numDswritePerIssueA > numMfmaPerIssue ||
        numDswritePerIssueB > numMfmaPerIssue) {
      schedHint.emitError("DS-write schedule exceeds the MFMA issue count");
      return failure();
    }

    for (size_t i = 0; i < numBufferLoadInstA; ++i) {
      for (size_t idswrite = 0; idswrite < numDswritePerIssueA; ++idswrite) {
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::ds_write,
                                1, 0);
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma,
                                1, 0);
      }
      if (numMfmaPerIssue > numDswritePerIssueA)
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma,
                                numMfmaPerIssue - numDswritePerIssueA, 0);
    }

    for (size_t i = 0; i < numBufferLoadInstB; ++i) {
      for (size_t idswrite = 0; idswrite < numDswritePerIssueB; ++idswrite) {
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::ds_write,
                                1, 0);
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma,
                                1, 0);
      }
      if (numMfmaPerIssue > numDswritePerIssueB)
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma,
                                numMfmaPerIssue - numDswritePerIssueB, 0);
    }

    // stage 2
    uint32_t numDsReadsRemainingA = numDsReadInstA;
    for (size_t i = 0; i < numDsreadAMfma; ++i) {
      const uint32_t numDsReadsInGroup = numDsReadsRemainingA < dsReadAMfmaRate
                                             ? numDsReadsRemainingA
                                             : dsReadAMfmaRate;
      createSchedGroupBarrier(rewriter, loc,
                              mlir::amdgpu::sched_barrier_opt_enum::ds_read,
                              numDsReadsInGroup, 0);
      numDsReadsRemainingA -= numDsReadsInGroup;
      createSchedGroupBarrier(
          rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma, 1, 0);
    }

    uint32_t numDsReadsRemainingB = numDsReadInstB;
    for (size_t i = 0; i < numDsreadBMfma; ++i) {
      const uint32_t numDsReadsInGroup = numDsReadsRemainingB < dsReadBMfmaRate
                                             ? numDsReadsRemainingB
                                             : dsReadBMfmaRate;
      createSchedGroupBarrier(rewriter, loc,
                              mlir::amdgpu::sched_barrier_opt_enum::ds_read,
                              numDsReadsInGroup, 0);
      numDsReadsRemainingB -= numDsReadsInGroup;
      createSchedGroupBarrier(
          rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma, 1, 0);
    }
    return success();
  }

  LogicalResult
  matchAndRewrite(triton::amdgpu::InstructionSchedHint instructionSchedHint,
                  PatternRewriter &rewriter) const override {

    if (this->schedulingType == SchedulingType::UNKNOWN) {
      instructionSchedHint.emitError(
          "unknown instruction scheduling variant has been provided");
      return failure();
    }

    if (schedulingType == SchedulingType::CK_V3) {
      auto moduleOp = instructionSchedHint->getParentOfType<ModuleOp>();
      if (!moduleOp || !isGfx942(moduleOp)) {
        LDBG("Skipping `ck_v3` scheduling outside gfx942.");
        rewriter.eraseOp(instructionSchedHint);
        return success();
      }
      if (!(instructionSchedHint.getIsBufferLoadsAEnabled() &&
            instructionSchedHint.getIsBufferLoadsBEnabled())) {
        LDBG("Skipping instruction scheduling because `ck_v3` "
             "scheduling can be used only with `buffer_load` instructions.");
        rewriter.eraseOp(instructionSchedHint);
        return success();
      }
    }

    // Default and CK_V3 variants bound the block; IGLP uses only its option.
    const bool limitSchedulingRange =
        !(schedulingType == SchedulingType::IGLP0 ||
          schedulingType == SchedulingType::IGLP1);
    Location loc = instructionSchedHint->getLoc();
    Block *block = instructionSchedHint->getBlock();
    if (schedulingType == SchedulingType::CK_V3) {
      rewriter.setInsertionPoint(block->getTerminator());
      if (failed(createCKV3Schedule(rewriter, loc, instructionSchedHint)))
        return failure();
    }
    if (limitSchedulingRange) {
      rewriter.setInsertionPointToStart(block);
      createSchedBarrier(rewriter, loc,
                         mlir::amdgpu::sched_barrier_opt_enum::none);
    }

    rewriter.setInsertionPoint(block->getTerminator());
    switch (schedulingType) {
    case SchedulingType::IGLP0:
      [[fallthrough]];
    case SchedulingType::IGLP1: {
      createIglpOpt(rewriter, loc, static_cast<int>(schedulingType) - 1);
      break;
    }
    case SchedulingType::CK_V3:
      break;
    case SchedulingType::NONE:
      [[fallthrough]];
    default: {
      break;
    }
    }

    if (limitSchedulingRange) {
      rewriter.setInsertionPoint(block->getTerminator());
      createSchedBarrier(rewriter, loc,
                         mlir::amdgpu::sched_barrier_opt_enum::none);
    }

    rewriter.eraseOp(instructionSchedHint);
    return success();
  }

private:
  int32_t numStages;
  SchedulingType schedulingType;
};

struct TritonAMDGPULowerInstructionSchedHints
    : public triton::impl::TritonAMDGPULowerInstructionSchedHintsBase<
          TritonAMDGPULowerInstructionSchedHints> {

  explicit TritonAMDGPULowerInstructionSchedHints(int32_t numStages,
                                                  std::string variant) {
    this->numStages = numStages;
    this->variant = variant;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<triton::amdgpu::InstructionSchedHint>();
    target.addLegalOp<ROCDL::SchedBarrier>();
    target.addLegalOp<ROCDL::IglpOpt>();
    target.addLegalOp<ROCDL::SchedGroupBarrier>();

    RewritePatternSet patterns(ctx);

    patterns.add<InstructionSchedHintsRewriter>(ctx, this->numStages,

                                                this->variant);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {

      signalPassFailure();
    }
  }
};

struct TritonAMDGPUInsertInstructionSchedHints
    : public triton::impl::TritonAMDGPUInsertInstructionSchedHintsBase<
          TritonAMDGPUInsertInstructionSchedHints> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();
    int64_t nextLoopId = 0;

    mod.walk([&](scf::ForOp forOp) {
      // Insert one hint only for a loop that owns exactly one `tt.dot`;
      // nested loops are handled independently.
      if (auto dotOp = getSingleDotOpIfExists(forOp)) {
        OpBuilder rewriter(ctx);
        auto loopId = rewriter.getI64IntegerAttr(nextLoopId++);
        forOp->walk([&](Operation *op) {
          if (op->getParentOfType<scf::ForOp>() != forOp)
            return;
          if (isa<triton::DotOp, triton::LoadOp, triton::amdgpu::BufferLoadOp,
                  triton::amdgpu::BufferLoadToLocalOp, gpu::LocalLoadOp,
                  gpu::LocalStoreOp, gpu::AsyncCopyGlobalToLocalOp>(op))
            op->setAttr(triton::amdgpu::InstructionSchedLoopIdAttrName, loopId);
        });
        rewriter.setInsertionPointAfter(dotOp);
        auto schedHint = rewriter.create<triton::amdgpu::InstructionSchedHint>(
            dotOp->getLoc());
        schedHint->setAttr(triton::amdgpu::InstructionSchedLoopIdAttrName,
                           loopId);
      }
    });
  }
};
} // namespace

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPULowerInstructionSchedHintsPass(int32_t numStages,
                                                 std::string variant) {
  return std::make_unique<TritonAMDGPULowerInstructionSchedHints>(numStages,
                                                                  variant);
}

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUInsertInstructionSchedHintsPass() {
  return std::make_unique<TritonAMDGPUInsertInstructionSchedHints>();
}
} // namespace mlir::triton
