#ifndef TRITON_KUNLUN_XPUC_EMITTER_H
#define TRITON_KUNLUN_XPUC_EMITTER_H

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>

namespace mlir::triton::kunlun {

// Lowers a TTIR module to a single self-contained XPU-C translation unit.
//
// The emitter is a reference code generator with two execution modes:
//
//   serial       one core executes one whole program id; every tensor value is
//                materialised in full in that core's local memory. Slow, but
//                every value is visible to one core, so it is the golden
//                reference the distributed mode is checked against.
//   distributed  all 16 cores of a cluster execute one program id together;
//                each core owns a contiguous 1/16 slice of every tensor, and
//                reductions publish one partial per core through cluster
//                shared memory (LM2SM / sync / SM2LM) where every core then
//                combines all 16 partials.
//
// Distributed mode requires every tensor to have a multiple of 16 elements
// and refuses broadcasts that change the element count, because those need
// elements another core holds. Unsupported input is a hard error on purpose:
// there is no silent fallback.
bool emitXPUCSource(ModuleOp module, int64_t lmBudgetBytes, bool distributed,
                    std::string &source, std::string &entryName,
                    std::string &error);

} // namespace mlir::triton::kunlun

#endif // TRITON_KUNLUN_XPUC_EMITTER_H
