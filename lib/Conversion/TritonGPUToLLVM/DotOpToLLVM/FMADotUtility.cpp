#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "../FMACompatibility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include <cstddef>
#include <unordered_map>

using namespace mlir;

namespace {

struct OperandValueKey {
  unsigned bRepIdx, nonKRepIdx;
  unsigned bIdx, nonKIdx, kIdx;

  bool operator==(const OperandValueKey &other) const {
    return bRepIdx == other.bRepIdx && nonKRepIdx == other.nonKRepIdx &&
           bIdx == other.bIdx && nonKIdx == other.nonKIdx && kIdx == other.kIdx;
  }
};

} // namespace

template <> struct std::hash<OperandValueKey> {
  std::size_t operator()(const OperandValueKey &key) const {
    return llvm::hash_combine(key.bRepIdx, key.nonKRepIdx, key.bIdx,
                              key.nonKIdx, key.kIdx);
  }
};

namespace {

using ValueTableFMA = std::unordered_map<OperandValueKey, Value>;

ValueTableFMA getValueTableFromStructFMA(
    Value value, ArrayRef<unsigned> perRepShape, ArrayRef<unsigned> repetitions,
    unsigned kDim, unsigned nonKDim, ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<unsigned> inRepOrder) {
  ValueTableFMA result;
  auto elements = unpackLLElements(loc, value, rewriter);
  assert(perRepShape.size() == 3);
  assert(repetitions.size() == 3);
  assert(inRepOrder.size() == 3);
  SmallVector<unsigned> perThreadShape(3);
  for (unsigned dim = 0; dim < 3; ++dim)
    perThreadShape[dim] = perRepShape[dim] * repetitions[dim];
  assert(elements.size() == product(perThreadShape));
  assert((kDim == 1 || kDim == 2) && (nonKDim == 1 || nonKDim == 2));

  // SharedToDotOperandFMA packs the full per-thread tile in D-layout order.
  // Decode combined coordinates so K and non-K repetitions may interleave.
  for (unsigned index = 0; index < elements.size(); ++index) {
    auto spatialIndex = mlir::triton::gpu::fma_compat::delinearizeIndex(
        index, perThreadShape, inRepOrder);
    SmallVector<unsigned> repSpatialIndex(3);
    SmallVector<unsigned> inRepSpatialIndex(3);
    for (unsigned dim = 0; dim < 3; ++dim) {
      repSpatialIndex[dim] = spatialIndex[dim] / perRepShape[dim];
      inRepSpatialIndex[dim] = spatialIndex[dim] % perRepShape[dim];
    }
    OperandValueKey key{repSpatialIndex[0], repSpatialIndex[nonKDim],
                        inRepSpatialIndex[0], inRepSpatialIndex[nonKDim],
                        inRepSpatialIndex[kDim]};
    result[key] = elements[index];
  }
  return result;
}

} // namespace

namespace mlir::triton::gpu {

LogicalResult parametricConvertFMADot(DotOp op, DotOp::Adaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      FMAVectorMultiplier &multiplier) {
  Location loc = op.getLoc();
  auto aTensorType = cast<RankedTensorType>(op.getA().getType());
  auto dTensorType = cast<RankedTensorType>(op.getResult().getType());

  SmallVector<int64_t> aShapePerCTA =
      mlir::triton::gpu::fma_compat::expandMatrixShapeWithBatch(
          ArrayRef(getShapePerCTA(aTensorType)));
  SmallVector<int64_t> dShapePerCTA =
      mlir::triton::gpu::fma_compat::expandMatrixShapeWithBatch(
          ArrayRef(getShapePerCTA(dTensorType)));

  auto dLayout = cast<BlockedEncodingAttr>(dTensorType.getEncoding());
  auto inRepOrder = mlir::triton::gpu::fma_compat::expandMatrixOrderWithBatch(
      dLayout.getOrder());
  // Blocked layout repetitions use the same order as in-tile elements.
  ArrayRef<unsigned> repOrder = inRepOrder;

  SmallVector<Value> accumulators =
      unpackLLElements(loc, adaptor.getC(), rewriter);
  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  SmallVector<unsigned> sizePerThread = getContigPerThread(dLayout);
  const unsigned numElementsPerThread = product(sizePerThread);
  SmallVector<unsigned> shapePerCTATile =
      getShapePerCTATile(dLayout, dTensorType.getShape());
  sizePerThread = mlir::triton::gpu::fma_compat::expandMatrixShapeWithBatch(
      ArrayRef(sizePerThread));
  shapePerCTATile = mlir::triton::gpu::fma_compat::expandMatrixShapeWithBatch(
      ArrayRef(shapePerCTATile));

  const unsigned kSize = aShapePerCTA[2];
  SmallVector<unsigned> repetitions(3);
  for (unsigned dim = 0; dim < 3; ++dim)
    repetitions[dim] =
        ceil(dShapePerCTA[dim], static_cast<int64_t>(shapePerCTATile[dim]));

  auto aValues = getValueTableFromStructFMA(
      llA, {sizePerThread[0], sizePerThread[1], kSize},
      {repetitions[0], repetitions[1], 1}, /*kDim=*/2, /*nonKDim=*/1, rewriter,
      loc, inRepOrder);
  auto bValues = getValueTableFromStructFMA(
      llB, {sizePerThread[0], kSize, sizePerThread[2]},
      {repetitions[0], 1, repetitions[2]}, /*kDim=*/1, /*nonKDim=*/2, rewriter,
      loc, inRepOrder);

  for (unsigned bRep = 0; bRep < repetitions[0]; ++bRep)
    for (unsigned mRep = 0; mRep < repetitions[1]; ++mRep)
      for (unsigned nRep = 0; nRep < repetitions[2]; ++nRep)
        for (unsigned b = 0; b < sizePerThread[0]; ++b)
          for (unsigned m = 0; m < sizePerThread[1]; ++m)
            for (unsigned n = 0; n < sizePerThread[2]; ++n) {
              SmallVector<unsigned> inRepIndex = {b, m, n};
              unsigned linearInRepIndex = static_cast<unsigned>(
                  mlir::triton::gpu::fma_compat::linearizeIndex(
                      inRepIndex, sizePerThread, inRepOrder));
              SmallVector<unsigned> repIndex = {bRep, mRep, nRep};
              unsigned linearRepIndex = static_cast<unsigned>(
                  mlir::triton::gpu::fma_compat::linearizeIndex(
                      repIndex, repetitions, repOrder));
              unsigned accumulatorIndex =
                  linearInRepIndex + linearRepIndex * numElementsPerThread;

              SmallVector<Value> aVector;
              SmallVector<Value> bVector;
              aVector.reserve(kSize);
              bVector.reserve(kSize);
              for (unsigned k = 0; k < kSize; ++k) {
                aVector.push_back(aValues.at({bRep, mRep, b, m, k}));
                bVector.push_back(bValues.at({bRep, nRep, b, n, k}));
              }
              accumulators[accumulatorIndex] = multiplier.multiplyVectors(
                  aVector, bVector, accumulators[accumulatorIndex]);
            }

  Value result =
      packLLElements(loc, typeConverter, accumulators, rewriter, dTensorType);
  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::triton::gpu
