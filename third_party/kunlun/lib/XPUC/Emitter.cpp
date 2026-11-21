#include "XPUC/Emitter.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace mlir::triton::kunlun {

namespace {

using llvm::StringRef;
using llvm::Twine;

// XPU1 has no GM->register path: every global access is a DMA into LM.
// GM2LM on KL1 needs a 32-byte aligned LM destination and a byte count that is
// a multiple of 32, so tensor accesses must be contiguous blocks. Stores go
// through LM2GM, which only requires 32-byte aligned LM source and any size>0.
// Compute runs over the LM arrays: f32 add/sub/mul use the 8-lane 256-bit SIMD
// intrinsics, everything else a scalar loop.
enum class StrideKind { Zero, One, Other };

struct Affine {
  std::string start;  // element offset, C integer expression
  StrideKind kind = StrideKind::Other;
};

struct PtrBlock {
  std::string base;  // C expression of type <elem C type>*
  Type elemTy;
  std::string start;
  StrideKind kind = StrideKind::Other;
};

// Device code cannot type-pun through a pointer, so f32 <-> i32 bitcasts go
// through this union. The core cannot touch GM directly, but a union lives in
// registers/LM, so it is legal. Only the direction a kernel uses is emitted.
static constexpr const char *kFloatToBitsHelper =
    "static inline unsigned int __device__ _kl_bits_of_float(float x) {\n"
    "  union { float f; unsigned int u; } c;\n"
    "  c.f = x;\n"
    "  return c.u;\n"
    "}\n\n";

static constexpr const char *kBitsToFloatHelper =
    "static inline float __device__ _kl_float_of_bits(unsigned int x) {\n"
    "  union { float f; unsigned int u; } c;\n"
    "  c.u = x;\n"
    "  return c.f;\n"
    "}\n\n";

class XPUCEmitter {
public:
  XPUCEmitter(ModuleOp module, int64_t lmBudget, bool distributed)
      : module_(module), lmBudget_(lmBudget), distributed_(distributed) {}

  bool run(std::string &source, std::string &entryName, std::string &error) {
    if (!findFunction() || !emitSignature() || !emitBody()) {
      error = error_;
      return false;
    }
    source = assemble();
    entryName = entryName_;
    return true;
  }

private:
  //===--------------------------------------------------------------------===//
  // Diagnostics
  //===--------------------------------------------------------------------===//

  bool fail(const Twine &msg) {
    if (error_.empty())
      error_ = msg.str();
    return false;
  }

  bool failOp(Operation *op, const Twine &msg) {
    if (!op)
      return fail(Twine("XPU-C reference emitter: ") + msg);
    return fail(Twine("XPU-C reference emitter: ") + msg + " (while lowering '" +
                 op->getName().getStringRef() + "')");
  }

  bool failValue(Value v, const Twine &msg) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return fail(Twine("XPU-C reference emitter: ") + msg + " (block argument)");
    return failOp(def, msg);
  }

  std::string failed(Operation *op, const Twine &msg) {
    failOp(op, msg);
    return "";
  }

  //===--------------------------------------------------------------------===//
  // Execution mode
  //===--------------------------------------------------------------------===//

  // A K200 launch always fills a whole cluster. In distributed mode those 16
  // cores split every tensor, so a value of n elements becomes n/16 per core.
  static constexpr int64_t kClusterCores = 16;
  static constexpr int64_t kSmBudget = 256 * 1024;

  // Failures that only exist because distributed mode hands each core a 1/16
  // slice. The backend retries exactly these in serial mode, so they carry a
  // stable marker: a real limitation must never look like a layout mismatch.
  static constexpr const char *kDistributedMark = "distributed layout: ";

  bool localCount(int64_t n, StringRef what, Operation *op, int64_t &out) {
    if (!distributed_) {
      out = n;
      return true;
    }
    if (n % kClusterCores != 0)
      return failOp(op, Twine(kDistributedMark) + Twine(what) + " has " + Twine(n) +
                            " elements, which does not split across the " + Twine(kClusterCores) +
                            " cores of a K200 cluster");
    out = n / kClusterCores;
    return true;
  }

  // Element offset of this core's slice inside a value, as a C expression.
  static std::string coreOffset(int64_t elems) {
    return "_kl_core * " + std::to_string(elems);
  }

  //===--------------------------------------------------------------------===//
  // Type mapping
  //===--------------------------------------------------------------------===//

  static std::string cType(Type t) {
    if (t.isF32())
      return "float";
    if (auto it = dyn_cast<IntegerType>(t)) {
      unsigned w = it.getWidth();
      // MLIR signless (the ordinary Triton i32) means "signed" for every op
      // that cares: divsi, shrsi, remsi, cmpi slt and sitofp all rely on the
      // C operand type being signed, and the *ui* variants cast explicitly.
      bool s = !it.isUnsigned();
      if (w == 1)
        return "char";
      if (w == 8)
        return s ? "char" : "unsigned char";
      if (w == 16)
        return s ? "short" : "unsigned short";
      if (w == 32)
        return s ? "int" : "unsigned";
      if (w == 64)
        return s ? "long long" : "unsigned long long";
      return "";
    }
    return "";
  }

  static int64_t cSize(Type t) {
    if (t.isF32())
      return 4;
    if (auto it = dyn_cast<IntegerType>(t)) {
      unsigned w = it.getWidth();
      if (w <= 8)
        return 1;
      if (w <= 16)
        return 2;
      if (w <= 32)
        return 4;
      return 8;
    }
    return 0;
  }

  // Unsigned C type of the same width. A plain `unsigned` cast would truncate
  // i64 values and sign-extend i32 ones, so every width needs its own spelling.
  static std::string unsignedCTy(Type t) {
    auto it = dyn_cast<IntegerType>(t);
    if (!it)
      return "";
    switch (it.getWidth()) {
    case 1:
    case 8:
      return "unsigned char";
    case 16:
      return "unsigned short";
    case 32:
      return "unsigned int";
    case 64:
      return "unsigned long long";
    }
    return "";
  }

  // Signed C type of the same width; cType() already spells signless integers
  // this way, so only an unsigned source ever needs it.
  static std::string signedCTy(Type t) {
    auto it = dyn_cast<IntegerType>(t);
    if (!it)
      return "";
    switch (it.getWidth()) {
    case 1:
    case 8:
      return "char";
    case 16:
      return "short";
    case 32:
      return "int";
    case 64:
      return "long long";
    }
    return "";
  }

  // True for an MLIR integer type that is explicitly unsigned (ui32 etc.).
  static bool isUnsignedInt(Type t) {
    auto it = dyn_cast<IntegerType>(t);
    return it && it.isUnsigned();
  }

  static std::string floatLit(double v) {
    if (std::isnan(v))
      return "__builtin_nanf(\"\")";
    if (std::isinf(v))
      return v > 0 ? "__builtin_inff()" : "(-__builtin_inff())";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    std::string s(buf);
    if (s.find_first_of(".eE") == std::string::npos)
      s += ".0";
    return s + "f";
  }

  // Rank-0 tensors (the result of reducing a 1-D block) are scalars here.
  static bool isTensor(Value v) {
    auto rt = dyn_cast<RankedTensorType>(v.getType());
    return rt && rt.getRank() > 0;
  }

  static Type elemTyOf(Type t) {
    if (auto rt = dyn_cast<RankedTensorType>(t))
      return rt.getElementType();
    return t;
  }

  static int64_t intAttr(Operation *op, StringRef name, int64_t dflt = 0) {
    if (auto a = op->getAttrOfType<IntegerAttr>(name))
      return a.getInt();
    return dflt;
  }

  static bool isIntOne(Value v) {
    Operation *def = v.getDefiningOp();
    if (!def || def->getName().getStringRef() != "arith.constant")
      return false;
    if (auto a = def->getAttrOfType<IntegerAttr>("value"))
      return a.getValue().isOne();
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Module / function discovery
  //===--------------------------------------------------------------------===//

  bool findFunction() {
    if (!module_)
      return fail("XPU-C reference emitter: empty module");
    for (Operation &op : module_.getBody()->getOperations()) {
      if (op.getName().getStringRef() != "tt.func")
        continue;
      if (func_)
        return fail("XPU-C reference emitter: expected exactly one tt.func");
      func_ = &op;
    }
    if (!func_)
      return fail("XPU-C reference emitter: no tt.func in module");
    return true;
  }

  bool emitSignature() {
    Region &region = func_->getRegion(0);
    if (!llvm::hasSingleElement(region))
      return failOp(func_, "expected a single-block kernel body");
    Block &block = region.front();

    if (auto sym = func_->getAttrOfType<StringAttr>("sym_name"))
      entryName_ = sanitize(sym.getValue());
    if (entryName_.empty())
      entryName_ = "triton_kernel";

    auto typeAttr = func_->getAttrOfType<TypeAttr>("function_type");
    if (!typeAttr)
      return failOp(func_, "missing function_type attribute");
    auto fnTy = dyn_cast<FunctionType>(typeAttr.getValue());
    if (!fnTy)
      return failOp(func_, "function_type is not a FunctionType");
    if (fnTy.getNumResults() != 0)
      return failOp(func_, "kernels returning values are not supported");
    if (fnTy.getNumInputs() != block.getNumArguments())
      return failOp(func_, "function_type does not match block arguments");

    std::string sig = "__global__ void " + entryName_ + "(";
    for (unsigned i = 0; i < fnTy.getNumInputs(); ++i) {
      Type ty = fnTy.getInput(i);
      Value arg = block.getArgument(i);
      std::string decl;
      if (auto ptrTy = dyn_cast<PointerType>(ty)) {
        std::string pt = cType(ptrTy.getPointeeType());
        if (pt.empty())
          return failOp(func_, "unsupported pointer element type");
        decl = pt + "* _kl_a" + std::to_string(i);
        scalarPtrs_[arg] = "_kl_a" + std::to_string(i);
        ptrElemTy_[arg] = ptrTy.getPointeeType();
      } else {
        if (isTensor(arg))
          return failOp(func_, "tensor kernel arguments are not supported");
        std::string ct = cType(ty);
        if (ct.empty())
          return failOp(func_, "unsupported kernel argument type");
        decl = ct + " _kl_a" + std::to_string(i);
        scalars_[arg] = "_kl_a" + std::to_string(i);
      }
      sig += (i == 0 ? "" : ", ") + decl;
    }
    // The launcher appends the grid so the kernel can run a persistent loop.
    // A kernel whose parameters are all constexpr has none of its own, so the
    // leading comma must not be emitted.
    sig += (fnTy.getNumInputs() == 0 ? "" : ", ") +
           "int _kl_np, int _kl_g0, int _kl_g1, int _kl_g2) {\n";
    sig_ = sig;
    return true;
  }

  static std::string sanitize(StringRef name) {
    std::string out;
    for (char c : name) {
      bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
      out += ok ? c : '_';
    }
    if (!out.empty() && out[0] >= '0' && out[0] <= '9')
      out = "k_" + out;
    return out;
  }

  //===--------------------------------------------------------------------===//
  // Body
  //===--------------------------------------------------------------------===//

  bool emitBody() {
    Block &block = func_->getRegion(0).front();
    for (Operation &op : block)
      if (!emitOp(&op))
        return false;
    if (lmBytes_ > lmBudget_)
      return fail(Twine("XPU-C reference emitter: kernel needs ") + Twine(lmBytes_) +
                  " bytes of local memory per core, budget is " + Twine(lmBudget_) +
                  "; every tensor is materialised in LM by the reference emitter");
    if (smBytes_ > kSmBudget)
      return fail(Twine("XPU-C reference emitter: kernel needs ") + Twine(smBytes_) +
                  " bytes of shared memory, one cluster has " + Twine(kSmBudget));
    return true;
  }

  bool emitOp(Operation *op) {
    StringRef name = op->getName().getStringRef();

    if (name == "tt.return" || name == "tt.get_program_id" || name == "tt.get_num_programs")
      return true;  // folded into expressions on use
    if (name == "arith.constant")
      return emitConstant(op);

    // Offset tensors are recomputed by affine() when the address is used, so
    // they never need an LM array of their own.
    if (name == "tt.make_range" || name == "tt.splat") {
      if (op->getNumResults() == 1 && isDeferredOffset(op->getResult(0)))
        return true;
      return name == "tt.make_range" ? emitMakeRange(op) : emitSplat(op);
    }
    if (name == "tt.broadcast" || name == "tt.expand_dims") {
      if (op->getNumResults() == 1 && isDeferredOffset(op->getResult(0)))
        return true;
      return emitBroadcast(op);
    }
    if (name == "tt.addptr")
      return emitAddPtr(op);
    if (name == "tt.load")
      return emitLoad(op);
    if (name == "tt.store")
      return emitStore(op);
    if (name == "tt.reduce")
      return emitReduce(op);
    if (isElementwise(op)) {
      if (op->getNumResults() == 1 && isDeferredOffset(op->getResult(0)))
        return true;
      return emitElementwise(op);
    }

    if (name == "scf.for" || name == "scf.while" || name == "scf.if")
      return failOp(op, "control flow is not supported yet; the reference emitter "
                        "executes a straight-line program per program id");
    if (name == "tt.dot")
      return failOp(op, "tt.dot is not supported by the reference emitter; it needs CDNN");
    if (name == "tt.atomic_rmw" || name == "tt.atomic_cas")
      return failOp(op, "atomics are not supported");
    if (name == "tt.scan")
      return failOp(op, "tt.scan is not supported");
    return failOp(op, "unsupported operation");
  }

  //===--------------------------------------------------------------------===//
  // Value materialisation
  //===--------------------------------------------------------------------===//

  std::string fresh(const char *prefix) {
    return std::string(prefix) + std::to_string(nextId_++);
  }

  // An LM array that takes part in a 256-bit access has to be 32-byte aligned.
  // On XPU1 `__local__` already expands to `__attribute__((aligned(32)))`, but
  // the macro is empty for the host pass and for later XPUs, so the attribute
  // is spelled out; a duplicate identical attribute is accepted. The declared
  // size is rounded up because that alignment can insert padding between two
  // arrays.
  std::string lmArray(Type elemTy, const std::string &name, int64_t count) {
    lmBytes_ += (count * cSize(elemTy) + 31) / 32 * 32;
    return "    __local__ " + cType(elemTy) + " " + name + "[" + std::to_string(count) +
           "] __attribute__((aligned(32)));\n";
  }

  // Declares an LM array for a tensor result and fills it element by element.
  template <typename Fn>
  bool emitTensor(Value result, Fn makeExpr) {
    auto ty = dyn_cast<RankedTensorType>(result.getType());
    if (!ty)
      return failValue(result, "expected a tensor result");
    std::string ct = cType(ty.getElementType());
    if (ct.empty())
      return failValue(result, "unsupported tensor element type");
    int64_t n = ty.getNumElements();

    // A one-element block is a scalar with a shape: every core holds it whole,
    // so it stays an expression and is never sliced. This is what carries a
    // reduced scalar across `expand_dims`/`broadcast` without the slice of the
    // broadcast result disagreeing with the slice of the scalar.
    if (n == 1) {
      std::string e = makeExpr("0");
      if (e.empty())
        return false;
      scalarTensors_[result] = e;
      return true;
    }

    int64_t count = n;
    if (!localCount(n, "tensor value", result.getDefiningOp(), count))
      return false;
    std::string nm = fresh("v");
    decls_ += lmArray(ty.getElementType(), nm, count);
    tensors_[result] = nm;
    std::string iv = fresh("_kl_i");
    std::string expr = makeExpr(iv);
    if (expr.empty())
      return false;
    body_ += "    for (int " + iv + " = 0; " + iv + " < " + std::to_string(count) + "; ++" + iv +
             ") " + nm + "[" + iv + "] = " + expr + ";\n";
    return true;
  }

  bool emitConstant(Operation *op) {
    Value result = op->getResult(0);
    Attribute attr = op->getAttr("value");
    if (!isTensor(result))
      return !constantLiteral(op).empty();  // scalars are inlined on use
    auto dense = dyn_cast<DenseElementsAttr>(attr);
    if (!dense)
      return failOp(op, "unsupported tensor constant attribute");
    if (!dense.isSplat())
      return failOp(op, "only splat tensor constants are supported");
    Type ety = cast<RankedTensorType>(result.getType()).getElementType();
    std::string lit;
    if (auto it = dyn_cast<IntegerType>(ety)) {
      int64_t v = dense.getSplatValue<APInt>().getSExtValue();
      lit = it.getWidth() == 1 ? (v == 0 ? "0" : "1") : std::to_string(v);
    } else if (ety.isF32()) {
      lit = floatLit(dense.getSplatValue<APFloat>().convertToFloat());
    } else {
      return failOp(op, "only integral and f32 tensor constants are supported");
    }
    // A splat constant needs no LM array until something moves the whole
    // block: every lane can read the literal, and a 256-bit op can use its
    // scalar-vector form.
    splatScalars_[result] = lit;
    return true;
  }

  // Declares a scalar result and assigns it.
  bool emitScalar(Value result, const std::string &expr) {
    std::string ct = cType(elemTyOf(result.getType()));
    if (ct.empty())
      return failValue(result, "unsupported scalar result type");
    std::string nm = fresh("s");
    decls_ += "    " + ct + " " + nm + ";\n";
    scalars_[result] = nm;
    body_ += "    " + nm + " = " + expr + ";\n";
    return true;
  }

  // Gives a lazily held scalar splat its LM array, for the operations that move
  // or fold a whole block and cannot read a per-lane expression.
  bool materialiseSplat(Value v) {
    auto it = splatScalars_.find(v);
    if (it == splatScalars_.end())
      return true;
    std::string e = it->second;
    if (!emitTensor(v, [&](const std::string &) { return e; }))
      return false;
    splatScalars_.erase(v);
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Element addressing
  //===--------------------------------------------------------------------===//

  static int64_t numElements(ArrayRef<int64_t> shape) {
    int64_t n = 1;
    for (int64_t d : shape)
      n *= d;
    return n;
  }

  // Maps a linear index `iv` over `dstShape` onto the linear index of `src`.
  static std::string indexMap(ArrayRef<int64_t> src, ArrayRef<int64_t> dst, const std::string &iv) {
    if (src == dst)
      return iv;
    if (src.size() != dst.size()) {
      // expand_dims only inserts size-1 dimensions, so the row-major linear
      // order is unchanged and `iv` stays valid.
      return numElements(src) == numElements(dst) ? iv : "";
    }
    std::vector<int64_t> dstride(dst.size(), 1), sstride(src.size(), 1);
    for (int i = int(dst.size()) - 2; i >= 0; --i)
      dstride[i] = dstride[i + 1] * dst[i + 1];
    for (int i = int(src.size()) - 2; i >= 0; --i)
      sstride[i] = sstride[i + 1] * src[i + 1];
    std::string out;
    for (size_t i = 0; i < src.size(); ++i) {
      if (src[i] == 1 || dst[i] == 1)
        continue;
      std::string term = "((" + iv + " / " + std::to_string(dstride[i]) + ") % " +
                         std::to_string(dst[i]) + ")";
      if (sstride[i] != 1)
        term = "(" + term + " * " + std::to_string(sstride[i]) + ")";
      if (!out.empty())
        out += " + ";
      out += term;
    }
    return out.empty() ? "0" : out;
  }

  // C expression for element `iv` of `v` when the surrounding result has
  // shape `dstShape`.
  std::string elem(Value v, ArrayRef<int64_t> dstShape, const std::string &iv,
                   const llvm::DenseMap<Value, std::string> &ctx) {
    if (isTensor(v)) {
      auto ty = cast<RankedTensorType>(v.getType());
      if (auto st = scalarTensors_.find(v); st != scalarTensors_.end())
        return st->second;
      // A scalar splat is replicated on every core, so it is a scalar here too
      // and needs no index.
      if (auto sp = splatScalars_.find(v); sp != splatScalars_.end())
        return sp->second;
      auto it = tensors_.find(v);
      if (it == tensors_.end()) {
        failValue(v, "tensor value was not materialised");
        return "";
      }
      if (distributed_ && numElements(ty.getShape()) != numElements(dstShape)) {
        // Element i of the result lives on the core that owns element i of its
        // operands, so an operand with a different element count would need
        // elements this core never loaded.
        failValue(v, Twine(kDistributedMark) +
                         "a tensor cannot be broadcast across cores (operand and result "
                         "hold a different number of elements)");
        return "";
      }
      if (distributed_ && ty.getShape() != dstShape &&
          static_cast<size_t>(ty.getRank()) == dstShape.size()) {
        // indexMap would reinterpret this core's local index with global
        // strides, which only works while the row-major order is preserved.
        failValue(v, Twine(kDistributedMark) +
                         "a tensor cannot be remapped between equally sized shapes");
        return "";
      }
      std::string idx = indexMap(ty.getShape(), dstShape, iv);
      if (idx.empty()) {
        failValue(v, "rank mismatch between operand and result");
        return "";
      }
      return it->second + "[" + idx + "]";
    }
    return scalarExpr(v, ctx);
  }

  // Pure scalar expression. Tensors are never expanded here.
  std::string scalarExpr(Value v, const llvm::DenseMap<Value, std::string> &ctx) {
    auto alias = ctx.find(v);
    if (alias != ctx.end())
      return alias->second;
    auto named = scalars_.find(v);
    if (named != scalars_.end())
      return named->second;
    // A rank-0 block (`tt.splat`, `tt.broadcast` of a reduced scalar) is also
    // a scalar: it was folded into an expression when it was emitted.
    auto folded = scalarTensors_.find(v);
    if (folded != scalarTensors_.end())
      return folded->second;
    // A scalar splat that no bulk transfer has needed yet.
    auto splat = splatScalars_.find(v);
    if (splat != splatScalars_.end())
      return splat->second;
    if (scalarPtrs_.count(v)) {
      failValue(v, "pointer used as an integer");
      return "";
    }

    Operation *def = v.getDefiningOp();
    if (!def) {
      failValue(v, "unsupported scalar value");
      return "";
    }
    StringRef name = def->getName().getStringRef();

    if (name == "arith.constant")
      return constantLiteral(def);
    if (name == "tt.get_program_id" || name == "tt.get_num_programs") {
      int64_t axis = intAttr(def, "axis");
      if (axis < 0 || axis > 2) {
        failOp(def, "program id axis must be 0, 1 or 2");
        return "";
      }
      return (name == "tt.get_program_id" ? "_kl_p" : "_kl_g") + std::to_string(axis);
    }
    if (isElementwise(def))
      return elementwiseExpr(def, {}, "", ctx);

    failOp(def, "unsupported scalar expression");
    return "";
  }

  std::string constantLiteral(Operation *def) {
    Type ty = def->getResult(0).getType();
    Attribute a = def->getAttr("value");
    if (auto fa = dyn_cast<FloatAttr>(a)) {
      if (!ty.isF32()) {
        failOp(def, "only f32 floating point constants are supported");
        return "";
      }
      return floatLit(fa.getValueAsDouble());
    }
    if (auto ia = dyn_cast<IntegerAttr>(a)) {
      if (ty.isInteger(1))
        return ia.getInt() == 0 ? "0" : "1";
      return std::to_string(ia.getInt());
    }
    failOp(def, "unsupported constant attribute");
    return "";
  }

  //===--------------------------------------------------------------------===//
  // Elementwise operations
  //===--------------------------------------------------------------------===//

  static bool isBitcast(StringRef n) { return n == "arith.bitcast" || n == "tt.bitcast"; }

  static bool isCast(StringRef n) {
    return n == "arith.extf" || n == "arith.truncf" || n == "arith.extsi" ||
           n == "arith.extui" || n == "arith.trunci" || n == "arith.sitofp" ||
           n == "arith.uitofp" || n == "arith.fptosi" || n == "arith.fptoui" ||
           n == "arith.index_cast" || isBitcast(n) || n == "tt.fp_to_fp";
  }

  static bool isMath(StringRef n) { return n.starts_with("math."); }

  static bool isElementwise(Operation *op) {
    StringRef n = op->getName().getStringRef();
    if (isCast(n) || isMath(n) || n == "arith.select")
      return true;
    static const char *kOps[] = {
        "arith.addi",    "arith.subi",    "arith.muli",    "arith.divsi",   "arith.divui",
        "arith.remsi",   "arith.remui",   "arith.andi",    "arith.ori",     "arith.xori",
        "arith.shli",    "arith.shrsi",   "arith.shrui",   "arith.addf",    "arith.subf",
        "arith.mulf",    "arith.divf",    "arith.remf",    "arith.maximumf", "arith.minimumf",
        "arith.maxnumf", "arith.minnumf", "arith.maxsi",   "arith.maxui",   "arith.minsi",
        "arith.minui",   "arith.cmpi",    "arith.cmpf",
    };
    for (const char *c : kOps)
      if (n == c)
        return true;
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Vectorised elementwise
  //===--------------------------------------------------------------------===//

  // XPU1 computes 8 f32 lanes per 256-bit instruction and the intrinsics read
  // and write LM directly, so an elementwise loop over an LM array maps onto
  // them at one instruction per 8 elements. Only add, sub and mul have a
  // 256-bit form; everything else keeps the scalar loop.
  //
  // A rejected fast path returns false without having emitted anything, so the
  // caller can fall back to the scalar loop.
  bool tryEmitSimd(Operation *op) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return false;
    Value result = op->getResult(0);
    auto ty = dyn_cast<RankedTensorType>(result.getType());
    if (!ty || !ty.getElementType().isF32())
      return false;
    StringRef name = op->getName().getStringRef();
    bool isAdd = name == "arith.addf";
    bool isSub = name == "arith.subf";
    bool isMul = name == "arith.mulf";
    if (!isAdd && !isSub && !isMul)
      return false;

    int64_t count = ty.getNumElements();
    if (count < 8)
      return false;
    // One 256-bit step covers a whole vector, so a slice shorter than that has
    // nothing to vectorise.
    if (!localCount(count, "tensor value", op, count) || count < 8)
      return false;

    // Each operand must be either an LM array of exactly this shape, so that
    // lane i of the loop is lane i of every operand, or a scalar. A broadcast
    // or a remapped shape needs an index expression the vector form cannot
    // express, so those stay scalar.
    bool isArray[2] = {false, false};
    std::string array[2], scalar[2];
    for (unsigned i = 0; i < 2; ++i) {
      Value v = op->getOperand(i);
      if (isTensor(v) && !scalarTensors_.count(v) && !splatScalars_.count(v)) {
        auto oty = cast<RankedTensorType>(v.getType());
        auto it = tensors_.find(v);
        if (!oty.getElementType().isF32() || oty.getShape() != ty.getShape() ||
            it == tensors_.end())
          return false;
        isArray[i] = true;
        array[i] = it->second;
        continue;
      }
      std::string e = scalarExpr(v, emptyCtx_);
      if (e.empty())
        return false;
      scalar[i] = e;
    }
    if (!isArray[0] && !isArray[1])
      return false;  // the result would be a scalar, not a block

    // `x - a` has no form with the scalar on the right, but -a + x is the same
    // value bit for bit, so the scalar operand is negated up front.
    bool negateRight = isSub && isArray[0] && !isArray[1];

    std::string iv = fresh("_kl_i");
    std::string nm = fresh("v");
    std::string hoist, arg[2];
    for (unsigned i = 0; i < 2; ++i) {
      if (isArray[i]) {
        arg[i] = "&" + array[i] + "[" + iv + "]";
        continue;
      }
      // Hoisting the scalar keeps the loop body a single call and gives the
      // negation above a value to apply to. It belongs in the body, not in the
      // declarations: `decls_` is emitted before `body_`, and a scalar operand
      // may be a reduction result that no statement has assigned yet.
      std::string tmp = fresh("_kl_s");
      hoist += "    float " + tmp + " = " +
               (negateRight && i == 1 ? "-(" + scalar[i] + ")" : scalar[i]) + ";\n";
      arg[i] = tmp;
    }

    const char *fn;
    if (isAdd)
      fn = isArray[0] && isArray[1] ? "_x256_vvadd_ls" : "_x256_svadd_ls";
    else if (isMul)
      fn = isArray[0] && isArray[1] ? "_x256_vvmul_ls" : "_x256_svmul_ls";
    else if (isArray[0] && isArray[1])
      fn = "_x256_vvsub_ls";
    else
      // a - x is svsub directly; x - a was negated above and became an add.
      fn = isArray[0] ? "_x256_svadd_ls" : "_x256_svsub_ls";

    // The vector forms take (vec, vec, out) or (scalar, vec, out), so the
    // scalar operand of a mixed op is always the first argument.
    std::string call;
    if (isArray[0] && isArray[1])
      call = std::string(fn) + "(" + arg[0] + ", " + arg[1] + ", &" + nm + "[" + iv + "])";
    else if (isArray[0])
      call = std::string(fn) + "(" + arg[1] + ", " + arg[0] + ", &" + nm + "[" + iv + "])";
    else
      call = std::string(fn) + "(" + arg[0] + ", " + arg[1] + ", &" + nm + "[" + iv + "])";

    // A block whose length is not a multiple of 8 keeps its last elements in a
    // scalar tail. It gets its own index, starting at the first multiple of 8:
    // the index of the vector loop goes out of scope with that loop, and a
    // fresh one makes the two ranges provably disjoint. The tail is built
    // before anything is committed, so a rejection leaves no trace.
    std::string tail, tv;
    if (count % 8 != 0) {
      tv = fresh("_kl_i");
      tail = elementwiseExpr(op, ty.getShape(), tv, emptyCtx_);
      if (tail.empty())
        return false;
    }

    decls_ += lmArray(ty.getElementType(), nm, count);
    tensors_[result] = nm;
    body_ += hoist;
    body_ += "    for (int " + iv + " = 0; " + iv + " + 8 <= " + std::to_string(count) + "; " +
             iv + " += 8)\n";
    body_ += "      " + call + ";\n";
    if (!tail.empty()) {
      body_ += "    for (int " + tv + " = " + std::to_string(count / 8 * 8) + "; " + tv + " < " +
               std::to_string(count) + "; ++" + tv + ")\n";
      body_ += "      " + nm + "[" + tv + "] = " + tail + ";\n";
    }
    return true;
  }

  bool emitElementwise(Operation *op) {
    if (op->getNumResults() != 1)
      return failOp(op, "expected a single result");
    Value result = op->getResult(0);
    if (isTensor(result)) {
      if (tryEmitSimd(op))
        return true;
      return emitTensor(result, [&](const std::string &iv) {
        auto ty = cast<RankedTensorType>(result.getType());
        return elementwiseExpr(op, ty.getShape(), iv, emptyCtx_);
      });
    }
    return emitScalar(result, elementwiseExpr(op, {}, "", emptyCtx_));
  }

  std::string elementwiseExpr(Operation *op, ArrayRef<int64_t> dstShape, const std::string &iv,
                              const llvm::DenseMap<Value, std::string> &ctx) {
    StringRef n = op->getName().getStringRef();

    if (isCast(n)) {
      Type srcTy = op->getOperand(0).getType();
      Type dstTy = op->getResult(0).getType();
      if (auto st = dyn_cast<RankedTensorType>(srcTy))
        srcTy = st.getElementType();
      if (auto dt = dyn_cast<RankedTensorType>(dstTy))
        dstTy = dt.getElementType();
      // Only a cast to the very same type is a no-op. Comparing C spellings
      // instead would also fold trunci i8 -> i1, because both are `char`, and
      // drop the low-bit extraction that narrowing to i1 needs.
      if (srcTy == dstTy) {
        std::string arg = elem(op->getOperand(0), dstShape, iv, ctx);
        return arg;
      }
      if (isBitcast(n) && (srcTy.isF32() || dstTy.isF32())) {
        // `((int)(f))` converts the value instead of moving the bits, and a
        // pointer cast would break aliasing, so the bits travel through a
        // union helper. arith.bitcast keeps the width and f32 is the only
        // floating type here, so the integer side is always 32 bits.
        Type intTy = srcTy.isF32() ? dstTy : srcTy;
        if (!intTy.isInteger(32))
          return failed(op, "only a 32-bit integer can be bitcast with f32");
        std::string ct = cType(dstTy);
        if (ct.empty()) {
          failOp(op, "unsupported cast destination type");
          return "";
        }
        std::string arg = elem(op->getOperand(0), dstShape, iv, ctx);
        if (arg.empty())
          return "";
        std::string call;
        if (srcTy.isF32()) {
          needFloatToBits_ = true;
          call = "_kl_bits_of_float(" + arg + ")";
        } else {
          needBitsToFloat_ = true;
          call = "_kl_float_of_bits((unsigned int)(" + arg + "))";
        }
        return "((" + ct + ")(" + call + "))";
      }
      std::string ct = cType(dstTy);
      if (ct.empty()) {
        failOp(op, "unsupported cast destination type");
        return "";
      }
      std::string arg = elem(op->getOperand(0), dstShape, iv, ctx);
      if (arg.empty())
        return "";
      // Same-width integer bitcasts are bit-preserving modulo 2^N in C, so the
      // plain conversion below is exactly the reinterpretation that was asked
      // for; only the narrowing-to-i1 and extension cases need special forms.
      if (dstTy.isInteger(1)) {
        if (n == "arith.trunci")
          // trunci to i1 keeps the low bit, while `!= 0` would keep every bit.
          return "(((" + arg + ") & 1) != 0)";
        // fptosi/fptoui to i1 would be undefined for out-of-range floats, and a
        // nonzero test on the float is the same predicate without the UB.
        return "((" + arg + ") != 0)";
      }
      if (n == "arith.extui" || n == "arith.uitofp") {
        if (isUnsignedInt(srcTy))
          return "((" + ct + ")(" + arg + "))";
        // Converting from the signed C spelling of a signless type would sign
        // extend, so widen through the unsigned spelling of the source width.
        std::string st = unsignedCTy(srcTy);
        if (st.empty()) {
          failOp(op, "unsupported unsigned cast source type");
          return "";
        }
        return "((" + ct + ")((" + st + ")(" + arg + ")))";
      }
      if (n == "arith.extsi" || n == "arith.sitofp") {
        if (!isUnsignedInt(srcTy))
          return "((" + ct + ")(" + arg + "))";
        // The mirror image: a negative value in an unsigned C spelling would be
        // converted, not sign-extended, without the cast through the signed one.
        std::string st = signedCTy(srcTy);
        if (st.empty()) {
          failOp(op, "unsupported signed cast source type");
          return "";
        }
        return "((" + ct + ")((" + st + ")(" + arg + ")))";
      }
      return "((" + ct + ")(" + arg + "))";
    }

    std::string a = elem(op->getOperand(0), dstShape, iv, ctx);
    if (a.empty())
      return "";
    std::string b;
    if (op->getNumOperands() > 1) {
      b = elem(op->getOperand(1), dstShape, iv, ctx);
      if (b.empty())
        return "";
    }

    if (n == "arith.select") {
      std::string c = elem(op->getOperand(2), dstShape, iv, ctx);
      if (c.empty())
        return "";
      return "((" + a + ") ? (" + b + ") : (" + c + "))";
    }

    if (n == "arith.cmpi")
      return cmpiExpr(op, a, b);
    if (n == "arith.cmpf")
      return cmpfExpr(op, a, b);

    if (isMath(n))
      return mathExpr(op, a, b);

    if (n == "arith.addf" || n == "arith.addi")
      return "(" + a + " + " + b + ")";
    if (n == "arith.subf" || n == "arith.subi")
      return "(" + a + " - " + b + ")";
    if (n == "arith.mulf" || n == "arith.muli")
      return "(" + a + " * " + b + ")";
    if (n == "arith.divf")
      return "(" + a + " / " + b + ")";
    if (n == "arith.divsi")
      return "(" + a + " / " + b + ")";
    if (n == "arith.divui") {
      std::string u = unsignedCTy(elemTyOf(op->getResult(0).getType()));
      if (u.empty())
        return failed(op, "expected an integer result");
      return "((" + u + ")(" + a + ") / (" + u + ")(" + b + "))";
    }
    if (n == "arith.remsi")
      return "(" + a + " % " + b + ")";
    if (n == "arith.remui") {
      std::string u = unsignedCTy(elemTyOf(op->getResult(0).getType()));
      if (u.empty())
        return failed(op, "expected an integer result");
      return "((" + u + ")(" + a + ") % (" + u + ")(" + b + "))";
    }
    if (n == "arith.remf") {
      failOp(op, "arith.remf has no device implementation on XPU1");
      return "";
    }
    if (n == "arith.andi")
      return "(" + a + " & " + b + ")";
    if (n == "arith.ori")
      return "(" + a + " | " + b + ")";
    if (n == "arith.xori")
      return "(" + a + " ^ " + b + ")";
    if (n == "arith.shli")
      return "(" + a + " << " + b + ")";
    if (n == "arith.shrsi")
      return "(" + a + " >> " + b + ")";
    if (n == "arith.shrui") {
      std::string u = unsignedCTy(elemTyOf(op->getResult(0).getType()));
      if (u.empty())
        return failed(op, "expected an integer result");
      return "((" + u + ")(" + a + ") >> (" + b + "))";
    }

    // NaN handling: maximumf/minimumf propagate NaN, the *numf variants do not.
    if (n == "arith.maximumf")
      return "(((" + a + ") != (" + a + ")) ? (" + a + ") : ((((" + b + ") != (" + b + ")) ? (" + b +
             ") : (((" + a + ") > (" + b + ")) ? (" + a + ") : (" + b + "))))";
    if (n == "arith.minimumf")
      return "(((" + a + ") != (" + a + ")) ? (" + a + ") : ((((" + b + ") != (" + b + ")) ? (" + b +
             ") : (((" + a + ") < (" + b + ")) ? (" + a + ") : (" + b + "))))";
    if (n == "arith.maxnumf")
      return "fmax(" + a + ", " + b + ")";
    if (n == "arith.minnumf")
      return "fmin(" + a + ", " + b + ")";
    // The device min()/max() helpers are int-only, so spell all integer
    // variants out and let the C type do the work.
    if (n == "arith.maxsi")
      return "(((" + a + ") > (" + b + ")) ? (" + a + ") : (" + b + "))";
    if (n == "arith.minsi")
      return "(((" + a + ") < (" + b + ")) ? (" + a + ") : (" + b + "))";
    if (n == "arith.maxui" || n == "arith.minui") {
      std::string u = unsignedCTy(elemTyOf(op->getResult(0).getType()));
      if (u.empty())
        return failed(op, "expected an integer result");
      const char *cmp = n == "arith.maxui" ? ">" : "<";
      return "(((" + u + ")(" + a + ") " + cmp + " (" + u + ")(" + b + ")) ? (" + a + ") : (" + b + "))";
    }

    failOp(op, "unsupported elementwise operation");
    return "";
  }

  std::string cmpiExpr(Operation *op, const std::string &a, const std::string &b) {
    // arith.cmpi: 0 eq, 1 ne, 2 slt, 3 sle, 4 sgt, 5 sge, 6 ult, 7 ule, 8 ugt, 9 uge.
    static const char *kOps[] = {"==", "!=", "<", "<=", ">", ">=", "<", "<=", ">", ">="};
    int64_t p = intAttr(op, "predicate");
    if (p < 0 || p > 9)
      return failed(op, "unsupported integer predicate");
    const char *o = kOps[p];
    if (p >= 6) {
      std::string u = unsignedCTy(elemTyOf(op->getOperand(0).getType()));
      if (u.empty())
        return failed(op, "expected integer operands");
      return "(((" + u + ")(" + a + ") " + o + " (" + u + ")(" + b + ")) ? 1 : 0)";
    }
    return "(((" + a + ") " + o + " (" + b + ")) ? 1 : 0)";
  }

  std::string cmpfExpr(Operation *op, const std::string &a, const std::string &b) {
    // arith.cmpf predicates. `ordered` predicates are false for NaN and fall
    // out of the plain C comparison; `unordered` ones are true for NaN.
    //   0 false, 1 oeq, 2 ogt, 3 oge, 4 olt, 5 ole, 6 one, 7 ord
    //   8 ueq, 9 ugt, 10 uge, 11 ult, 12 ule, 13 une, 14 uno, 15 true
    int64_t p = intAttr(op, "predicate");
    if (p < 0 || p > 15)
      return failed(op, "unsupported float predicate");
    auto ea = "(" + a + ")";
    auto eb = "(" + b + ")";
    auto nan = "(" + ea + " != " + ea + ") || (" + eb + " != " + eb + ")";
    switch (p) {
    case 0:
      return "0";
    case 15:
      return "1";
    case 1:
      return "(((" + ea + " == " + eb + ")) ? 1 : 0)";
    case 2:
      return "(((" + ea + " > " + eb + ")) ? 1 : 0)";
    case 3:
      return "(((" + ea + " >= " + eb + ")) ? 1 : 0)";
    case 4:
      return "(((" + ea + " < " + eb + ")) ? 1 : 0)";
    case 5:
      return "(((" + ea + " <= " + eb + ")) ? 1 : 0)";
    case 6:  // one: ordered and not equal, so false when either side is NaN
      return "((!((" + nan + ")) && " + ea + " != " + eb + ") ? 1 : 0)";
    case 7:  // ord
      return "((!(" + nan + ")) ? 1 : 0)";
    case 8:
      return "((" + nan + " || " + ea + " == " + eb + ") ? 1 : 0)";
    case 9:
      return "((" + nan + " || " + ea + " > " + eb + ") ? 1 : 0)";
    case 10:
      return "((" + nan + " || " + ea + " >= " + eb + ") ? 1 : 0)";
    case 11:
      return "((" + nan + " || " + ea + " < " + eb + ") ? 1 : 0)";
    case 12:
      return "((" + nan + " || " + ea + " <= " + eb + ") ? 1 : 0)";
    case 13:  // une: unordered or not equal
      return "(((" + ea + " != " + eb + ")) ? 1 : 0)";
    case 14:  // uno
      return "((" + nan + ") ? 1 : 0)";
    }
    return failed(op, "unsupported float predicate");
  }

  std::string mathExpr(Operation *op, const std::string &a, const std::string &b) {
    StringRef n = op->getName().getStringRef();
    // math.h on KL1 maps expf/logf onto the exp2/log2 builtins, so spell the
    // scaling out instead of relying on the header's float wrappers.
    if (n == "math.exp")
      return "__builtin_xpu_exp2f((" + a + ") * 1.4426950408889634f)";
    if (n == "math.exp2")
      return "__builtin_xpu_exp2f(" + a + ")";
    if (n == "math.log")
      return "(__builtin_xpu_log2f(" + a + ") * 0.6931471805599453f)";
    if (n == "math.log2")
      return "__builtin_xpu_log2f(" + a + ")";
    if (n == "math.sqrt")
      return "__builtin_xpu_sqrtf(" + a + ")";
    if (n == "math.rsqrt")
      return "(1.0f / __builtin_xpu_sqrtf(" + a + "))";
    if (n == "math.absf")
      return "fabs(" + a + ")";
    if (n == "math.absi")
      return "(((" + a + ") < 0) ? (-(" + a + ")) : (" + a + "))";
    if (n == "math.powf")
      return "pow(" + a + ", " + b + ")";
    if (n == "math.erf")
      // The header's fast_erf() calls exp(-x*x), and exp() is 2^x on KL1, so
      // the vendor implementation is not the error function.
      return failed(op, "math.erf cannot be emitted: fast_erf in the KL1 math "
                        "header is built on exp(), which is 2^x there");
    if (n == "math.minnumf")
      return "fmin(" + a + ", " + b + ")";
    if (n == "math.maxnumf")
      return "fmax(" + a + ", " + b + ")";
    failOp(op, "unsupported math operation on XPU1");
    return "";
  }

  //===--------------------------------------------------------------------===//
  // Shape and pointer operations
  //===--------------------------------------------------------------------===//

  bool emitMakeRange(Operation *op) {
    int64_t begin = intAttr(op, "start");
    int64_t end = intAttr(op, "end");
    auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!ty || ty.getRank() != 1 || ty.getNumElements() != end - begin)
      return failOp(op, "malformed tt.make_range");
    // These values are indices, so they stay global: element `iv` of this
    // core's slice must carry the index of the element it stands for.
    int64_t chunk = distributed_ ? (end - begin) / kClusterCores : 0;
    return emitTensor(op->getResult(0), [&](const std::string &iv) {
      std::string idx = distributed_ ? "(" + coreOffset(chunk) + " + " + iv + ")" : iv;
      return "(" + std::to_string(begin) + " + " + idx + ")";
    });
  }

  bool emitSplat(Operation *op) {
    Value src = op->getOperand(0);
    if (scalarPtrs_.count(src)) {
      Type elemTy = pointeeOf(src);
      ptrBlocks_[op->getResult(0)] = PtrBlock{scalarPtrs_[src], elemTy, "0", StrideKind::Zero};
      ptrElemTy_[op->getResult(0)] = elemTy;
      return true;
    }
    std::string e = scalarExpr(src, emptyCtx_);
    if (e.empty())
      return false;
    // No LM array yet: every lane can read the scalar expression directly,
    // which is also what lets a 256-bit op use its scalar-vector form. A bulk
    // transfer or a reduction materialises the block on demand.
    splatScalars_[op->getResult(0)] = e;
    return true;
  }

  bool emitBroadcast(Operation *op) {
    Value src = op->getOperand(0);
    Value result = op->getResult(0);
    auto ty = dyn_cast<RankedTensorType>(result.getType());
    if (!ty)
      return failOp(op, "expected a tensor result");
    return emitTensor(result, [&](const std::string &iv) {
      return elem(src, ty.getShape(), iv, emptyCtx_);
    });
  }

  Type pointeeOf(Value ptr) {
    auto it = ptrElemTy_.find(ptr);
    return it == ptrElemTy_.end() ? Type() : it->second;
  }

  // True when every use of `v` is address arithmetic, i.e. a chain that
  // affine() can rebuild from scratch. Such values skip LM materialisation.
  bool isDeferredOffset(Value v) {
    if (!isTensor(v) || !isa<IntegerType>(elemTyOf(v.getType())))
      return false;
    for (Operation *user : v.getUsers()) {
      StringRef n = user->getName().getStringRef();
      if (n == "tt.addptr")
        continue;
      if (user->getNumResults() != 1)
        return false;
      if (n == "tt.broadcast" || n == "tt.expand_dims" || n == "arith.extsi" ||
          n == "arith.extui" || n == "arith.trunci" || n == "arith.index_cast") {
        if (isDeferredOffset(user->getResult(0)))
          continue;
        return false;
      }
      if (isBitcast(n)) {
        if (!isa<IntegerType>(elemTyOf(user->getOperand(0).getType())))
          return false;
        if (isDeferredOffset(user->getResult(0)))
          continue;
        return false;
      }
      if (n == "arith.addi" || n == "arith.subi" || n == "arith.muli") {
        if (isDeferredOffset(user->getResult(0)))
          continue;
        return false;
      }
      return false;
    }
    return true;
  }

  // Recovers `start + stride * i` for a 1-D offset tensor (or a scalar).
  bool affine(Value v, Affine &out) {
    if (!isTensor(v)) {
      out.start = scalarExpr(v, emptyCtx_);
      out.kind = StrideKind::Zero;
      return !out.start.empty();
    }
    auto ty = cast<RankedTensorType>(v.getType());
    if (ty.getRank() != 1)
      return failValue(v, "only 1-D offsets can be turned into a contiguous block");

    Operation *def = v.getDefiningOp();
    if (!def)
      return failValue(v, "tensor block argument offsets are not supported");
    StringRef n = def->getName().getStringRef();

    if (n == "tt.make_range") {
      out.start = std::to_string(intAttr(def, "start"));
      out.kind = StrideKind::One;
      return true;
    }
    if (n == "tt.splat") {
      return affine(def->getOperand(0), out);
    }
    if (n == "tt.broadcast" || n == "tt.expand_dims")
      return affine(def->getOperand(0), out);
    if (n == "arith.extsi" || n == "arith.extui" || n == "arith.index_cast" ||
        n == "arith.trunci")
      return affine(def->getOperand(0), out);
    if (isBitcast(n)) {
      // A same-width integer bitcast leaves the value alone, so the offset
      // chain is unchanged; f32 <-> i32 is not address arithmetic at all.
      if (!isa<IntegerType>(elemTyOf(def->getOperand(0).getType())) ||
          !isa<IntegerType>(elemTyOf(def->getResult(0).getType())))
        return failOp(def, "a bitcast in a pointer offset must be integer-to-integer");
      return affine(def->getOperand(0), out);
    }

    if (n == "arith.addi" || n == "arith.subi" || n == "arith.muli") {
      Affine lhs, rhs;
      if (!affine(def->getOperand(0), lhs) || !affine(def->getOperand(1), rhs))
        return false;
      if (n == "arith.muli") {
        if (lhs.kind != StrideKind::Zero && rhs.kind != StrideKind::Zero)
          return failOp(def, "cannot prove contiguous access (product of two ranges)");
        if (lhs.kind == StrideKind::Zero && rhs.kind == StrideKind::Zero) {
          out.start = "((" + lhs.start + ") * (" + rhs.start + "))";
          out.kind = StrideKind::Zero;
          return true;
        }
        // A range times a stride is a strided access, not a contiguous block.
        Value cst = lhs.kind == StrideKind::Zero ? def->getOperand(0) : def->getOperand(1);
        if (isIntOne(cst)) {
          out = lhs.kind == StrideKind::Zero ? rhs : lhs;
          return true;
        }
        return failOp(def, "cannot prove contiguous access (range scaled by a stride)");
      }
      if (n == "arith.addi") {
        out.start = "((" + lhs.start + ") + (" + rhs.start + "))";
        if (lhs.kind == StrideKind::Zero)
          out.kind = rhs.kind;
        else if (rhs.kind == StrideKind::Zero)
          out.kind = lhs.kind;
        else if (lhs.kind == StrideKind::One && rhs.kind == StrideKind::One)
          out.kind = StrideKind::Other;
        else
          out.kind = StrideKind::Other;
        return true;
      }
      out.start = "((" + lhs.start + ") - (" + rhs.start + "))";
      if (rhs.kind == StrideKind::Zero)
        out.kind = lhs.kind;
      else if (lhs.kind == StrideKind::One && rhs.kind == StrideKind::One)
        out.kind = StrideKind::Zero;
      else
        out.kind = StrideKind::Other;
      return true;
    }

    return failOp(def, "cannot prove the pointer offset is contiguous");
  }

  bool emitAddPtr(Operation *op) {
    Value base = op->getOperand(0);
    Value offs = op->getOperand(1);
    Value result = op->getResult(0);

    Affine off;
    if (!affine(offs, off))
      return false;

    auto blockIt = ptrBlocks_.find(base);
    if (blockIt != ptrBlocks_.end()) {
      PtrBlock nb = blockIt->second;
      if (nb.kind == StrideKind::Zero) {
        // Every lane of a splat pointer block holds the same address, so the
        // sum takes the offset's stride: splat(p) + range is a contiguous block.
        if (off.kind == StrideKind::Other)
          return failOp(op, "cannot prove the pointer tensor is contiguous");
        nb.kind = off.kind;
      } else if (off.kind != StrideKind::Zero) {
        return failOp(op, "pointer-tensor + tensor offset is not a contiguous block");
      }
      nb.start = "((" + nb.start + ") + (" + off.start + "))";
      ptrBlocks_[result] = nb;
      ptrElemTy_[result] = nb.elemTy;
      return true;
    }

    auto scalarIt = scalarPtrs_.find(base);
    if (scalarIt == scalarPtrs_.end())
      return failOp(op, "unsupported pointer operand");
    Type elemTy = pointeeOf(base);
    if (!elemTy)
      return failOp(op, "unknown pointee type");
    if (isTensor(result)) {
      if (off.kind == StrideKind::Other)
        return failOp(op, "cannot prove the pointer tensor is contiguous");
      ptrBlocks_[result] = PtrBlock{scalarIt->second, elemTy, off.start, off.kind};
      ptrElemTy_[result] = elemTy;
      return true;
    }
    scalarPtrs_[result] = offsetAddr(scalarIt->second, off.start);
    ptrElemTy_[result] = elemTy;
    return true;
  }

  // Widening the offset here is what keeps the address arithmetic overflow
  // safe: `start` is evaluated in the width the IR gave it, but the core slice
  // and the element index are added in 64 bits, so a block near the top of the
  // 32-bit address space cannot overflow a signed int. Truncating a 64-bit byte
  // offset back to 32 bits is exact for every address inside it, and an i32
  // offset keeps Triton's own wrap-around because the cast happens after the
  // expression has been evaluated.
  static std::string offsetAddr(const std::string &base, const std::string &start) {
    if (start == "0")
      return base;
    return base + " + (long long)(" + start + ")";
  }

  std::string blockAddress(const PtrBlock &b) { return offsetAddr(b.base, b.start); }

  bool emitLoad(Operation *op) {
    if (op->getNumOperands() > 1)
      return emitMaskedLoad(op);
    Value ptr = op->getOperand(0);
    Value result = op->getResult(0);

    auto it = ptrBlocks_.find(ptr);
    if (it == ptrBlocks_.end()) {
      if (scalarPtrs_.count(ptr))
        return failOp(op, "scalar loads are not supported: XPU1 GM2LM reads in 32-byte "
                          "units and would over-read past the element");
      return failOp(op, "unsupported load address");
    }
    PtrBlock &b = it->second;
    if (b.kind != StrideKind::One)
      return failOp(op, "pointer tensor is not a contiguous block");
    auto ty = dyn_cast<RankedTensorType>(result.getType());
    if (!ty)
      return failOp(op, "expected a tensor result");
    if (cSize(b.elemTy) == 0)
      return failOp(op, "unsupported pointee type");
    if (cType(ty.getElementType()) != cType(b.elemTy))
      return failOp(op, "loaded type does not match the pointer element type; the "
                        "reference emitter copies raw bytes");

    int64_t elems = ty.getNumElements();
    if (!localCount(elems, "load", op, elems))
      return false;
    int64_t bytes = elems * cSize(b.elemTy);
    if (bytes % 32 != 0)
      return failOp(op, Twine("load size ") + Twine(bytes) +
                           " bytes is not a multiple of 32; pad the block or the "
                           "reference emitter cannot express this access");

    std::string nm = fresh("v");
    decls_ += lmArray(ty.getElementType(), nm, elems);
    tensors_[result] = nm;
    std::string addr = blockAddress(b);
    if (distributed_)
      addr = "(" + addr + ") + " + coreOffset(elems);
    body_ += "    GM2LM(" + addr + ", " + nm + ", " + std::to_string(bytes) + ");\n";
    return true;
  }

  bool emitStore(Operation *op) {
    if (op->getNumOperands() > 2)
      return emitMaskedStore(op);
    Value ptr = op->getOperand(0);
    Value value = op->getOperand(1);

    auto it = ptrBlocks_.find(ptr);
    if (it != ptrBlocks_.end()) {
      PtrBlock &b = it->second;
      if (b.kind != StrideKind::One)
        return failOp(op, "pointer tensor is not a contiguous block");
      if (!materialiseSplat(value))
        return false;
      auto arr = tensors_.find(value);
      if (arr == tensors_.end()) {
        if (scalarTensors_.count(value))
          return failOp(op, "a one-element block is folded into a scalar when it is "
                            "computed, so it has no LM array to store; use a scalar store");
        return failOp(op, "store value was not materialised");
      }
      auto vty = cast<RankedTensorType>(value.getType());
      if (cType(vty.getElementType()) != cType(b.elemTy))
        return failOp(op, "store value type does not match the pointer element type");
      int64_t elems = vty.getNumElements();
      if (!localCount(elems, "store", op, elems))
        return false;
      int64_t bytes = elems * cSize(b.elemTy);
      std::string addr = blockAddress(b);
      if (distributed_)
        addr = "(" + addr + ") + " + coreOffset(elems);
      body_ += "    LM2GM(" + arr->second + ", " + addr + ", " + std::to_string(bytes) + ");\n";
      return true;
    }

    auto sp = scalarPtrs_.find(ptr);
    if (sp == scalarPtrs_.end())
      return failOp(op, "unsupported store address");
    if (isTensor(value))
      return failOp(op, "tensor store through a scalar pointer is not supported");

    std::string e = scalarExpr(value, emptyCtx_);
    if (e.empty())
      return false;
    std::string ct = cType(pointeeOf(ptr));
    std::string tmp = fresh("_kl_st");
    decls_ += lmArray(pointeeOf(ptr), tmp, 8);
    // A reduced scalar is identical on every core, so exactly one core writes
    // it: concurrent writes to one global address are not supported on XPU1.
    std::string store = tmp + "[0] = (" + ct + ")(" + e + ");\n" +
                        "LM2GM(" + tmp + ", " + sp->second + ", " +
                        std::to_string(cSize(pointeeOf(ptr))) + ");\n";
    if (distributed_) {
      body_ += "    if (_kl_core == 0) {\n";
      body_ += "      " + store;
      body_ += "    }\n";
    } else {
      body_ += "    " + store;
    }
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Masked access
  //===--------------------------------------------------------------------===//

  // Emits a walk over the mask of a block that is partly dead. XPU1 moves
  // memory in 32-byte units and cannot touch a single element, so a run of
  // `chunk` consecutive live elements shares one 32-byte transfer and only a
  // broken run pays one transfer per element. `onDead`, `onRun` and `onOne`
  // emit the statements for a masked-out element, a full run and a single live
  // element; each is handed the index expression to use.
  template <typename MaskFn, typename DeadFn, typename RunFn, typename OneFn>
  void emitMaskWalk(int64_t n, int64_t chunk, MaskFn maskAt, DeadFn onDead, RunFn onRun,
                    OneFn onOne) {
    std::string s = fresh("_kl_s"), r = fresh("_kl_r"), j = fresh("_kl_j");
    std::string ns = std::to_string(n), cs = std::to_string(chunk);
    body_ += "    for (int " + s + " = 0; " + s + " < " + ns + ";) {\n";
    body_ += "      if (!(" + maskAt(s) + ")) {\n";
    onDead(s);
    body_ += "        ++" + s + ";\n        continue;\n      }\n";
    body_ += "      int " + r + " = 1;\n";
    body_ += "      while (" + r + " < " + cs + " && " + s + " + " + r + " < " + ns + " && (" +
             maskAt(s + " + " + r) + ")) ++" + r + ";\n";
    body_ += "      if (" + r + " == " + cs + ") {\n";
    onRun(s);
    body_ += "        " + s + " += " + cs + ";\n";
    body_ += "      } else {\n";
    body_ += "        for (int " + j + " = 0; " + j + " < " + r + "; ++" + j + ") {\n";
    onOne(s + " + " + j);
    body_ += "        }\n";
    body_ += "        " + s + " += " + r + ";\n";
    body_ += "      }\n";
    body_ += "    }\n";
  }

  bool emitMaskedLoad(Operation *op) {
    Value ptr = op->getOperand(0);
    Value mask = op->getOperand(1);
    Value result = op->getResult(0);
    Value other = op->getNumOperands() > 2 ? op->getOperand(2) : Value();

    auto ty = dyn_cast<RankedTensorType>(result.getType());
    if (!ty)
      return failOp(op, "a masked load must produce a tensor");
    auto it = ptrBlocks_.find(ptr);
    if (it == ptrBlocks_.end())
      return failOp(op, "a masked load needs a contiguous pointer block; a masked "
                        "scalar load would need a byte-granular GM read, which XPU1 "
                        "does not have");
    PtrBlock &b = it->second;
    if (b.kind == StrideKind::Other)
      return failOp(op, "pointer tensor is not a contiguous block");
    int64_t esz = cSize(b.elemTy);
    if (esz == 0 || esz > 32)
      return failOp(op, "unsupported pointee type");
    std::string ct = cType(ty.getElementType());
    if (ct != cType(b.elemTy))
      return failOp(op, "loaded type does not match the pointer element type; the "
                        "reference emitter copies raw bytes");

    int64_t count = ty.getNumElements();
    if (!localCount(count, "load", op, count))
      return false;

    std::string nm = fresh("v");
    decls_ += lmArray(ty.getElementType(), nm, count);
    tensors_[result] = nm;

    // 32 bytes of scratch, typed so no LM pointer cast is needed: one transfer
    // always lands exactly `chunk` elements. GM2LM wants a 32-byte aligned LM
    // destination.
    int64_t chunk = 32 / esz;
    std::string cc = fresh("_kl_c");
    decls_ += lmArray(ty.getElementType(), cc, chunk);

    std::string addr = blockAddress(b);
    if (b.kind == StrideKind::One && distributed_)
      addr = "(" + addr + ") + " + coreOffset(count);

    bool ok = true;
    auto at = [&](Value v, const std::string &i) {
      std::string e = elem(v, ty.getShape(), i, emptyCtx_);
      if (e.empty())
        ok = false;
      return e;
    };
    // Element i of a contiguous block sits i elements after its first one; a
    // splat block holds the same address in every lane.
    auto addrAt = [&](const std::string &i) {
      return b.kind == StrideKind::One ? "(" + addr + ") + (" + i + ")" : addr;
    };

    emitMaskWalk(
        count, chunk, [&](const std::string &i) { return at(mask, i); },
        // A masked-out lane keeps `other`, or zero: TTIR leaves it undefined
        // without `other`, and zero keeps the kernel deterministic.
        [&](const std::string &i) {
          body_ += "        " + nm + "[" + i + "] = " + (other ? at(other, i) : "0") + ";\n";
        },
        [&](const std::string &i) {
          std::string k = fresh("_kl_k");
          body_ += "        GM2LM(" + addrAt(i) + ", " + cc + ", 32);\n";
          body_ += "        for (int " + k + " = 0; " + k + " < " + std::to_string(chunk) +
                   "; ++" + k + ")\n";
          body_ += "          " + nm + "[" + i + " + " + k + "] = " + cc + "[" + k + "];\n";
        },
        [&](const std::string &i) {
          // Reading the element alone is impossible, so the 32-byte window
          // starts at the element and only its first slot is kept. That reads
          // up to 28 bytes past the element, which stays inside the page a
          // device allocation ends on but may cross it for the very last one;
          // there is no smaller transfer on XPU1.
          body_ += "          GM2LM(" + addrAt(i) + ", " + cc + ", 32);\n";
          body_ += "          " + nm + "[" + i + "] = " + cc + "[0];\n";
        });
    return ok;
  }

  bool emitMaskedStore(Operation *op) {
    Value ptr = op->getOperand(0);
    Value value = op->getOperand(1);
    Value mask = op->getOperand(2);

    auto pty = dyn_cast<RankedTensorType>(ptr.getType());
    auto it = ptrBlocks_.find(ptr);
    if (!pty || it == ptrBlocks_.end())
      return failOp(op, "a masked store needs a contiguous pointer block; a masked "
                        "scalar store would need a byte-granular GM write, which "
                        "XPU1 does not have");
    PtrBlock &b = it->second;
    if (b.kind == StrideKind::Other)
      return failOp(op, "pointer tensor is not a contiguous block");
    int64_t esz = cSize(b.elemTy);
    if (esz == 0 || esz > 32)
      return failOp(op, "unsupported pointee type");
    if (cType(elemTyOf(value.getType())) != cType(b.elemTy))
      return failOp(op, "store value type does not match the pointer element type");

    int64_t count = pty.getNumElements();
    if (!localCount(count, "store", op, count))
      return false;

    int64_t chunk = 32 / esz;
    std::string cc = fresh("_kl_c");
    decls_ += lmArray(b.elemTy, cc, chunk);

    std::string addr = blockAddress(b);
    if (b.kind == StrideKind::One && distributed_)
      addr = "(" + addr + ") + " + coreOffset(count);

    bool ok = true;
    auto at = [&](Value v, const std::string &i) {
      std::string e = elem(v, pty.getShape(), i, emptyCtx_);
      if (e.empty())
        ok = false;
      return e;
    };
    auto addrAt = [&](const std::string &i) {
      return b.kind == StrideKind::One ? "(" + addr + ") + (" + i + ")" : addr;
    };

    emitMaskWalk(
        count, chunk, [&](const std::string &i) { return at(mask, i); },
        [&](const std::string &) {},  // masked out: nothing is written
        [&](const std::string &i) {
          std::string k = fresh("_kl_k");
          body_ += "        for (int " + k + " = 0; " + k + " < " + std::to_string(chunk) +
                   "; ++" + k + ")\n";
          body_ += "          " + cc + "[" + k + "] = " + at(value, i + " + " + k) + ";\n";
          body_ += "        LM2GM(" + cc + ", " + addrAt(i) + ", 32);\n";
        },
        [&](const std::string &i) {
          // Only the live element is written: LM2GM has no alignment limit on
          // the GM side and takes any size, so the neighbours stay untouched.
          body_ += "          " + cc + "[0] = " + at(value, i) + ";\n";
          body_ += "          LM2GM(" + cc + ", " + addrAt(i) + ", " + std::to_string(esz) +
                   ");\n";
        });
    return ok;
  }

  //===--------------------------------------------------------------------===//
  // Reduction
  //===--------------------------------------------------------------------===//

  bool emitReduce(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failOp(op, "only single-input, single-result reductions are supported");
    Value src = op->getOperand(0);
    Value result = op->getResult(0);
    auto srcTy = dyn_cast<RankedTensorType>(src.getType());
    if (!srcTy || srcTy.getRank() != 1 || intAttr(op, "axis", 0) != 0)
      return failOp(op, "only reductions of a 1-D tensor along axis 0 are supported");
    if (isTensor(result))
      return failOp(op, "reductions that keep a tensor result are not supported");

    if (!materialiseSplat(src))
      return false;
    auto arr = tensors_.find(src);
    if (arr == tensors_.end()) {
      // A one-element block is already a scalar, so the reduction is a no-op.
      if (auto st = scalarTensors_.find(src); st != scalarTensors_.end()) {
        scalars_[result] = st->second;
        return true;
      }
      return failOp(op, "reduction input was not materialised");
    }

    Region &region = op->getRegion(0);
    if (!llvm::hasSingleElement(region))
      return failOp(op, "expected a single-block combine region");
    Block &combine = region.front();
    if (combine.getNumArguments() != 2)
      return failOp(op, "expected a two-argument combine region");

    std::string ct = cType(elemTyOf(result.getType()));
    if (ct.empty())
      return failOp(op, "unsupported reduction result type");
    int64_t n = srcTy.getNumElements();
    int64_t count = n;
    if (!localCount(n, "reduction input", op, count))
      return false;
    std::string acc = fresh("_kl_acc");
    std::string rhs = fresh("_kl_rhs");
    std::string it = fresh("_kl_r");

    decls_ += "    " + ct + " " + acc + ";\n";
    decls_ += "    " + ct + " " + rhs + ";\n";
    scalars_[result] = acc;

    auto combineExpr = [&](const std::string &lhsName, const std::string &rhsName) {
      llvm::DenseMap<Value, std::string> ctx;
      ctx[combine.getArgument(0)] = lhsName;
      ctx[combine.getArgument(1)] = rhsName;
      for (Operation &rop : combine) {
        if (rop.getName().getStringRef() != "tt.reduce.return")
          continue;
        if (rop.getNumOperands() != 1)
          return failed(&rop, "expected a single combine value");
        return scalarExpr(rop.getOperand(0), ctx);
      }
      return failed(op, "combine region has no tt.reduce.return");
    };

    std::string term = combineExpr(acc, rhs);
    if (term.empty())
      return false;

    // Partial reduction over this core's slice.
    body_ += "    " + acc + " = " + arr->second + "[0];\n";
    body_ += "    for (int " + it + " = 1; " + it + " < " + std::to_string(count) + "; ++" + it +
             ") {\n";
    body_ += "      " + rhs + " = " + arr->second + "[" + it + "];\n";
    body_ += "      " + acc + " = " + term + ";\n";
    body_ += "    }\n";

    if (!distributed_)
      return true;

    // Cross-core: publish one partial per core through cluster shared memory
    // and combine all 16 on every core, so the result stays replicated.
    // SM transfers move data in 64-byte units and every SM address must be
    // 64-byte aligned, so each core owns a padded 64-byte slot and the whole
    // array is read back with a single SM2LM.
    int64_t csz = cSize(elemTyOf(result.getType()));
    if (csz == 0)
      return failOp(op, "unsupported reduction element type");
    int64_t slotBytes = 64;
    int64_t perCore = slotBytes / csz;
    int64_t slots = kClusterCores * perCore;
    std::string sm = fresh("_kl_sm");
    std::string part = fresh("_kl_part");
    std::string all = fresh("_kl_all");

    smDecls_ += "  __shared__ " + ct + " " + sm + "[" + std::to_string(slots) +
                "] __attribute__((aligned(64)));\n";
    smBytes_ += slots * csz;
    decls_ += lmArray(elemTyOf(result.getType()), part, perCore);
    decls_ += lmArray(elemTyOf(result.getType()), all, slots);

    body_ += "    " + part + "[0] = " + acc + ";\n";
    body_ += "    LM2SM(" + part + ", " + sm + " + " + coreOffset(perCore) + ", " +
             std::to_string(slotBytes) + ");\n";
    body_ += "    sync();\n";
    body_ += "    SM2LM(" + sm + ", " + all + ", " + std::to_string(slots * csz) + ");\n";
    // Keep the second barrier: without it a core could start overwriting its
    // slot in the next iteration while another core is still reading.
    body_ += "    sync();\n";
    body_ += "    " + acc + " = " + all + "[0];\n";
    for (int64_t j = 1; j < kClusterCores; ++j) {
      body_ += "    " + rhs + " = " + all + "[" + std::to_string(j * perCore) + "];\n";
      body_ += "    " + acc + " = " + term + ";\n";
    }
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Assembly
  //===--------------------------------------------------------------------===//

  std::string assemble() const {
    std::string out;
    out += "// Generated by the Triton Kunlun reference emitter (TTIR -> XPU-C).\n";
    if (distributed_)
      out += "// One program id runs on every core of one cluster; each core owns a\n"
             "// contiguous 1/16 slice of every tensor and reductions are combined\n"
             "// through cluster shared memory. Requires exactly 16 cores per cluster,\n"
             "// which is why distributed mode rejects any other num_warps.\n";
    else
      out += "// One core runs one whole program id; every tensor lives in LM; global\n"
             "// traffic is bulk DMA over contiguous blocks.\n";
    out += "#include \"xpu/kernel/cluster_header.h\"\n";
    out += "#include \"xpu/kernel/math.h\"\n\n";
    if (needFloatToBits_)
      out += kFloatToBitsHelper;
    if (needBitsToFloat_)
      out += kBitsToFloatHelper;
    out += sig_;
    out += smDecls_;
    if (distributed_) {
      out += "  int _kl_core = core_id();\n";
      out += "  for (int _kl_pid = cluster_id(); _kl_pid < _kl_np; _kl_pid += cluster_num()) {\n";
    } else {
      out += "  int _kl_lcore = core_id();\n";
      out += "  if (_kl_lcore >= core_num()) return;\n";
      out += "  int _kl_nw = cluster_num() * core_num();\n";
      out += "  for (int _kl_pid = cluster_id() * core_num() + _kl_lcore; _kl_pid < _kl_np;\n";
      out += "       _kl_pid += _kl_nw) {\n";
    }
    out += "    int _kl_p0 = _kl_g0 > 0 ? _kl_pid % _kl_g0 : 0;\n";
    out += "    int _kl_q0 = _kl_g0 > 0 ? _kl_pid / _kl_g0 : 0;\n";
    out += "    int _kl_p1 = _kl_g1 > 0 ? _kl_q0 % _kl_g1 : 0;\n";
    out += "    int _kl_p2 = _kl_g1 > 0 ? _kl_q0 / _kl_g1 : 0;\n";
    out += decls_;
    out += body_;
    out += "  }\n";
    out += "}\n";
    return out;
  }

  ModuleOp module_;
  int64_t lmBudget_;
  bool distributed_ = false;
  Operation *func_ = nullptr;
  std::string entryName_;
  std::string sig_;
  std::string smDecls_;
  std::string decls_;
  std::string body_;
  std::string error_;
  int64_t lmBytes_ = 0;
  int64_t smBytes_ = 0;
  int nextId_ = 0;
  // Whether a kernel bitcasts f32 to an integer or back, which needs a union
  // helper for that direction.
  bool needFloatToBits_ = false;
  bool needBitsToFloat_ = false;

  llvm::DenseMap<Value, std::string> scalars_;
  llvm::DenseMap<Value, std::string> scalarTensors_;
  // Tensors that are a splat of a scalar, still held as that scalar expression.
  llvm::DenseMap<Value, std::string> splatScalars_;
  llvm::DenseMap<Value, std::string> tensors_;
  llvm::DenseMap<Value, PtrBlock> ptrBlocks_;
  llvm::DenseMap<Value, std::string> scalarPtrs_;
  llvm::DenseMap<Value, Type> ptrElemTy_;
  llvm::DenseMap<Value, std::string> emptyCtx_;
};

} // namespace

bool emitXPUCSource(ModuleOp module, int64_t lmBudgetBytes, bool distributed,
                    std::string &source, std::string &entryName,
                    std::string &error) {
  XPUCEmitter emitter(module, lmBudgetBytes, distributed);
  return emitter.run(source, entryName, error);
}

} // namespace mlir::triton::kunlun
