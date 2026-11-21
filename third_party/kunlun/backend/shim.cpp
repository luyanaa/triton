// C ABI over libxpujitc / libxpurt, loaded by jitc.py through ctypes.
//
// Everything vendor-specific stays on this side of the boundary: Python only
// ever sees byte buffers and integers, so it never needs the layout of
// xpurtc::Kernel or the C++ name mangling of XPUStream.
//
// Build (done at runtime by jitc.py, recorded here for reference):
//   g++ -std=c++17 -O1 -fPIC -shared -o shim.so shim.cpp \
//       -I$XTDK/include -I$XRE/include \
//       -L$XTDK/shlib -lxpujitc -L$XRE/so -lxpurt \
//       -Wl,-rpath,$XTDK/shlib -Wl,-rpath,$XRE/so

#include "xpu/runtime.h"
#include "xpu/xpurtc.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {

// clang and ld.lld are spawned by CompileContext, but the JIT driver itself is
// not documented as reentrant; Triton compiles from several threads.
std::mutex g_compile_mutex;
// Per-thread so a concurrent failure cannot invalidate a message another
// thread is still reading.
thread_local std::string g_error;

void set_error(const std::string &msg) { g_error = msg; }

void set_error(const char *what, const std::string &detail) {
  g_error = std::string(what) + ": " + detail;
}

} // namespace

extern "C" {

// The returned pointer is owned by the shim and stays valid until the next
// call on any thread; callers must copy or use it immediately.
const char *kl_last_error(void) { return g_error.c_str(); }

void kl_free(void *ptr) { std::free(ptr); }

// Compiles `source` for XPU1 and returns a heap copy of the device code so the
// caller can cache it independently of this CompileContext. `mangled` receives
// the entry symbol the launcher must hand to launch_kernel.
int kl_compile(const char *source, const char *entry, void **code,
               unsigned int *code_len, unsigned long long *code_hash,
               char *mangled, unsigned int mangled_cap, int *is_cdnn) {
  if (source == nullptr || entry == nullptr || code == nullptr ||
      code_len == nullptr || code_hash == nullptr || mangled == nullptr ||
      is_cdnn == nullptr) {
    set_error("kl_compile: null argument");
    return -1;
  }

  std::lock_guard<std::mutex> guard(g_compile_mutex);
  try {
    // 1 selects --xpu-arch=xpu1; libxpujitc spells the flag as
    // "--xpu-arch=xpu" + the decimal arch.
    xpurtc::CompileContext ctx(1);
    ctx.add_source(std::string(source), xpurtc::SourceKind::XPU,
                   "triton_kunlun");
    xpurtc::Kernel kernel = ctx.get_by_short_name(entry);
    if (!kernel.is_valid()) {
      // One __global__ per translation unit, so this is unambiguous; it also
      // covers a short-name map that keys on something other than the MLIR
      // symbol.
      kernel = ctx.get_kernel();
    }
    if (!kernel.is_valid()) {
      set_error("no kernel named `" + std::string(entry) +
                "` in the generated source");
      return 2;
    }

    const uint8_t *device_code = kernel.code();
    uint32_t size = kernel.size();
    if (device_code == nullptr || size == 0) {
      set_error("the JIT produced an empty kernel image");
      return 3;
    }

    const std::string &name = kernel.mangled_name();
    if (name.size() + 1 > mangled_cap) {
      set_error("mangled entry name is too long");
      return 4;
    }

    void *copy = std::malloc(size);
    if (copy == nullptr) {
      set_error("out of host memory copying the kernel image");
      return 5;
    }
    std::memcpy(copy, device_code, size);

    std::memcpy(mangled, name.data(), name.size());
    mangled[name.size()] = '\0';
    *code = copy;
    *code_len = size;
    *code_hash = kernel.hash();
    *is_cdnn = kernel.is_cdnn_kernel() ? 1 : 0;
    return 0;
  } catch (const std::exception &e) {
    set_error("xpu JIT compilation failed", e.what());
    return 6;
  } catch (...) {
    set_error("xpu JIT compilation failed with an unknown exception");
    return 7;
  }
}

int kl_launch(const void *code, unsigned int code_len,
              unsigned long long code_hash, int ncluster, int ncore,
              void *stream, const void *params, unsigned int params_len,
              int use_cdnn, const char *mangled) {
  if (code == nullptr || code_len == 0 || mangled == nullptr) {
    set_error("kl_launch: invalid kernel image");
    return -1;
  }
  try {
    int ret = xpurtc::launch_kernel(code, code_len, code_hash, ncluster, ncore,
                                    (XPUStream)stream, params, params_len,
                                    use_cdnn != 0, mangled);
    if (ret != 0) {
      set_error("launch_kernel returned " + std::to_string(ret));
      return ret;
    }
    return 0;
  } catch (const std::exception &e) {
    set_error("launch failed", e.what());
    return 1;
  } catch (...) {
    set_error("launch failed with an unknown exception");
    return 1;
  }
}

} // extern "C"
