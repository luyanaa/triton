"""Lazy ctypes bridge between the Triton backend and the Kunlun JIT.

``libxpujitc`` is a thin driver: it execs the toolchain's ``clang`` and
``ld.lld`` and links only against ``libxpurt`` (verified with ``DT_NEEDED``),
so loading it does *not* pull LLVM 10 into the process.  We still load it with
``RTLD_LOCAL`` and only from inside this module, so nothing else in the process
can accidentally bind to vendor symbols.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from triton.runtime.cache import default_cache_dir

_HERE = Path(__file__).resolve().parent
_SHIM_SRC = _HERE / "shim.cpp"

# A kernel that exercises clang, ld.lld and the XPU1 memory model end to end.
_PREFLIGHT_SOURCE = """\
#include "xpu/kernel/cluster_header.h"
__global__ void kl_preflight(float* out) {
  __local__ float v[8];
  v[0] = 1.0f;
  LM2GM(v, out, 4);
}
"""


class JitcError(RuntimeError):
    pass


class KernelImage:
    """Device image plus everything launch_kernel needs to know about it."""

    __slots__ = ("code", "hash", "mangled", "is_cdnn")

    def __init__(self, code: bytes, khash: int, mangled: str, is_cdnn: bool):
        self.code = code
        self.hash = khash
        self.mangled = mangled
        self.is_cdnn = is_cdnn


def _first_dir(*candidates: str) -> str:
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return ""


def xtdk_root() -> str:
    """Root of the XTDK tree that owns clang, libxpujitc and the device headers."""
    root = _first_dir(os.environ.get("KUNLUN_XTDK", ""), os.environ.get("CLANG_PATH", ""))
    if not root:
        raise JitcError("cannot locate the XTDK tree: set KUNLUN_XTDK or CLANG_PATH to the "
                        "directory that contains bin/clang and shlib/libxpujitc.so")
    return root


def xre_root() -> str:
    """Root of the XPURT runtime drop (include/xpu/runtime.h, so/libxpurt.so)."""
    root = _first_dir(os.environ.get("KUNLUN_XRE", ""), os.environ.get("RT_DIR", ""))
    if not root:
        raise JitcError("cannot locate the XPURT runtime: set KUNLUN_XRE or RT_DIR to the "
                        "directory that contains include/xpu/runtime.h and so/libxpurt.so")
    return root


def _jitc_library() -> Path:
    override = os.environ.get("KUNLUN_XPUJITC", "")
    if override:
        path = Path(override)
        if not path.is_file():
            raise JitcError(f"KUNLUN_XPUJITC points at {override}, which does not exist")
        return path.resolve()
    root = Path(xtdk_root())
    for candidate in (root / "shlib" / "libxpujitc.so", root / "lib" / "libxpujitc.so",
                      root / "lib64" / "libxpujitc.so"):
        if candidate.is_file():
            return candidate
    raise JitcError(f"libxpujitc.so not found under {root}; set KUNLUN_XPUJITC to its full path")


def _lib_dir(name: str, root: Path) -> Path:
    for sub in ("shlib", "so", "lib", "lib64"):
        candidate = root / sub
        if candidate.is_dir() and any(candidate.glob(f"lib{name}.so*")):
            return candidate
    raise JitcError(f"cannot find lib{name}.so under {root}")


def _build_shim(out_dir: Path) -> Path:
    xtdk, xre = Path(xtdk_root()), Path(xre_root())
    include_dirs = [xtdk / "include", xre / "include"]
    for inc in include_dirs:
        if not (inc / "xpu").is_dir():
            raise JitcError(f"{inc} does not contain the xpu headers")

    jitc_dir = _jitc_library().parent
    rt_dir = _lib_dir("xpurt", xre)

    # The rpath is baked in, so a different SDK location needs a different shim.
    source = _SHIM_SRC.read_text()
    key = hashlib.sha256((source + os.environ.get("CXX", "g++") + str(jitc_dir) +
                          str(rt_dir)).encode()).hexdigest()[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = out_dir / f"shim-{key}.so"
    if so_path.is_file():
        return so_path

    tmp_so = out_dir / f".{so_path.name}.{os.getpid()}.tmp"
    cmd = [
        os.environ.get("CXX", "g++"), "-std=c++17", "-O1", "-fPIC", "-shared", "-o",
        str(tmp_so), str(_SHIM_SRC),
        *[f"-I{d}" for d in include_dirs],
        f"-L{jitc_dir}", "-lxpujitc", f"-L{rt_dir}", "-lxpurt",
        # DT_RPATH, not DT_RUNPATH: libxpurt may dlopen further vendor
        # libraries (libxpuml) that are only findable through an inherited path.
        f"-Wl,-rpath,{jitc_dir}", f"-Wl,-rpath,{rt_dir}", "-Wl,--disable-new-dtags",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
    if proc.returncode != 0:
        tmp_so.unlink(missing_ok=True)
        raise JitcError("failed to build the Kunlun JIT shim:\n  " + " ".join(cmd) + "\n" +
                        proc.stdout + proc.stderr)
    os.replace(tmp_so, so_path)
    return so_path


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.kl_last_error.restype = ctypes.c_char_p
    lib.kl_last_error.argtypes = []
    lib.kl_free.restype = None
    lib.kl_free.argtypes = [ctypes.c_void_p]
    lib.kl_compile.restype = ctypes.c_int
    lib.kl_compile.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_char_p,
        ctypes.c_uint, ctypes.POINTER(ctypes.c_int)
    ]
    lib.kl_launch.restype = ctypes.c_int
    lib.kl_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_char_p
    ]
    return lib


def _load(so_path: Path) -> ctypes.CDLL:
    # RTLD_LOCAL: the vendor driver must not become a symbol source for the
    # rest of the process.
    return _bind(ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL))


def _compile(lib: ctypes.CDLL, source: str, entry: str) -> KernelImage:
    code = ctypes.c_void_p()
    length = ctypes.c_uint()
    khash = ctypes.c_ulonglong()
    mangled = ctypes.create_string_buffer(256)
    is_cdnn = ctypes.c_int()
    rc = lib.kl_compile(source.encode(), entry.encode(), ctypes.byref(code), ctypes.byref(length),
                        ctypes.byref(khash), mangled, ctypes.c_uint(len(mangled)),
                        ctypes.byref(is_cdnn))
    if rc != 0:
        msg = lib.kl_last_error()
        raise JitcError(msg.decode("utf-8", "replace") if msg else
                        f"JIT compilation failed with code {rc}")
    try:
        blob = ctypes.string_at(code, length.value)
    finally:
        lib.kl_free(code)
    return KernelImage(blob, khash.value, mangled.value.decode(), bool(is_cdnn.value))


def _preflight(so_path: Path) -> None:
    """Compile once in a child process.

    libxpujitc reports toolchain failures through exceptions in the builds we
    have seen, but a stray exit() inside it would take the whole training
    process down with no usable traceback.  A dead child is a clear error
    instead.  The timeout covers the case where the driver blocks forever
    (xpu_malloc is known to hang on bad requests), which no traceback would
    ever explain.
    """
    if os.environ.get("KUNLUN_SKIP_PREFLIGHT", "0") == "1":
        return
    timeout = float(os.environ.get("KUNLUN_PREFLIGHT_TIMEOUT", "300"))
    script = ("import sys\n"
              f"sys.path[:0] = {sys.path!r}\n"
              "from pathlib import Path\n"
              "from triton.backends.kunlun import jitc\n"
              f"img = jitc._load(Path({str(so_path)!r}))\n"
              f"img = jitc._compile(img, jitc._PREFLIGHT_SOURCE, 'kl_preflight')\n"
              "print(img.hash, img.mangled, len(img.code))\n")
    try:
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        raise JitcError(
            "the Kunlun JIT preflight compile did not finish within %gs, so Triton will not "
            "load libxpujitc in-process.\nRaise KUNLUN_PREFLIGHT_TIMEOUT if the toolchain is "
            "just slow, or set KUNLUN_SKIP_PREFLIGHT=1 to bypass the check." % timeout) from None
    if proc.returncode != 0:
        raise JitcError(
            "the Kunlun JIT driver crashed during the preflight compile, so Triton will not "
            "load it in-process.\nExit code: %d\n--stdout--\n%s\n--stderr--\n%s" %
            (proc.returncode, proc.stdout, proc.stderr))


_SHIM = None


def shim() -> ctypes.CDLL:
    global _SHIM
    if _SHIM is not None:
        return _SHIM
    so_path = _build_shim(Path(default_cache_dir()) / "kunlun")
    sentinel = so_path.with_suffix(".ok")
    if not sentinel.is_file():
        _preflight(so_path)
        sentinel.touch()
    _SHIM = _load(so_path)
    return _SHIM


def compile_source(source: str, entry: str) -> KernelImage:
    return _compile(shim(), source, entry)


def launch(image_addr: int, image_len: int, khash: int, ncluster: int, ncore: int, stream: int,
           params: bytes, use_cdnn: bool, mangled: str) -> None:
    lib = shim()
    buf = ctypes.create_string_buffer(params, len(params))
    rc = lib.kl_launch(ctypes.c_void_p(image_addr), image_len, khash, ncluster, ncore,
                       ctypes.c_void_p(stream), buf, len(params), int(use_cdnn), mangled.encode())
    if rc != 0:
        msg = lib.kl_last_error()
        raise JitcError(msg.decode("utf-8", "replace") if msg else
                        f"kernel launch failed with code {rc}")
