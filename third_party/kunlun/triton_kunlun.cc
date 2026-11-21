#include "XPUC/Emitter.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <tuple>

namespace py = pybind11;

void init_triton_kunlun(py::module &&m) {
  m.def(
      "translate_to_xpuc",
      [](mlir::ModuleOp module, int64_t lmBudgetBytes, bool distributed) {
        std::string source, entry, error;
        if (!mlir::triton::kunlun::emitXPUCSource(module, lmBudgetBytes, distributed,
                                                 source, entry, error))
          throw std::runtime_error(error);
        return std::make_tuple(source, entry);
      },
      py::arg("module"), py::arg("lm_budget_bytes"), py::arg("distributed"));
}
