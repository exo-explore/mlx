// Copyright © 2023-2024 Apple Inc.
#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/backend/metal/metal.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "python/src/small_vector.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

bool DEPRECATE(const char* old_fn, const char* new_fn) {
  std::cerr << old_fn << " is deprecated and will be removed in a future "
            << "version. Use " << new_fn << " instead." << std::endl;
  return true;
}

#define DEPRECATE(oldfn, newfn) static bool dep = DEPRECATE(oldfn, newfn)

void init_metal(nb::module_& m) {
  nb::module_ metal = m.def_submodule("metal", "mlx.metal");
  metal.def(
      "is_available",
      &mx::metal::is_available,
      R"pbdoc(
      Check if the Metal back-end is available.
      )pbdoc");
  metal.def("get_active_memory", []() {
    DEPRECATE("mx.metal.get_active_memory", "mx.get_active_memory");
    return mx::get_active_memory();
  });
  metal.def("get_peak_memory", []() {
    DEPRECATE("mx.metal.get_peak_memory", "mx.get_peak_memory");
    return mx::get_peak_memory();
  });
  metal.def("reset_peak_memory", []() {
    DEPRECATE("mx.metal.reset_peak_memory", "mx.reset_peak_memory");
    mx::reset_peak_memory();
  });
  metal.def("get_cache_memory", []() {
    DEPRECATE("mx.metal.get_cache_memory", "mx.get_cache_memory");
    return mx::get_cache_memory();
  });
  metal.def(
      "set_memory_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_memory_limit", "mx.set_memory_limit");
        return mx::set_memory_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_cache_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_cache_limit", "mx.set_cache_limit");
        return mx::set_cache_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_wired_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_wired_limit", "mx.set_wired_limit");
        return mx::set_wired_limit(limit);
      },
      "limit"_a);
  metal.def("clear_cache", []() {
    DEPRECATE("mx.metal.clear_cache", "mx.clear_cache");
    mx::clear_cache();
  });
  metal.def(
      "start_capture",
      &mx::metal::start_capture,
      "path"_a,
      R"pbdoc(
      Start a Metal capture.

      Args:
        path (str): The path to save the capture which should have
          the extension ``.gputrace``.
      )pbdoc");
  metal.def(
      "stop_capture",
      &mx::metal::stop_capture,
      R"pbdoc(
      Stop a Metal capture.
      )pbdoc");
  metal.def(
      "push_debug_group",
      &mx::metal::push_debug_group,
      "label"_a,
      R"pbdoc(
      Push a debug group for organizing GPU operations in Metal captures.

      Debug groups create hierarchical labels in Metal GPU traces
      (created with :func:`start_capture`), making it easier to
      understand GPU workloads in Xcode's GPU debugger.

      Args:
        label (str): The label for the debug group.

      Example:
        >>> mx.metal.start_capture("trace.gputrace")
        >>> mx.metal.push_debug_group("attention")
        >>> # ... GPU operations ...
        >>> mx.metal.pop_debug_group()
        >>> mx.metal.stop_capture()
      )pbdoc");
  metal.def(
      "pop_debug_group",
      &mx::metal::pop_debug_group,
      R"pbdoc(
      Pop the most recent debug group.
      )pbdoc");

  class PyDebugGroup {
   public:
    PyDebugGroup(const std::string& label) : label_(label) {}
    void enter() {
      mx::metal::push_debug_group(label_);
    }
    void exit() {
      mx::metal::pop_debug_group();
    }

   private:
    std::string label_;
  };

  nb::class_<PyDebugGroup>(
      metal,
      "DebugGroup",
      R"pbdoc(
      Context manager for Metal debug groups.

      Example:
        >>> with mx.metal.DebugGroup("forward_pass"):
        ...     # GPU operations appear under "forward_pass" in traces
        ...     pass
      )pbdoc")
      .def(nb::init<const std::string&>(), "label"_a)
      .def(
          "__enter__",
          [](PyDebugGroup& dg) -> PyDebugGroup& {
            dg.enter();
            return dg;
          })
      .def(
          "__exit__",
          [](PyDebugGroup& dg,
             const std::optional<nb::type_object>&,
             const std::optional<nb::object>&,
             const std::optional<nb::object>&) { dg.exit(); },
          "exc_type"_a = nb::none(),
          "exc_value"_a = nb::none(),
          "traceback"_a = nb::none());

  metal.def("device_info", []() {
    DEPRECATE("mx.metal.device_info", "mx.device_info");
    return mx::device_info(mx::Device(mx::Device::gpu, 0));
  });
}
