// Copyright © 2024 Apple Inc.

#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "mlx/graph_introspection.h"
#include "mlx/primitives.h"

namespace mlx::core {

PrimitiveInfo get_primitive_info(const array& arr) {
  if (!arr.has_primitive()) {
    throw std::runtime_error(
        "Array has no primitive - it is an input node. "
        "Use has_primitive() to check before calling get_primitive_info().");
  }

  // Collect input shapes and dtypes
  std::vector<Shape> input_shapes;
  std::vector<Dtype> input_dtypes;
  for (const auto& input : arr.inputs()) {
    input_shapes.push_back(input.shape());
    input_dtypes.push_back(input.dtype());
  }

  return PrimitiveInfo(
      arr.primitive().name(),
      std::move(input_shapes),
      std::move(input_dtypes),
      arr.shape(),
      arr.dtype(),
      arr.primitive_id(),
      arr.id());
}

bool has_primitive(const array& arr) {
  return arr.has_primitive();
}

std::vector<array> get_inputs(const array& arr) {
  if (!arr.has_primitive()) {
    return {};
  }
  return arr.inputs();
}

std::vector<GraphNode> get_graph_nodes(const std::vector<array>& outputs) {
  std::vector<GraphNode> nodes;
  std::unordered_set<std::uintptr_t> visited;

  std::function<void(const array&)> traverse;
  traverse = [&](const array& arr) {
    auto id = arr.id();
    if (visited.find(id) != visited.end()) {
      return;
    }
    visited.insert(id);

    // Also mark siblings as visited
    for (const auto& sibling : arr.siblings()) {
      visited.insert(sibling.id());
    }

    // Traverse inputs first (DFS)
    for (const auto& input : arr.inputs()) {
      traverse(input);
    }

    // Collect input array IDs
    std::vector<std::uintptr_t> input_ids;
    for (const auto& input : arr.inputs()) {
      input_ids.push_back(input.id());
    }

    // Create node for this array
    if (arr.has_primitive()) {
      nodes.emplace_back(
          get_primitive_info(arr),
          std::move(input_ids),
          false);
    } else {
      // For inputs, create basic info
      PrimitiveInfo info(
          "Input",
          std::vector<Shape>{},
          std::vector<Dtype>{},
          arr.shape(),
          arr.dtype(),
          0,
          arr.id());
      nodes.emplace_back(
          std::move(info),
          std::move(input_ids),
          true);
    }
  };

  for (const auto& output : outputs) {
    traverse(output);
  }

  return nodes;
}

std::vector<array> get_graph_arrays(const std::vector<array>& outputs) {
  std::vector<array> arrays;
  std::unordered_set<std::uintptr_t> visited;

  std::function<void(const array&)> traverse;
  traverse = [&](const array& arr) {
    auto id = arr.id();
    if (visited.find(id) != visited.end()) {
      return;
    }
    visited.insert(id);

    // Also mark siblings as visited
    for (const auto& sibling : arr.siblings()) {
      visited.insert(sibling.id());
    }

    // Traverse inputs first (DFS)
    for (const auto& input : arr.inputs()) {
      traverse(input);
    }

    arrays.push_back(arr);
  };

  for (const auto& output : outputs) {
    traverse(output);
  }

  return arrays;
}

} // namespace mlx::core
