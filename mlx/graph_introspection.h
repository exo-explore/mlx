// Copyright © 2024 Apple Inc.

#pragma once

#include <string>
#include <vector>

#include "mlx/array.h"

namespace mlx::core {

/**
 * Information about a primitive (operation) in the computation graph.
 */
struct PrimitiveInfo {
  std::string name;                     // Operation name (e.g., "MatMul", "Add")
  std::vector<Shape> input_shapes;      // Shapes of input arrays
  std::vector<Dtype> input_dtypes;      // Dtypes of input arrays
  Shape output_shape;                   // Shape of output array
  Dtype output_dtype;                   // Dtype of output array
  std::uintptr_t primitive_id;          // Unique identifier for this primitive
  std::uintptr_t array_id;              // Unique identifier for the output array

  // Constructor for proper initialization
  PrimitiveInfo(
      std::string name_,
      std::vector<Shape> input_shapes_,
      std::vector<Dtype> input_dtypes_,
      Shape output_shape_,
      Dtype output_dtype_,
      std::uintptr_t primitive_id_,
      std::uintptr_t array_id_)
      : name(std::move(name_)),
        input_shapes(std::move(input_shapes_)),
        input_dtypes(std::move(input_dtypes_)),
        output_shape(std::move(output_shape_)),
        output_dtype(output_dtype_),
        primitive_id(primitive_id_),
        array_id(array_id_) {}
};

/**
 * A node in the computation graph, representing an array and its producing operation.
 */
struct GraphNode {
  PrimitiveInfo primitive_info;         // Info about the operation
  std::vector<std::uintptr_t> input_ids; // Array IDs of inputs
  bool is_input;                        // True if this is an input (no primitive)

  // Constructor
  GraphNode(
      PrimitiveInfo primitive_info_,
      std::vector<std::uintptr_t> input_ids_,
      bool is_input_)
      : primitive_info(std::move(primitive_info_)),
        input_ids(std::move(input_ids_)),
        is_input(is_input_) {}
};

/**
 * Get information about the primitive that produces an array.
 *
 * @param arr The array to get primitive info for
 * @return PrimitiveInfo containing operation name, shapes, etc.
 * @throws std::runtime_error if the array has no primitive (is an input)
 */
PrimitiveInfo get_primitive_info(const array& arr);

/**
 * Check if an array has a primitive (is not an input).
 *
 * @param arr The array to check
 * @return true if the array has a primitive, false if it's an input
 */
bool has_primitive(const array& arr);

/**
 * Get all nodes in the computation graph leading to the given outputs.
 *
 * Performs a depth-first traversal starting from the outputs and returns
 * all nodes in the graph, including both intermediate computations and inputs.
 *
 * @param outputs The output arrays to trace back from
 * @return Vector of GraphNode representing all nodes in the graph
 */
std::vector<GraphNode> get_graph_nodes(const std::vector<array>& outputs);

/**
 * Get the immediate inputs of an array (the arrays that feed into its primitive).
 *
 * @param arr The array to get inputs for
 * @return Vector of input arrays, or empty if the array is an input itself
 */
std::vector<array> get_inputs(const array& arr);

/**
 * Get all arrays in the computation graph (convenience function).
 *
 * @param outputs The output arrays to trace back from
 * @return Vector of all arrays in the graph (in DFS order)
 */
std::vector<array> get_graph_arrays(const std::vector<array>& outputs);

} // namespace mlx::core
