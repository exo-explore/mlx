// Copyright © 2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/array.h"
#include "mlx/graph_introspection.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

void init_graph_introspection(nb::module_& m) {
  // PrimitiveInfo struct
  nb::class_<mx::PrimitiveInfo>(
      m,
      "PrimitiveInfo",
      R"pbdoc(
        Information about a primitive (operation) in the computation graph.

        Attributes:
            name (str): The operation name (e.g., "MatMul", "Add", "Softmax").
            input_shapes (List[List[int]]): Shapes of input arrays.
            input_dtypes (List[Dtype]): Data types of input arrays.
            output_shape (List[int]): Shape of the output array.
            output_dtype (Dtype): Data type of the output array.
            primitive_id (int): Unique identifier for this primitive instance.
            array_id (int): Unique identifier for the output array.
      )pbdoc")
      .def_ro("name", &mx::PrimitiveInfo::name)
      .def_prop_ro(
          "input_shapes",
          [](const mx::PrimitiveInfo& info) {
            std::vector<std::vector<int>> shapes;
            for (const auto& shape : info.input_shapes) {
              shapes.push_back(std::vector<int>(shape.begin(), shape.end()));
            }
            return shapes;
          })
      .def_prop_ro(
          "input_dtypes",
          [](const mx::PrimitiveInfo& info) { return info.input_dtypes; })
      .def_prop_ro(
          "output_shape",
          [](const mx::PrimitiveInfo& info) {
            return std::vector<int>(
                info.output_shape.begin(), info.output_shape.end());
          })
      .def_ro("output_dtype", &mx::PrimitiveInfo::output_dtype)
      .def_ro("primitive_id", &mx::PrimitiveInfo::primitive_id)
      .def_ro("array_id", &mx::PrimitiveInfo::array_id)
      .def("__repr__", [](const mx::PrimitiveInfo& info) {
        std::ostringstream os;
        os << "PrimitiveInfo(name='" << info.name << "', ";
        os << "input_shapes=[";
        for (size_t i = 0; i < info.input_shapes.size(); ++i) {
          os << "[";
          for (size_t j = 0; j < info.input_shapes[i].size(); ++j) {
            os << info.input_shapes[i][j];
            if (j < info.input_shapes[i].size() - 1) os << ", ";
          }
          os << "]";
          if (i < info.input_shapes.size() - 1) os << ", ";
        }
        os << "], output_shape=[";
        for (size_t i = 0; i < info.output_shape.size(); ++i) {
          os << info.output_shape[i];
          if (i < info.output_shape.size() - 1) os << ", ";
        }
        os << "])";
        return os.str();
      });

  // GraphNode struct
  nb::class_<mx::GraphNode>(
      m,
      "GraphNode",
      R"pbdoc(
        A node in the computation graph.

        Represents either an input array or an operation (primitive) that
        produces an array.

        Attributes:
            primitive_info (PrimitiveInfo): Information about the operation.
            input_ids (List[int]): Array IDs of the inputs to this operation.
            is_input (bool): True if this is an input node (no operation).
      )pbdoc")
      .def_ro("primitive_info", &mx::GraphNode::primitive_info)
      .def_ro("input_ids", &mx::GraphNode::input_ids)
      .def_ro("is_input", &mx::GraphNode::is_input)
      .def("__repr__", [](const mx::GraphNode& node) {
        std::ostringstream os;
        os << "GraphNode(";
        if (node.is_input) {
          os << "is_input=True";
        } else {
          os << "op='" << node.primitive_info.name << "'";
        }
        os << ", id=" << node.primitive_info.array_id;
        os << ")";
        return os.str();
      });

  // has_primitive function
  m.def(
      "has_primitive",
      &mx::has_primitive,
      "arr"_a,
      R"pbdoc(
        Check if an array has an associated primitive (operation).

        Returns True if the array is the result of a computation,
        False if it's an input (leaf) array.

        Args:
            arr (array): The array to check.

        Returns:
            bool: True if the array has a primitive.

        Example:
          >>> a = mx.array([1, 2, 3])
          >>> mx.has_primitive(a)
          False
          >>> b = a + 1
          >>> mx.has_primitive(b)
          True
      )pbdoc");

  // get_primitive_info function
  m.def(
      "get_primitive_info",
      &mx::get_primitive_info,
      "arr"_a,
      R"pbdoc(
        Get information about the primitive (operation) that produces an array.

        Args:
            arr (array): The array to get primitive info for.

        Returns:
            PrimitiveInfo: Information about the operation including name,
            input/output shapes, and unique identifiers.

        Raises:
            RuntimeError: If the array has no primitive (is an input).

        Example:
          >>> a = mx.array([[1, 2], [3, 4]])
          >>> b = mx.array([[5, 6], [7, 8]])
          >>> c = a @ b
          >>> info = mx.get_primitive_info(c)
          >>> info.name
          'MatMul'
          >>> info.input_shapes
          [[2, 2], [2, 2]]
      )pbdoc");

  // get_inputs function
  m.def(
      "get_inputs",
      &mx::get_inputs,
      "arr"_a,
      R"pbdoc(
        Get the immediate input arrays of an operation.

        Returns the arrays that feed into the primitive that produces this
        array.

        Args:
            arr (array): The array to get inputs for.

        Returns:
            List[array]: The input arrays, or empty list if arr is an input.

        Example:
          >>> a = mx.array([1, 2, 3])
          >>> b = mx.array([4, 5, 6])
          >>> c = a + b
          >>> inputs = mx.get_inputs(c)
          >>> len(inputs)
          2
      )pbdoc");

  // get_graph_nodes function
  m.def(
      "get_graph_nodes",
      &mx::get_graph_nodes,
      "outputs"_a,
      R"pbdoc(
        Get all nodes in the computation graph.

        Performs a depth-first traversal starting from the output arrays and
        returns all nodes (both operations and inputs) in the graph.

        Args:
            outputs (List[array]): The output arrays to trace back from.

        Returns:
            List[GraphNode]: All nodes in the graph in DFS order.

        Example:
          >>> a = mx.array([1, 2, 3])
          >>> b = a * 2
          >>> c = b + 1
          >>> nodes = mx.get_graph_nodes([c])
          >>> for node in nodes:
          ...     print(node.primitive_info.name)
          Input
          Multiply
          Add
      )pbdoc");

  // get_graph_arrays function
  m.def(
      "get_graph_arrays",
      &mx::get_graph_arrays,
      "outputs"_a,
      R"pbdoc(
        Get all arrays in the computation graph.

        Convenience function that returns all array objects in the graph.

        Args:
            outputs (List[array]): The output arrays to trace back from.

        Returns:
            List[array]: All arrays in the graph in DFS order.

        Example:
          >>> a = mx.array([1, 2, 3])
          >>> b = a * 2
          >>> c = b + 1
          >>> arrays = mx.get_graph_arrays([c])
          >>> len(arrays)
          3
      )pbdoc");
}
