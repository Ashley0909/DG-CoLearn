#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "partition.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_partition, m) {
    m.doc() = "Partition module";

    m.def("CoLearnPartition", &CoLearnPartition,
          py::arg("adj_list"), py::arg("global_size"), py::arg("node_labels"), py::arg("K"));
}