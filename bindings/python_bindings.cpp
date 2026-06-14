#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vector_db.hpp"

namespace py = pybind11;

PYBIND11_MODULE(veloxdb, m) {
    m.doc() = "VeloxDB: A high-performance vector database written in C++";

    py::class_<VectorIndex>(m, "VectorIndex")
        .def(py::init<>())
        .def("add_vector",  &VectorIndex::add_vector,  "Add a float vector to the index")
        .def("load_fvecs",  &VectorIndex::load_fvecs,  "Memory-map a .fvecs file")
        .def("get_vector",  &VectorIndex::get_vector,  "Retrieve a vector by integer ID")
        .def("build_index", &VectorIndex::build_index, "Build IVF index via K-Means clustering.",
             py::arg("num_clusters"), py::arg("epochs") = 10, py::arg("metric") = "eucl")
        .def("write_fvecs", &VectorIndex::write_fvecs, "Export in-memory vectors to disk")
        .def("save_index",  &VectorIndex::save_index,  "Save IVF index to .ivf file")
        .def("load_index",  &VectorIndex::load_index,  "Load IVF index from .ivf file")
        .def("set_simd",    &VectorIndex::set_simd)
        // Returns list of (id, distance) tuples sorted nearest-first.
        // k      — number of results to return.
        // nprobe — number of IVF clusters to probe (higher = better recall, slower).
        .def("search", &VectorIndex::search,
             py::arg("query"), py::arg("k") = 1, py::arg("nprobe") = 1,
             py::arg("metric") = "eucl");
}
