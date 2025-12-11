#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "vector_db.hpp"

namespace py = pybind11;

PYBIND11_MODULE(veloxdb, m) {
    m.doc() = "VeloxDB: A high-performance vector database written in C++";


    py::class_<VectorIndex>(m, "VectorIndex")
        .def(py::init<>()) // Constructor
        .def("add_vector", &VectorIndex::add_vector, "Add a float vector to the index")
        .def("load_fvecs", &VectorIndex::load_fvecs, "Load the fvecs file for vector at requested index.")
        .def("get_vector", &VectorIndex::get_vector, "Retrieve a vector by integer ID")
        .def("build_index", &VectorIndex::build_index, " Build IVF Index for ANN search using K-Means Clustering."
        , py::arg("num_clusters"), py::arg("max_iters")=10)
        .def("search", &VectorIndex::search, "Find the nearest neighbor index");
}