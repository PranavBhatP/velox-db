#pragma once
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include "storage.hpp"
#include "metrics.hpp"

// Shared distance dispatch used by every index algorithm implementation.
inline float compute_dist(const float* a, const float* b, int n,
                           bool use_simd, const std::string& metric) {
    if (metric == "cos")
        return use_simd ? cosine_dist_simd(a, b, n) : cosine_dist(a, b, n);
    return use_simd ? euclidean_dist_simd(a, b, n) : euclidean_dist(a, b, n);
}

// Flat hyperparameter bag covering both current index algorithms (IVF and
// HNSW). Each concrete IndexAlgorithm reads only the fields it needs.
struct IndexParams {
    std::string metric = "eucl";
    bool use_simd = false;

    // IVF
    int num_clusters = 0;
    int epochs = 10;
    int nprobe = 1;

    // HNSW
    int M = 16;
    int ef_construction = 200;
    int ef_search = 50;
};

// Interface implemented by each concrete index algorithm (IVF, HNSW, ...).
// VectorIndex (the facade) owns a VectorStorage and delegates build/search/
// persistence to whichever IndexAlgorithm is currently active.
class IndexAlgorithm {
public:
    virtual ~IndexAlgorithm() = default;

    virtual void build(const VectorStorage& storage, const IndexParams& params) = 0;

    virtual std::vector<std::pair<int, float>> search(
        const VectorStorage& storage, const float* query, int k,
        const IndexParams& params, bool use_simd) const = 0;

    // Persist/restore algorithm-specific state only; the facade owns the
    // common file header (magic, version, type discriminator, dim).
    virtual void save(std::ofstream& out) const = 0;
    virtual void load(std::ifstream& in, int dim) = 0;

    virtual bool is_built() const = 0;
    virtual const char* type_name() const = 0;
};
