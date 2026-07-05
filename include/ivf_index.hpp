#pragma once
#include "index_base.hpp"

// K-Means inverted-file index: partitions vectors into num_clusters
// centroids and, at search time, probes the nprobe closest centroids'
// inverted lists instead of scanning every vector.
class IVFIndex : public IndexAlgorithm {
public:
    void build(const VectorStorage& storage, const IndexParams& params) override;

    std::vector<std::pair<int, float>> search(
        const VectorStorage& storage, const float* query, int k,
        const IndexParams& params, bool use_simd) const override;

    void save(std::ofstream& out) const override;
    void load(std::ifstream& in, int dim) override;

    // Reads the pre-VELOX_VERSION-2 payload layout (num_clusters/dim were
    // already consumed by the facade before this is called).
    void load_legacy_v1(std::ifstream& in, int num_clusters, int dim);

    bool is_built() const override { return built_; }
    const char* type_name() const override { return "ivf"; }

private:
    std::vector<std::vector<float>> centroids_;
    std::vector<std::vector<int>> inverted_lists_;
    bool built_ = false;
    int dim_ = 0;
};
