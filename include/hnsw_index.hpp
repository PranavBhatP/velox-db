#pragma once
#include "index_base.hpp"
#include <random>

// Hierarchical Navigable Small World graph (Malkov & Yashunin). Builds a
// multi-layer proximity graph; higher layers are sparser "express lanes"
// used to greedily descend toward a good entry point before an
// exhaustive-ish search at layer 0. Neighbor selection uses simple
// closest-M pruning rather than the paper's diversity heuristic.
class HNSWIndex : public IndexAlgorithm {
public:
    void build(const VectorStorage& storage, const IndexParams& params) override;

    std::vector<std::pair<int, float>> search(
        const VectorStorage& storage, const float* query, int k,
        const IndexParams& params, bool use_simd) const override;

    void save(std::ofstream& out) const override;
    void load(std::ifstream& in, int dim) override;

    bool is_built() const override { return built_; }
    const char* type_name() const override { return "hnsw"; }

private:
    struct Node {
        int level = 0;
        std::vector<std::vector<int>> neighbors; // neighbors[layer] = adjacent node ids
    };

    // Best-first search within a single layer, starting from `entry`.
    // Returns up to `ef` (distance, id) pairs sorted nearest-first.
    std::vector<std::pair<float, int>> search_layer(
        const VectorStorage& storage, const float* query, int entry, int ef,
        int layer, bool use_simd, const std::string& metric) const;

    int random_level();

    std::vector<Node> nodes_;
    int entry_point_ = -1;
    int max_level_ = -1;
    int M_ = 16;
    int M_max0_ = 32;
    int ef_construction_ = 200;
    bool built_ = false;
    std::mt19937 rng_{std::random_device{}()};
};
