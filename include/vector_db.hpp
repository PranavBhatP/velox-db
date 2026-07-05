#pragma once
#include <vector>
#include <string>
#include <memory>
#include <shared_mutex>
#include <utility>
#include "storage.hpp"
#include "index_base.hpp"

// Facade: owns raw vector storage plus whichever IndexAlgorithm (IVF or
// HNSW) is currently active, and guards both with a single coarse
// shared_mutex (shared lock for reads, unique lock for writes/rebuilds).
class VectorIndex {
public:
    VectorIndex();
    ~VectorIndex();

    void add_vector(const std::vector<float>& vec);
    void load_fvecs(const std::string& filename);
    void write_fvecs(const std::string& filename);
    std::vector<float> get_vector(int index);
    void set_simd(bool enable);

    void build_index(int num_clusters, int epochs = 10, const std::string& metric = "eucl");
    void build_index_hnsw(int M = 16, int ef_construction = 200, const std::string& metric = "eucl");

    // Returns up to k nearest neighbors as (id, distance) pairs, sorted nearest-first.
    // nprobe controls IVF cluster probing; ef_search controls HNSW search breadth.
    // Whichever doesn't apply to the currently-built index is ignored.
    std::vector<std::pair<int, float>> search(
        const std::vector<float>& query,
        int k = 1,
        int nprobe = 1,
        const std::string& metric = "eucl",
        int ef_search = -1
    );

    void save_index(const std::string& filename);
    void load_index(const std::string& filename);

    // "none" if untrained, otherwise "ivf" or "hnsw".
    std::string get_index_type() const;

private:
    VectorStorage storage_;
    std::unique_ptr<IndexAlgorithm> algo_;
    bool use_simd_ = false;
    mutable std::shared_mutex rw_mutex_;
};
