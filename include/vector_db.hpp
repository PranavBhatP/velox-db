#pragma once
#include <vector>
#include <iostream>
#include <shared_mutex>
#include <utility>

class VectorIndex {
public:
    VectorIndex();
    ~VectorIndex();

    void add_vector(const std::vector<float>& vec);
    void load_fvecs(const std::string& filename);
    void build_index(int num_clusters, int epochs = 10, const std::string& metric = "eucl");
    void write_fvecs(const std::string& filename);
    void save_index(const std::string& filename);
    void load_index(const std::string& filename);
    std::vector<float> get_vector(int index);
    void set_simd(bool enable);

    // Returns up to k nearest neighbors as (id, distance) pairs, sorted nearest-first.
    // nprobe controls how many IVF clusters are probed (higher = better recall, slower).
    std::vector<std::pair<int, float>> search(
        const std::vector<float>& query,
        int k = 1,
        int nprobe = 1,
        const std::string& metric = "eucl"
    );

private:
    // Flat row-major storage: element [i][d] is at flat_database[i*dim + d].
    // Contiguous layout gives cache-friendly SIMD access vs row-of-rows scatter.
    std::vector<float> flat_database;

    bool use_mmap = false;
    void* mmap_ptr = nullptr;
    size_t mmap_size = 0;
    int dim = 0;
    int num_vectors = 0;
    bool is_indexed = false;

    std::vector<std::vector<float>> centroids;
    std::vector<std::vector<int>> inverted_lists;
    bool use_simd = false;

    mutable std::shared_mutex rw_mutex;

    // Internal unlocked accessor — callers must already hold rw_mutex.
    const float* raw_vec_ptr(int index) const;
    std::vector<float> get_vector_nolock(int index) const;
};
