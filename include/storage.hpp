#pragma once
#include <vector>
#include <string>

// Owns raw vector storage: either an in-RAM flat buffer (populated via
// add_vector) or a read-only mmap'd .fvecs file (populated via load_fvecs).
// Not thread-safe on its own — callers (VectorIndex) are responsible for
// locking around calls into this class.
class VectorStorage {
public:
    VectorStorage() = default;
    ~VectorStorage();

    VectorStorage(const VectorStorage&) = delete;
    VectorStorage& operator=(const VectorStorage&) = delete;

    void add_vector(const std::vector<float>& vec);
    void load_fvecs(const std::string& filename);
    void write_fvecs(const std::string& filename) const;

    // Bounds-checked copy of vector `index`. Throws std::out_of_range.
    std::vector<float> get_vector(int index) const;

    // Zero-copy pointer to vector `index`'s float data. Caller must ensure
    // index is in range — no bounds check (hot path for build/search loops).
    const float* raw_vec_ptr(int index) const;

    int dim() const { return dim_; }
    int size() const { return num_vectors_; }
    bool is_mmapped() const { return use_mmap_; }

private:
    // Flat row-major storage: element [i][d] is at flat_database_[i*dim_ + d].
    std::vector<float> flat_database_;

    bool use_mmap_ = false;
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;

    int dim_ = 0;
    int num_vectors_ = 0;
};
