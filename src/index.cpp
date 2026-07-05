#include "vector_db.hpp"
#include "ivf_index.hpp"
#include "hnsw_index.hpp"
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <queue>
#include <algorithm>
#include <mutex>

// Magic bytes written at the start of every index file so we can detect
// stale or corrupted files instead of silently misreading them. Version 2
// adds a 1-byte index_type discriminator; version 1 files (IVF-only, no
// discriminator) are still readable via the legacy path in load_index.
static constexpr uint32_t VELOX_MAGIC   = 0x564C5846; // 'V','L','X','F'
static constexpr uint16_t VELOX_VERSION = 2;

VectorIndex::VectorIndex() {
    std::cout << "VectorIndex initialised!\n";
}

VectorIndex::~VectorIndex() = default;

void VectorIndex::add_vector(const std::vector<float>& vec) {
    std::unique_lock lock(rw_mutex_);
    storage_.add_vector(vec);
}

void VectorIndex::load_fvecs(const std::string& filename) {
    std::unique_lock lock(rw_mutex_);
    storage_.load_fvecs(filename);
}

void VectorIndex::write_fvecs(const std::string& filename) {
    std::shared_lock lock(rw_mutex_);
    storage_.write_fvecs(filename);
}

std::vector<float> VectorIndex::get_vector(int index) {
    std::shared_lock lock(rw_mutex_);
    return storage_.get_vector(index);
}

void VectorIndex::set_simd(bool enable) {
    std::unique_lock lock(rw_mutex_);
    use_simd_ = enable;
    std::cout << "SIMD: " << (use_simd_ ? "enabled" : "disabled") << "\n";
}

void VectorIndex::build_index(int num_clusters, int epochs, const std::string& metric) {
    std::unique_lock lock(rw_mutex_);
    IndexParams params;
    params.metric = metric;
    params.use_simd = use_simd_;
    params.num_clusters = num_clusters;
    params.epochs = epochs;

    auto ivf = std::make_unique<IVFIndex>();
    ivf->build(storage_, params);
    algo_ = std::move(ivf);
}

void VectorIndex::build_index_hnsw(int M, int ef_construction, const std::string& metric) {
    std::unique_lock lock(rw_mutex_);
    IndexParams params;
    params.metric = metric;
    params.use_simd = use_simd_;
    params.M = M;
    params.ef_construction = ef_construction;

    auto hnsw = std::make_unique<HNSWIndex>();
    hnsw->build(storage_, params);
    algo_ = std::move(hnsw);
}

std::vector<std::pair<int, float>> VectorIndex::search(
    const std::vector<float>& query, int k, int nprobe,
    const std::string& metric, int ef_search)
{
    std::shared_lock lock(rw_mutex_);

    if (static_cast<int>(query.size()) != storage_.dim())
        throw std::runtime_error(
            "Query dim=" + std::to_string(query.size()) +
            " != index dim=" + std::to_string(storage_.dim()));

    if (!algo_ || !algo_->is_built()) {
        // Brute-force fallback over every stored vector (algorithm-agnostic,
        // so it lives here rather than in either concrete IndexAlgorithm).
        int num_vectors = storage_.size();
        using Entry = std::pair<float, int>;
        std::priority_queue<Entry> heap;

        for (int vid = 0; vid < num_vectors; vid++) {
            float d = compute_dist(storage_.raw_vec_ptr(vid), query.data(), storage_.dim(), use_simd_, metric);
            if (static_cast<int>(heap.size()) < k) {
                heap.emplace(d, vid);
            } else if (d < heap.top().first) {
                heap.pop();
                heap.emplace(d, vid);
            }
        }

        std::vector<std::pair<int, float>> results;
        results.reserve(heap.size());
        while (!heap.empty()) {
            results.emplace_back(heap.top().second, heap.top().first);
            heap.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

    IndexParams params;
    params.metric = metric;
    params.nprobe = nprobe;
    params.ef_search = ef_search < 0 ? 50 : ef_search;

    return algo_->search(storage_, query.data(), k, params, use_simd_);
}

void VectorIndex::save_index(const std::string& filename) {
    std::shared_lock lock(rw_mutex_);
    if (!algo_ || !algo_->is_built())
        throw std::runtime_error("No index to save.");

    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output file.");

    out.write(reinterpret_cast<const char*>(&VELOX_MAGIC),   sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&VELOX_VERSION), sizeof(uint16_t));

    uint8_t type_id = (std::string(algo_->type_name()) == "hnsw") ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&type_id), sizeof(uint8_t));

    int dim = storage_.dim();
    out.write(reinterpret_cast<const char*>(&dim), sizeof(int));

    algo_->save(out);

    out.close();
    std::cout << "Index saved to " << filename << "\n";
}

void VectorIndex::load_index(const std::string& filename) {
    std::unique_lock lock(rw_mutex_);
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open index file.");

    uint32_t magic;
    uint16_t version;
    in.read(reinterpret_cast<char*>(&magic),   sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&version), sizeof(uint16_t));

    if (magic != VELOX_MAGIC)
        throw std::runtime_error(
            "Not a VeloxDB index file (bad magic bytes). "
            "Delete the index file and retrain.");

    if (version == 1) {
        // Legacy format: no index_type discriminator, always IVF, and
        // num_clusters/dim were written in that order (not dim-first).
        int num_clusters, loaded_dim;
        in.read(reinterpret_cast<char*>(&num_clusters), sizeof(int));
        in.read(reinterpret_cast<char*>(&loaded_dim),   sizeof(int));

        if (loaded_dim == 0 || loaded_dim != storage_.dim())
            throw std::runtime_error(
                "Dimension mismatch: data dim=" + std::to_string(storage_.dim()) +
                ", index dim=" + std::to_string(loaded_dim));

        auto ivf = std::make_unique<IVFIndex>();
        ivf->load_legacy_v1(in, num_clusters, loaded_dim);
        algo_ = std::move(ivf);
        std::cout << "Index loaded (legacy v1): " << num_clusters << " clusters\n";
        return;
    }

    if (version != VELOX_VERSION)
        throw std::runtime_error("Unsupported index version: " + std::to_string(version));

    uint8_t type_id;
    in.read(reinterpret_cast<char*>(&type_id), sizeof(uint8_t));
    int loaded_dim;
    in.read(reinterpret_cast<char*>(&loaded_dim), sizeof(int));

    if (loaded_dim == 0 || loaded_dim != storage_.dim())
        throw std::runtime_error(
            "Dimension mismatch: data dim=" + std::to_string(storage_.dim()) +
            ", index dim=" + std::to_string(loaded_dim));

    if (type_id == 1) {
        auto hnsw = std::make_unique<HNSWIndex>();
        hnsw->load(in, loaded_dim);
        algo_ = std::move(hnsw);
        std::cout << "Index loaded: hnsw\n";
    } else {
        auto ivf = std::make_unique<IVFIndex>();
        ivf->load(in, loaded_dim);
        algo_ = std::move(ivf);
        std::cout << "Index loaded: ivf\n";
    }
}

std::string VectorIndex::get_index_type() const {
    std::shared_lock lock(rw_mutex_);
    return algo_ ? algo_->type_name() : "none";
}
