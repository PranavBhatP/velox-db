#include "vector_db.hpp"
#include "metrics.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <random>
#include <algorithm>
#include <numeric>
#include <queue>
#include <immintrin.h>
#include <fstream>
#include <mutex>

// Magic bytes written at the start of every .ivf file so we can detect stale
// or corrupted index files instead of silently misreading them.
static constexpr uint32_t VELOX_MAGIC   = 0x564C5846; // 'V','L','X','F'
static constexpr uint16_t VELOX_VERSION = 1;

inline float compute_dist(const float* a, const float* b, int n,
                           bool use_simd, const std::string& metric) {
    if (metric == "cos")
        return use_simd ? cosine_dist_simd(a, b, n) : cosine_dist(a, b, n);
    return use_simd ? euclidean_dist_simd(a, b, n) : euclidean_dist(a, b, n);
}

VectorIndex::VectorIndex() {
    std::cout << "VectorIndex initialised!\n";
}

VectorIndex::~VectorIndex() {
    if (use_mmap && mmap_ptr != nullptr)
        munmap(mmap_ptr, mmap_size);
}

// ---------------------------------------------------------------------------
// raw_vec_ptr — zero-copy pointer to vector i's float data.
// Caller must hold rw_mutex (shared or exclusive).
// ---------------------------------------------------------------------------
const float* VectorIndex::raw_vec_ptr(int index) const {
    if (!use_mmap)
        return flat_database.data() + index * dim;

    const char* base = static_cast<const char*>(mmap_ptr);
    size_t row_bytes = sizeof(int) + dim * sizeof(float);
    return reinterpret_cast<const float*>(base + index * row_bytes + sizeof(int));
}

std::vector<float> VectorIndex::get_vector_nolock(int index) const {
    if (index < 0 || index >= num_vectors)
        throw std::out_of_range("Index out of bounds");
    const float* p = raw_vec_ptr(index);
    return std::vector<float>(p, p + dim);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void VectorIndex::add_vector(const std::vector<float>& vec) {
    std::unique_lock lock(rw_mutex);
    if (use_mmap)
        throw std::runtime_error("Cannot add vectors to a read-only mmap index.");
    if (num_vectors == 0) {
        dim = static_cast<int>(vec.size());
    } else if (static_cast<int>(vec.size()) != dim) {
        throw std::runtime_error("Vector dimension mismatch.");
    }
    flat_database.insert(flat_database.end(), vec.begin(), vec.end());
    num_vectors++;
}

void VectorIndex::load_fvecs(const std::string& filename) {
    std::unique_lock lock(rw_mutex);
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1)
        throw std::runtime_error("Could not open file: " + filename);

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Could not stat file: " + filename);
    }

    mmap_size = sb.st_size;
    mmap_ptr = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mmap_ptr == MAP_FAILED)
        throw std::runtime_error("mmap failed.");

    const int* header = static_cast<const int*>(mmap_ptr);
    dim = header[0];
    size_t row_bytes = sizeof(int) + dim * sizeof(float);
    num_vectors = static_cast<int>(mmap_size / row_bytes);
    use_mmap = true;

    std::cout << "[VeloxDB] Loaded " << num_vectors
              << " vectors (dim=" << dim << ") via mmap.\n";
}

void VectorIndex::write_fvecs(const std::string& filename) {
    std::shared_lock lock(rw_mutex);
    if (use_mmap)
        throw std::runtime_error("Already using mmap; cannot export in-memory data.");
    if (num_vectors == 0)
        throw std::runtime_error("No data to write.");

    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output file.");

    for (int i = 0; i < num_vectors; i++) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(flat_database.data() + i * dim),
                  dim * sizeof(float));
    }
    out.close();
    std::cout << "Wrote " << num_vectors << " vectors to " << filename << "\n";
}

// ---------------------------------------------------------------------------
// save_index / load_index — versioned binary format with magic bytes.
// Old .ivf files (no magic) will be rejected rather than misread.
// ---------------------------------------------------------------------------

void VectorIndex::save_index(const std::string& filename) {
    std::shared_lock lock(rw_mutex);
    if (!is_indexed)
        throw std::runtime_error("No index to save.");

    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output file.");

    out.write(reinterpret_cast<const char*>(&VELOX_MAGIC),   sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&VELOX_VERSION), sizeof(uint16_t));

    int num_clusters = static_cast<int>(centroids.size());
    out.write(reinterpret_cast<const char*>(&num_clusters), sizeof(int));
    out.write(reinterpret_cast<const char*>(&dim),          sizeof(int));

    for (const auto& c : centroids)
        out.write(reinterpret_cast<const char*>(c.data()), dim * sizeof(float));

    for (const auto& lst : inverted_lists) {
        int sz = static_cast<int>(lst.size());
        out.write(reinterpret_cast<const char*>(&sz),       sizeof(int));
        out.write(reinterpret_cast<const char*>(lst.data()), sz * sizeof(int));
    }

    out.close();
    std::cout << "Index saved to " << filename << "\n";
}

void VectorIndex::load_index(const std::string& filename) {
    std::unique_lock lock(rw_mutex);
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open index file.");

    uint32_t magic;
    uint16_t version;
    in.read(reinterpret_cast<char*>(&magic),   sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&version), sizeof(uint16_t));

    if (magic != VELOX_MAGIC)
        throw std::runtime_error(
            "Not a VeloxDB index file (bad magic bytes). "
            "Delete the .ivf file and retrain.");
    if (version != VELOX_VERSION)
        throw std::runtime_error(
            "Unsupported index version: " + std::to_string(version));

    int num_clusters, loaded_dim;
    in.read(reinterpret_cast<char*>(&num_clusters), sizeof(int));
    in.read(reinterpret_cast<char*>(&loaded_dim),   sizeof(int));

    if (loaded_dim == 0 || loaded_dim != dim)
        throw std::runtime_error(
            "Dimension mismatch: data dim=" + std::to_string(dim) +
            ", index dim=" + std::to_string(loaded_dim));

    centroids.resize(num_clusters);
    for (int i = 0; i < num_clusters; i++) {
        centroids[i].resize(loaded_dim);
        in.read(reinterpret_cast<char*>(centroids[i].data()), loaded_dim * sizeof(float));
    }

    inverted_lists.resize(num_clusters);
    for (int i = 0; i < num_clusters; i++) {
        int sz;
        in.read(reinterpret_cast<char*>(&sz), sizeof(int));
        inverted_lists[i].resize(sz);
        in.read(reinterpret_cast<char*>(inverted_lists[i].data()), sz * sizeof(int));
    }

    is_indexed = true;
    std::cout << "Index loaded: " << num_clusters << " clusters\n";
}

std::vector<float> VectorIndex::get_vector(int index) {
    std::shared_lock lock(rw_mutex);
    return get_vector_nolock(index);
}

void VectorIndex::set_simd(bool enable) {
    std::unique_lock lock(rw_mutex);
    use_simd = enable;
    std::cout << "SIMD: " << (use_simd ? "enabled" : "disabled") << "\n";
}

// ---------------------------------------------------------------------------
// build_index — K-Means IVF training.
//
// Key optimisations vs the original:
//  1. Pre-caches all vectors into a flat buffer before training starts.
//     This avoids O(N × epochs × k) heap allocations from get_vector() in
//     the hot loop, and makes mmap-backed data cache-friendly.
//  2. Uses raw float pointers throughout — no temporary std::vector per step.
// ---------------------------------------------------------------------------
void VectorIndex::build_index(int num_clusters, int epochs,
                               const std::string& metric) {
    std::unique_lock lock(rw_mutex);
    if (num_vectors < num_clusters)
        throw std::runtime_error("Not enough vectors to fill " +
                                 std::to_string(num_clusters) + " clusters.");

    std::cout << "Training IVF index: " << num_clusters
              << " clusters, " << epochs << " epochs.\n";

    // Pre-cache all vectors in a flat contiguous buffer.
    // For in-memory storage this is zero-cost (flat_database is already flat).
    // For mmap it avoids per-iteration page-fault + allocation overhead.
    const float* raw_data;
    std::vector<float> mmap_cache;

    if (!use_mmap) {
        raw_data = flat_database.data();
    } else {
        mmap_cache.resize(static_cast<size_t>(num_vectors) * dim);
        for (int i = 0; i < num_vectors; i++) {
            const float* src = raw_vec_ptr(i);
            std::copy(src, src + dim, mmap_cache.data() + i * dim);
        }
        raw_data = mmap_cache.data();
    }

    // Random initialisation: shuffle indices and pick first num_clusters.
    std::vector<int> indices(num_vectors);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), g);

    centroids.resize(num_clusters);
    for (int i = 0; i < num_clusters; i++) {
        const float* src = raw_data + indices[i] * dim;
        centroids[i].assign(src, src + dim);
    }

    std::vector<int> assignments(num_vectors);

    for (int it = 0; it < epochs; it++) {
        // Flat centroid accumulator — avoids row-of-rows scatter during update.
        std::vector<float> new_centroids(num_clusters * dim, 0.0f);
        std::vector<int> counts(num_clusters, 0);

        for (int i = 0; i < num_vectors; i++) {
            const float* vec = raw_data + i * dim;
            float min_d = std::numeric_limits<float>::max();
            int best_c = -1;

            for (int c = 0; c < num_clusters; c++) {
                float d = compute_dist(vec, centroids[c].data(), dim, use_simd, metric);
                if (d < min_d) { min_d = d; best_c = c; }
            }

            assignments[i] = best_c;
            float* acc = new_centroids.data() + best_c * dim;
            for (int d = 0; d < dim; d++) acc[d] += vec[d];
            counts[best_c]++;
        }

        for (int c = 0; c < num_clusters; c++) {
            if (counts[c] > 0) {
                float inv = 1.0f / counts[c];
                float* src = new_centroids.data() + c * dim;
                for (int d = 0; d < dim; d++) centroids[c][d] = src[d] * inv;
            }
        }

        std::cout << "KMeans epoch [" << it + 1 << "/" << epochs << "] done.\n";
    }

    inverted_lists.clear();
    inverted_lists.resize(num_clusters);
    for (int i = 0; i < num_vectors; i++)
        inverted_lists[assignments[i]].push_back(i);

    is_indexed = true;
    std::cout << "Indexing complete.\n";
}

// ---------------------------------------------------------------------------
// search — ANN with IVF + nprobe, returning top-k results.
//
// Algorithm:
//  1. Validate query dimension.
//  2. If no index: brute-force over all vectors.
//     If indexed: score all centroids, sort, pick top-nprobe clusters.
//  3. Scan candidate vectors; maintain a max-heap of size k.
//     Heap top = worst of the best-k seen so far; prune anything worse.
//  4. Return results sorted nearest-first.
//
// Complexity (indexed): O(C·d) centroid scan + O(nprobe·(N/C)·d) bucket scan
//   + O(candidates·log k) heap ops, where C = num_clusters.
// ---------------------------------------------------------------------------
std::vector<std::pair<int, float>> VectorIndex::search(
    const std::vector<float>& query, int k, int nprobe,
    const std::string& metric)
{
    if (static_cast<int>(query.size()) != dim)
        throw std::runtime_error(
            "Query dim=" + std::to_string(query.size()) +
            " != index dim=" + std::to_string(dim));

    std::shared_lock lock(rw_mutex);

    const float* qdata = query.data();
    std::vector<int> candidates;

    if (!is_indexed) {
        candidates.resize(num_vectors);
        std::iota(candidates.begin(), candidates.end(), 0);
    } else {
        // Score all centroids and pick the top-nprobe.
        std::vector<std::pair<float, int>> cdists;
        cdists.reserve(centroids.size());
        for (int c = 0; c < static_cast<int>(centroids.size()); c++) {
            float d = compute_dist(centroids[c].data(), qdata, dim, use_simd, metric);
            cdists.emplace_back(d, c);
        }

        int np = std::min(nprobe, static_cast<int>(centroids.size()));
        std::partial_sort(cdists.begin(), cdists.begin() + np, cdists.end());

        for (int i = 0; i < np; i++)
            for (int vid : inverted_lists[cdists[i].second])
                candidates.push_back(vid);
    }

    // Max-heap: (distance, id). Top = largest distance seen among best-k.
    // We keep at most k entries; replace the top whenever a closer vector is found.
    using Entry = std::pair<float, int>;
    std::priority_queue<Entry> heap;

    for (int vid : candidates) {
        float d = compute_dist(raw_vec_ptr(vid), qdata, dim, use_simd, metric);
        if (static_cast<int>(heap.size()) < k) {
            heap.emplace(d, vid);
        } else if (d < heap.top().first) {
            heap.pop();
            heap.emplace(d, vid);
        }
    }

    // Extract sorted nearest-first.
    std::vector<std::pair<int, float>> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.emplace_back(heap.top().second, heap.top().first);
        heap.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}
