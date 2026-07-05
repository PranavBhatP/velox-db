#include "ivf_index.hpp"
#include <algorithm>
#include <numeric>
#include <queue>
#include <limits>
#include <random>
#include <stdexcept>
#include <iostream>

// ---------------------------------------------------------------------------
// build — K-Means IVF training.
//
// Pre-caches mmap-backed vectors into a flat buffer before training starts,
// to avoid per-iteration page-fault + allocation overhead. In-memory storage
// is already flat, so no copy is needed there.
// ---------------------------------------------------------------------------
void IVFIndex::build(const VectorStorage& storage, const IndexParams& params) {
    int num_vectors = storage.size();
    dim_ = storage.dim();
    int num_clusters = params.num_clusters;
    int epochs = params.epochs;

    if (num_vectors < num_clusters)
        throw std::runtime_error("Not enough vectors to fill " +
                                 std::to_string(num_clusters) + " clusters.");

    std::cout << "Training IVF index: " << num_clusters
              << " clusters, " << epochs << " epochs.\n";

    const float* raw_data;
    std::vector<float> mmap_cache;

    if (!storage.is_mmapped()) {
        raw_data = storage.raw_vec_ptr(0);
    } else {
        mmap_cache.resize(static_cast<size_t>(num_vectors) * dim_);
        for (int i = 0; i < num_vectors; i++) {
            const float* src = storage.raw_vec_ptr(i);
            std::copy(src, src + dim_, mmap_cache.data() + i * dim_);
        }
        raw_data = mmap_cache.data();
    }

    std::vector<int> indices(num_vectors);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), g);

    centroids_.resize(num_clusters);
    for (int i = 0; i < num_clusters; i++) {
        const float* src = raw_data + indices[i] * dim_;
        centroids_[i].assign(src, src + dim_);
    }

    std::vector<int> assignments(num_vectors);

    for (int it = 0; it < epochs; it++) {
        std::vector<float> new_centroids(num_clusters * dim_, 0.0f);
        std::vector<int> counts(num_clusters, 0);

        for (int i = 0; i < num_vectors; i++) {
            const float* vec = raw_data + i * dim_;
            float min_d = std::numeric_limits<float>::max();
            int best_c = -1;

            for (int c = 0; c < num_clusters; c++) {
                float d = compute_dist(vec, centroids_[c].data(), dim_, params.use_simd, params.metric);
                if (d < min_d) { min_d = d; best_c = c; }
            }

            assignments[i] = best_c;
            float* acc = new_centroids.data() + best_c * dim_;
            for (int d = 0; d < dim_; d++) acc[d] += vec[d];
            counts[best_c]++;
        }

        for (int c = 0; c < num_clusters; c++) {
            if (counts[c] > 0) {
                float inv = 1.0f / counts[c];
                float* src = new_centroids.data() + c * dim_;
                for (int d = 0; d < dim_; d++) centroids_[c][d] = src[d] * inv;
            }
        }

        std::cout << "KMeans epoch [" << it + 1 << "/" << epochs << "] done.\n";
    }

    inverted_lists_.clear();
    inverted_lists_.resize(num_clusters);
    for (int i = 0; i < num_vectors; i++)
        inverted_lists_[assignments[i]].push_back(i);

    built_ = true;
    std::cout << "Indexing complete.\n";
}

// ---------------------------------------------------------------------------
// search — score all centroids, probe the nprobe closest, top-k over their
// inverted lists via a bounded max-heap.
// ---------------------------------------------------------------------------
std::vector<std::pair<int, float>> IVFIndex::search(
    const VectorStorage& storage, const float* query, int k,
    const IndexParams& params, bool use_simd) const
{
    int dim = storage.dim();
    std::vector<int> candidates;

    std::vector<std::pair<float, int>> cdists;
    cdists.reserve(centroids_.size());
    for (int c = 0; c < static_cast<int>(centroids_.size()); c++) {
        float d = compute_dist(centroids_[c].data(), query, dim, use_simd, params.metric);
        cdists.emplace_back(d, c);
    }

    int np = std::min(params.nprobe, static_cast<int>(centroids_.size()));
    std::partial_sort(cdists.begin(), cdists.begin() + np, cdists.end());

    for (int i = 0; i < np; i++)
        for (int vid : inverted_lists_[cdists[i].second])
            candidates.push_back(vid);

    using Entry = std::pair<float, int>;
    std::priority_queue<Entry> heap;

    for (int vid : candidates) {
        float d = compute_dist(storage.raw_vec_ptr(vid), query, dim, use_simd, params.metric);
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

void IVFIndex::save(std::ofstream& out) const {
    int num_clusters = static_cast<int>(centroids_.size());
    out.write(reinterpret_cast<const char*>(&num_clusters), sizeof(int));

    for (const auto& c : centroids_)
        out.write(reinterpret_cast<const char*>(c.data()), dim_ * sizeof(float));

    for (const auto& lst : inverted_lists_) {
        int sz = static_cast<int>(lst.size());
        out.write(reinterpret_cast<const char*>(&sz), sizeof(int));
        out.write(reinterpret_cast<const char*>(lst.data()), sz * sizeof(int));
    }
}

void IVFIndex::load(std::ifstream& in, int dim) {
    dim_ = dim;
    int num_clusters;
    in.read(reinterpret_cast<char*>(&num_clusters), sizeof(int));

    centroids_.resize(num_clusters);
    for (int i = 0; i < num_clusters; i++) {
        centroids_[i].resize(dim_);
        in.read(reinterpret_cast<char*>(centroids_[i].data()), dim_ * sizeof(float));
    }

    inverted_lists_.resize(num_clusters);
    for (int i = 0; i < num_clusters; i++) {
        int sz;
        in.read(reinterpret_cast<char*>(&sz), sizeof(int));
        inverted_lists_[i].resize(sz);
        in.read(reinterpret_cast<char*>(inverted_lists_[i].data()), sz * sizeof(int));
    }

    built_ = true;
}

void IVFIndex::load_legacy_v1(std::ifstream& in, int num_clusters, int dim) {
    dim_ = dim;

    centroids_.resize(num_clusters);
    for (int i = 0; i < num_clusters; i++) {
        centroids_[i].resize(dim_);
        in.read(reinterpret_cast<char*>(centroids_[i].data()), dim_ * sizeof(float));
    }

    inverted_lists_.resize(num_clusters);
    for (int i = 0; i < num_clusters; i++) {
        int sz;
        in.read(reinterpret_cast<char*>(&sz), sizeof(int));
        inverted_lists_[i].resize(sz);
        in.read(reinterpret_cast<char*>(inverted_lists_[i].data()), sz * sizeof(int));
    }

    built_ = true;
}
