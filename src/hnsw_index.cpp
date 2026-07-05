#include "hnsw_index.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

int HNSWIndex::random_level() {
    double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    double level_mult = 1.0 / std::log(static_cast<double>(M_));
    return static_cast<int>(std::floor(-std::log(r) * level_mult));
}

// ---------------------------------------------------------------------------
// search_layer — best-first traversal of a single layer's graph.
// Maintains a min-heap of candidates to expand and a max-heap of the best
// `ef` results seen so far; stops expanding once the closest remaining
// candidate is farther than the current worst kept result.
// ---------------------------------------------------------------------------
std::vector<std::pair<float, int>> HNSWIndex::search_layer(
    const VectorStorage& storage, const float* query, int entry, int ef,
    int layer, bool use_simd, const std::string& metric) const
{
    int dim = storage.dim();
    auto dist_to = [&](int id) {
        return compute_dist(storage.raw_vec_ptr(id), query, dim, use_simd, metric);
    };

    using Entry = std::pair<float, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> candidates;
    std::priority_queue<Entry> results; // max-heap: top() = worst of the best-ef

    std::unordered_set<int> visited;
    float entry_dist = dist_to(entry);
    visited.insert(entry);
    candidates.emplace(entry_dist, entry);
    results.emplace(entry_dist, entry);

    while (!candidates.empty()) {
        auto [cur_dist, cur_id] = candidates.top();
        candidates.pop();

        if (static_cast<int>(results.size()) >= ef && cur_dist > results.top().first)
            break;

        for (int neighbor : nodes_[cur_id].neighbors[layer]) {
            if (visited.count(neighbor)) continue;
            visited.insert(neighbor);

            float d = dist_to(neighbor);
            if (static_cast<int>(results.size()) < ef || d < results.top().first) {
                candidates.emplace(d, neighbor);
                results.emplace(d, neighbor);
                if (static_cast<int>(results.size()) > ef)
                    results.pop();
            }
        }
    }

    std::vector<Entry> out;
    out.reserve(results.size());
    while (!results.empty()) {
        out.push_back(results.top());
        results.pop();
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ---------------------------------------------------------------------------
// build — insert vectors one at a time: descend greedily from the current
// entry point down to the new node's level, then at each layer from that
// level down to 0, find ef_construction candidates and connect to the
// closest M (M_max0 at layer 0), pruning any neighbor that overflows its cap.
// ---------------------------------------------------------------------------
void HNSWIndex::build(const VectorStorage& storage, const IndexParams& params) {
    int n = storage.size();
    M_ = params.M;
    M_max0_ = 2 * M_;
    ef_construction_ = params.ef_construction;

    nodes_.assign(n, Node{});
    entry_point_ = -1;
    max_level_ = -1;

    for (int i = 0; i < n; i++) {
        int level = random_level();
        nodes_[i].level = level;
        nodes_[i].neighbors.resize(level + 1);

        const float* vec_i = storage.raw_vec_ptr(i);

        if (entry_point_ == -1) {
            entry_point_ = i;
            max_level_ = level;
            continue;
        }

        int ep = entry_point_;
        for (int lc = max_level_; lc > level; lc--) {
            auto res = search_layer(storage, vec_i, ep, 1, lc, params.use_simd, params.metric);
            if (!res.empty()) ep = res.front().second;
        }

        for (int lc = std::min(level, max_level_); lc >= 0; lc--) {
            auto candidates = search_layer(storage, vec_i, ep, ef_construction_, lc, params.use_simd, params.metric);
            if (candidates.empty()) continue;

            int cap = (lc == 0) ? M_max0_ : M_;
            int take = std::min(cap, static_cast<int>(candidates.size()));

            for (int t = 0; t < take; t++) {
                int neighbor_id = candidates[t].second;
                nodes_[i].neighbors[lc].push_back(neighbor_id);

                auto& nlist = nodes_[neighbor_id].neighbors[lc];
                nlist.push_back(i);

                int neighbor_cap = (lc == 0) ? M_max0_ : M_;
                if (static_cast<int>(nlist.size()) > neighbor_cap) {
                    const float* nvec = storage.raw_vec_ptr(neighbor_id);
                    std::vector<std::pair<float, int>> scored;
                    scored.reserve(nlist.size());
                    for (int nb : nlist)
                        scored.emplace_back(
                            compute_dist(nvec, storage.raw_vec_ptr(nb), storage.dim(), params.use_simd, params.metric),
                            nb);
                    std::sort(scored.begin(), scored.end());
                    nlist.clear();
                    for (int t2 = 0; t2 < neighbor_cap; t2++)
                        nlist.push_back(scored[t2].second);
                }
            }

            ep = candidates.front().second;
        }

        if (level > max_level_) {
            entry_point_ = i;
            max_level_ = level;
        }
    }

    built_ = true;
}

// ---------------------------------------------------------------------------
// search — greedy descent to layer 0, then a best-first search bounded by
// ef_search, returning the k closest results.
// ---------------------------------------------------------------------------
std::vector<std::pair<int, float>> HNSWIndex::search(
    const VectorStorage& storage, const float* query, int k,
    const IndexParams& params, bool use_simd) const
{
    if (entry_point_ == -1) return {};

    int ep = entry_point_;
    for (int lc = max_level_; lc > 0; lc--) {
        auto res = search_layer(storage, query, ep, 1, lc, use_simd, params.metric);
        if (!res.empty()) ep = res.front().second;
    }

    int ef = std::max(params.ef_search, k);
    auto candidates = search_layer(storage, query, ep, ef, 0, use_simd, params.metric);

    int take = std::min(k, static_cast<int>(candidates.size()));
    std::vector<std::pair<int, float>> results;
    results.reserve(take);
    for (int i = 0; i < take; i++)
        results.emplace_back(candidates[i].second, candidates[i].first);
    return results;
}

void HNSWIndex::save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&M_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&M_max0_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&entry_point_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&max_level_), sizeof(int));

    int num_nodes = static_cast<int>(nodes_.size());
    out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(int));

    for (const auto& node : nodes_) {
        out.write(reinterpret_cast<const char*>(&node.level), sizeof(int));
        int num_layers = static_cast<int>(node.neighbors.size());
        out.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
        for (const auto& layer_neighbors : node.neighbors) {
            int cnt = static_cast<int>(layer_neighbors.size());
            out.write(reinterpret_cast<const char*>(&cnt), sizeof(int));
            out.write(reinterpret_cast<const char*>(layer_neighbors.data()), cnt * sizeof(int));
        }
    }
}

void HNSWIndex::load(std::ifstream& in, int /*dim*/) {
    in.read(reinterpret_cast<char*>(&M_), sizeof(int));
    in.read(reinterpret_cast<char*>(&M_max0_), sizeof(int));
    in.read(reinterpret_cast<char*>(&ef_construction_), sizeof(int));
    in.read(reinterpret_cast<char*>(&entry_point_), sizeof(int));
    in.read(reinterpret_cast<char*>(&max_level_), sizeof(int));

    int num_nodes;
    in.read(reinterpret_cast<char*>(&num_nodes), sizeof(int));

    nodes_.assign(num_nodes, Node{});
    for (auto& node : nodes_) {
        in.read(reinterpret_cast<char*>(&node.level), sizeof(int));
        int num_layers;
        in.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
        node.neighbors.resize(num_layers);
        for (auto& layer_neighbors : node.neighbors) {
            int cnt;
            in.read(reinterpret_cast<char*>(&cnt), sizeof(int));
            layer_neighbors.resize(cnt);
            in.read(reinterpret_cast<char*>(layer_neighbors.data()), cnt * sizeof(int));
        }
    }

    built_ = true;
}
