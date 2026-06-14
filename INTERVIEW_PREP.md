# VeloxDB — Interview Prep & Design Notes

A revision sheet for explaining this project in SDE interviews. Covers the *why* behind each decision, the tradeoffs, the things you'd say when an interviewer pushes, and a roadmap of resume-worthy improvements.

> **One-line pitch:** "I built an embedded vector database in C++17 with AVX2 SIMD distance kernels, an IVF (inverted-file) approximate-nearest-neighbor index, memory-mapped persistence for larger-than-RAM datasets, pybind11 Python bindings, and a FastAPI + Next.js layer for semantic text search."

---

## 1. The 30-second / 2-minute / deep-dive pitch

**30 sec:** VeloxDB is a from-scratch vector database. The hot path — distance computation and indexing — is native C++ with hand-written AVX2 SIMD. It uses K-Means clustering to build an IVF index so search is approximate but sub-linear instead of brute force. Data persists in the `.fvecs` binary format and is memory-mapped on load so you can query datasets bigger than RAM. A pybind11 layer exposes it to Python, and a FastAPI server + Next.js UI turn it into a semantic search engine using sentence-transformer embeddings.

**2 min:** Add the layering (C++ core → pybind11 → FastAPI → Next.js), the two distance metrics (Euclidean / cosine, each with scalar + SIMD variants), the brute-force-vs-IVF search modes, and the persistence story (separate files for vectors, index, and metadata).

**Deep dive:** Be ready to whiteboard the IVF search (compare query to *k* centroids → pick nearest cluster → exhaustive search inside that one bucket), the K-Means training loop, the `.fvecs` byte layout, and how `mmap` + pointer arithmetic gives you zero-copy random access to any vector.

---

## 2. Core concepts you must be able to explain

### Vector / embedding similarity search
- An **embedding** maps text/images into a fixed-dim float vector where semantic similarity ≈ geometric proximity.
- "Search" = find the vector(s) closest to a query vector under some distance metric.
- This project uses `all-MiniLM-L6-v2` (384-dim) via `sentence-transformers` on the server side.

### Distance metrics ([src/metrics.cpp](src/metrics.cpp))
- **Euclidean (L2):** `Σ(aᵢ−bᵢ)²`. Note it returns the **squared** distance — you don't need the `sqrt` to rank nearest neighbors, so it's skipped for speed. Good talking point.
- **Cosine distance:** `1 − (a·b)/(‖a‖‖b‖)`. Measures angle, scale-invariant — preferred for normalized text embeddings.
- Each metric has a scalar and an AVX2 variant, selectable at runtime via `set_simd()`.

### SIMD / AVX2 ([src/metrics.cpp](src/metrics.cpp))
- `__m256` holds **8 floats**; one `_mm256_*_ps` instruction processes all 8 lanes at once (data-level parallelism).
- Pattern: accumulate partial sums in a vector register across the bulk of the array, then a **horizontal reduction** (store to buffer, sum the 8 lanes), then a **scalar tail loop** for the remaining `n % 8` elements.
- Uses `_mm256_loadu_ps` (**u** = unaligned) so it works on arbitrary `std::vector` data without alignment guarantees.
- **Tradeoff:** requires AVX2-capable CPU; falls back to scalar. Float accumulation order differs between scalar and SIMD, so results aren't bit-identical (floating-point non-associativity) — fine for ANN, worth mentioning.

### IVF index (Inverted File) ([src/index.cpp](src/index.cpp))
- **Train:** run K-Means to produce `num_clusters` centroids; assign every vector to its nearest centroid; build `inverted_lists[c] = [vector ids in cluster c]`.
- **Search:** compare query to the `num_clusters` centroids (cheap), pick the nearest cluster, then do an exhaustive scan **only within that bucket**.
- Turns an O(N·d) brute-force scan into roughly O(k·d + (N/k)·d). With `k ≈ √N` that's ~O(√N·d).
- **It's approximate (ANN):** the true nearest neighbor can live in a *neighboring* cluster you didn't probe. This is the recall/speed tradeoff at the heart of the project.

### K-Means ([src/index.cpp](src/index.cpp) `build_index`)
- Lloyd's algorithm: random init (shuffle indices, take first *k*) → assign step → update step (mean of assigned vectors) → repeat for `epochs`.
- **Talking points:** sensitive to init (k-means++ would be better), can produce empty clusters (the code guards `counts[c] > 0`), runs a fixed iteration count with no convergence check.

### Memory-mapped I/O ([src/index.cpp](src/index.cpp) `load_fvecs` / `get_vector`)
- `mmap` maps the file into the process address space; the OS pages data in on demand instead of `read()`-ing the whole file into RAM.
- `get_vector(i)` does **pointer arithmetic** into the mapped region — `base + i*row_size + sizeof(int)` — giving zero-copy O(1) random access.
- `MAP_PRIVATE | PROT_READ` → read-only; that's why `add_vector` throws when `use_mmap` is true (you can't append to a memory-mapped read-only file).

### The `.fvecs` format
- Per vector: a 4-byte `int` dimension header, followed by `dim` little-endian `float32`s.
- `num_vectors = file_size / (4 + dim*4)`. Self-describing, trivially seekable, FAISS-compatible.

---

## 3. Architecture & layering

```
Next.js UI (web/)  ──HTTP──▶  FastAPI (server/)  ──pybind11──▶  C++ core (src/, include/)
   ingest/search/train          embeds text,                    VectorIndex:
                                 holds global state,              storage, KMeans/IVF,
                                 metadata JSON                    SIMD metrics, mmap
```

- **Why C++ core:** distance math is the hot path; manual memory layout + SIMD + no GC.
- **Why pybind11:** expose the fast core to Python's ecosystem (embeddings, web frameworks) without rewriting it.
- **Why split files** (`vectors.fvecs`, `index.ivf`, `metadata.json`): vectors are the source of truth, the index is a derived/rebuildable artifact, and text metadata is a Python-side concern the C++ core doesn't know about. Clean separation of responsibilities.
- **Server state** ([server/state.py](server/state.py)): a single global `VectorIndex` plus `vector_count`/`dim`/`is_indexed` flags, hydrated from disk on startup.

---

## 4. Key tradeoff decisions (interviewers love these)

| Decision | Chosen | Alternative | Why / tradeoff |
|---|---|---|---|
| Index type | IVF (flat) | HNSW graph, brute force | IVF is simple, low-memory, fast to build; HNSW has better recall/latency but is far more complex. Brute force is exact but O(N). |
| Distance return | squared L2 (no sqrt) | true L2 | sqrt is monotonic — unnecessary for ranking. Saves a sqrt per comparison. |
| SIMD | runtime toggle (AVX2) | always-on / compile-time / AVX-512 | Portability + ability to A/B benchmark scalar vs SIMD. |
| Storage | `std::vector<std::vector<float>>` in RAM, or mmap | flat contiguous array; on-disk index | Simple; but row-of-rows is *not* cache-contiguous (see Known Issues). |
| Persistence | custom `.fvecs` + raw binary index | SQLite, protobuf, FAISS index | Minimal deps, full control, FAISS-compatible vectors. No schema evolution/versioning. |
| Search result | single nearest id (top-1) | top-k with scores | Simpler API; but top-1 only is a real product limitation. |
| `nprobe` | hardcoded to 1 cluster | tunable nprobe | 1 cluster = fastest but lowest recall. Biggest easy win to expose. |
| Bindings | pybind11 | ctypes, Cython, raw CPython API | Ergonomic C++↔Python, header-only, STL conversions for free. |

---

## 5. Complexity cheat-sheet

| Operation | Cost |
|---|---|
| `add_vector` | O(d) amortized |
| Brute-force search | O(N·d) |
| IVF train (K-Means) | O(epochs · N · k · d) |
| IVF search (nprobe=1) | O(k·d) centroid scan + O((N/k)·d) bucket scan |
| `get_vector` (mmap) | O(d) copy, O(1) seek |
| SIMD speedup | ~theoretical 8× on the distance inner loop (float32 lanes) |

---

## 6. Known issues / things to fix (great "what would you improve" answers)

These are real, in the current code — owning them shows engineering maturity:

1. **IVF centroid-selection bug** — [src/index.cpp:318](src/index.cpp#L318): `min_c_dist = c;` assigns the **cluster index** instead of the distance `d`. The "nearest centroid" logic is broken; it doesn't actually track the minimum distance. *(This is the single most important thing to fix and a perfect "I found a bug in my own code" story.)*
2. **`nprobe` hardcoded to 1** — only the single nearest cluster is searched, capping recall. Should be a parameter; searching the top-`nprobe` clusters trades latency for recall.
3. **Top-1 only** — `search` returns one id. Real vector DBs return top-k with distances (needs a max-heap / partial sort).
4. **K-Means copies per access** — the training loop calls `get_vector(i)` (which allocates a `std::vector`) inside nested loops; with mmap that's a copy every iteration. Cache a contiguous buffer instead.
5. **Non-contiguous storage** — `vector<vector<float>>` scatters rows across the heap, defeating cache locality and undercutting the SIMD wins. A single flat `float*` of `N*d` would be much better.
6. **No thread safety** — the FastAPI server mutates global `VectorIndex` state with no locking; concurrent `add`/`train`/`search` will race.
7. **No index versioning / magic bytes** — `.fvecs` and `.ivf` files have no version header; format changes silently corrupt old data.
8. **`max_iters` vs `epochs` naming mismatch** between the Python signature (`max_iters`) and C++ (`epochs`) — minor, but worth a consistency pass.
9. **No input validation on dim** at the C++ search boundary beyond the index path.

---

## 7. Likely interview questions & crisp answers

- **"Why is your search approximate?"** IVF only scans the nearest cluster(s); the true NN can be just across a cluster boundary. You trade exactness for sub-linear latency, tuned by `num_clusters` and `nprobe`.
- **"How does SIMD give a speedup?"** One AVX2 instruction does 8 float ops; the distance loop is the bottleneck, so vectorizing it ~8× the arithmetic throughput, minus the horizontal-reduction and tail-loop overhead.
- **"How do you handle data bigger than RAM?"** `mmap` the `.fvecs` file; the OS pages it in on demand and `get_vector` does pointer arithmetic for zero-copy random access.
- **"Why `num_clusters ≈ √N`?"** Balances the two halves of IVF cost: centroid scan O(k) grows with k, bucket scan O(N/k) shrinks with k; the sum is minimized near √N.
- **"What's the failure mode of K-Means here?"** Bad random init → poor clusters; no convergence check (fixed epochs); empty clusters; sensitive to scale (cosine vs L2 matters).
- **"How would you make it production-grade?"** See roadmap below — concurrency, top-k, deletions/updates, HNSW, quantization, versioned formats, observability.
- **"Why C++ and Python together?"** Performance-critical kernels in C++, ecosystem/glue (embeddings, HTTP) in Python; pybind11 bridges them.

---

## 8. Resume-worthy roadmap (ordered by impact ÷ effort)

**High impact, low effort:**
1. **Fix the centroid bug + add `nprobe`** — instantly turns "broken approximate search" into a real recall/latency knob. Then **measure recall@k vs nprobe** and put a graph in the README. *Benchmarks are what make this resume-strong.*
2. **Top-k search** with distances (partial sort / max-heap of size k).
3. **Contiguous flat storage** (`std::vector<float>` of size N·d) — cleaner SIMD, real cache wins; benchmark before/after.
4. **A proper benchmark harness** — run against a standard dataset (e.g. SIFT1M from the ANN-benchmarks suite), report QPS, recall@10, build time, and brute-force-vs-IVF-vs-SIMD-vs-scalar. Numbers on a resume beat adjectives.

**Medium effort, strong signal:**
5. **Thread safety + concurrency** — a read-write lock so the server handles concurrent searches; or sharding. Talk about it as a real systems problem.
6. **Delete / update support** — tombstones + periodic re-index; this is genuinely hard in IVF and shows depth.
7. **HNSW index** as an alternative — graph-based ANN with better recall/latency; lets you compare two index families.
8. **Scalar quantization (int8) or PQ (product quantization)** — shrink memory 4×+ and speed up distance; classic vector-DB technique, very interview-relevant.
9. **Versioned, checksummed file formats** (magic bytes + version + CRC).

**Polish / breadth:**
10. **Metadata filtering** ("search where category = X") — pre/post-filter on the metadata store.
11. **Batched search** API + SIMD over multiple queries.
12. **Observability** — latency/recall metrics endpoint, structured logging.
13. **CI** — GitHub Actions running the C++ and Python tests, plus the benchmark on a fixed dataset to catch regressions.
14. **Packaging** — multi-platform wheels (cibuildwheel) with runtime AVX2 detection so it works on non-AVX2 machines.

---

## 9. Two-line résumé bullet templates

> *Built **VeloxDB**, an embedded vector database in C++17 with hand-written AVX2 SIMD distance kernels and an IVF/K-Means approximate-nearest-neighbor index, achieving **~Nx** faster search than brute force at **R%** recall@10 on SIFT1M.*

> *Designed memory-mapped `.fvecs` persistence for larger-than-RAM datasets and exposed the core via pybind11 to a FastAPI + Next.js semantic-search stack using sentence-transformer embeddings.*

*(Fill in the **Nx / R%** once you've run the benchmark in roadmap item #4 — concrete numbers are the difference between a good and a great bullet.)*

---

## 10. File map (where to point during a walkthrough)

| Area | File |
|---|---|
| Core class / API surface | [include/vector_db.hpp](include/vector_db.hpp) |
| Storage, K-Means, IVF, mmap, search | [src/index.cpp](src/index.cpp) |
| Distance kernels (scalar + AVX2) | [src/metrics.cpp](src/metrics.cpp) · [include/metrics.hpp](include/metrics.hpp) |
| Python bindings | [bindings/python_bindings.cpp](bindings/python_bindings.cpp) |
| REST API | [server/app.py](server/app.py) |
| Server state / persistence | [server/state.py](server/state.py) |
| Embeddings | [server/embedder.py](server/embedder.py) |
| Web UI | [web/app/](web/app/) |
</content>
</invoke>
