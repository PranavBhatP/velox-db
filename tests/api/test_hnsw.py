import os
import sys

import numpy as np

sys.path.append(os.path.abspath("build"))
import veloxdb


def test_hnsw_build_search_roundtrip():
    print("--- HNSW smoke check ---")

    N, dim = 300, 16
    rng = np.random.default_rng(42)
    data = rng.random((N, dim)).astype(np.float32)

    db = veloxdb.VectorIndex()
    for i in range(N):
        db.add_vector(data[i])

    assert db.get_index_type() == "none"

    db.build_index_hnsw(M=16, ef_construction=200, metric="eucl")
    assert db.get_index_type() == "hnsw"
    print("build_index_hnsw: OK")

    query = data[0]
    results = db.search(query, k=5, metric="eucl", ef_search=50)
    assert len(results) == 5
    assert results[0][0] == 0  # nearest to itself should be itself
    print(f"search: OK ({results})")

    path = "/tmp/velox_hnsw_smoke_test.idx"
    db.save_index(path)

    reloaded = veloxdb.VectorIndex()
    for i in range(N):
        reloaded.add_vector(data[i])
    reloaded.load_index(path)
    os.remove(path)

    assert reloaded.get_index_type() == "hnsw"
    reloaded_results = reloaded.search(query, k=5, metric="eucl", ef_search=50)
    assert reloaded_results == results
    print("save/load roundtrip: OK")

    print("✅ All HNSW smoke checks passed.")


if __name__ == "__main__":
    test_hnsw_build_search_roundtrip()
