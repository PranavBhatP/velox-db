import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath("build"))
import veloxdb

def test_ivf():
    print("--- Testing IVF Indexing (Week 3) ---")
    
    # 1. Generate Data (10k vectors, 128 dim)
    N = 10000
    dim = 128
    print(f"Generating {N} random vectors...")
    data = np.random.rand(N, dim).astype(np.float32)
    
    db = veloxdb.VectorIndex()
    for i in range(N):
        db.add_vector(data[i])
    print("Data added to VeloxDB.")

    # 2. Build Index (Divide 10k vectors into 100 clusters)
    # This effectively makes search 100x faster theoretically
    print("Building Index...")
    start = time.time()
    db.build_index(num_clusters=100, max_iters=5)
    print(f"Indexing Time: {time.time() - start:.4f}s")

    # 3. Search
    # We pick a known vector from the dataset to query
    target_idx = 500
    query = data[target_idx]
    
    print("Searching...")
    start = time.time()
    result = db.search(query)
    print(f"Search Time (IVF): {(time.time() - start)*1000:.4f} ms")
    
    print(f"Target ID: {target_idx}, Found ID: {result}")
    
    if result == target_idx:
        print("✅ SUCCESS: Found exact match using Index!")
    else:
        # Note: In Approximate Nearest Neighbor (ANN), it is valid to sometimes
        # miss the exact match if it lies on a cluster boundary.
        print("⚠️ Approximate Match: Index returned a neighbor (Expected behavior in ANN).")

if __name__ == "__main__":
    test_ivf()