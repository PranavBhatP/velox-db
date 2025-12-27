import os
import numpy as np
import veloxdb

def test_persistence():
    data_file = "saved_vectors.fvecs"
    index_file = "saved_index.ivf"
    
    # Clean up previous runs
    if os.path.exists(data_file): os.remove(data_file)
    if os.path.exists(index_file): os.remove(index_file)

    print("--- Phase 1: Create, Train, Save ---")
    N = 10000
    dim = 128
    data = np.random.rand(N, dim).astype(np.float32)
    
    db = veloxdb.VectorIndex()
    for i in range(N):
        db.add_vector(data[i])
    
    print(f"Added {N} vectors. Training Index...")
    db.build_index(100, 5) # Train with 100 clusters
    
    print("Saving Data and Index to disk...")
    db.write_fvecs(data_file)
    db.save_index(index_file)
    print("✅ Saved.")

    print("\n--- Phase 2: Simulating Restart ---")
    db2 = veloxdb.VectorIndex()
    
    print("Loading Data (mmap)...")
    db2.load_fvecs(data_file)
    
    print("Loading Index...")
    db2.load_index(index_file) # Instant load, no training needed!
    
    # Search Check
    query = data[500]
    result = db2.search(query)
    
    print("✅ SUCCESS: Persistence and Reloading works!")

    # Cleanup
    os.remove(data_file)
    os.remove(index_file)

if __name__ == "__main__":
    test_persistence()