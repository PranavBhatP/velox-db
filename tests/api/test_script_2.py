import sys
import os
import numpy as np
import struct

#######################################################
# Tests generated using an AI. These are not verified.#
#######################################################

sys.path.append(os.path.abspath("build"))
import veloxdb

def generate_dummy_fvecs(filename, num_vectors, dim):
    print(f"Generating {filename} with {num_vectors} vectors of dim {dim}...")
    with open(filename, 'wb') as f:
        for i in range(num_vectors):
            # Create a vector where all elements equal 'i'
            # e.g., vector 0 = [0,0...], vector 5 = [5,5...], vector 999 = [999,999...]
            vec = np.full(dim, float(i), dtype=np.float32)
            
            # .fvecs format: [int32_dim] [float_data]
            f.write(struct.pack('i', dim)) # i is integer format code for struct.
            f.write(vec.tobytes())

def test_mmap():
    filename = "test_data.fvecs"
    dim = 128
    num_vectors = 1000
    
    # 1. Generate Binary File
    generate_dummy_fvecs(filename, num_vectors, dim)
    
    # 2. Load into VeloxDB via mmap
    print("\n--- Testing VeloxDB mmap Loading ---")
    db = veloxdb.VectorIndex()
    db.load_fvecs(filename)
    
    # 3. Check Retrieval
    # Vector 42 should contain all 42.0s
    vec_42 = db.get_vector(42)
    print(f"Retrieved Vector 42 sample: {vec_42[:5]}...")
    assert vec_42[0] == 42.0
    
    # 4. Check Search
    # Query for a vector full of 50.0s -> Should return index 50
    query = [50.0] * dim
    result = db.search(query)
    print(f"Search for [50.0...], found index: {result}")
    
    assert result == 50
    print("✅ SUCCESS: Memory Mapping works!")
    
    # Cleanup
    os.remove(filename)

if __name__ == "__main__":
    test_mmap()