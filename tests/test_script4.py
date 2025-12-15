import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath("build"))
import veloxdb

def benchmark_simd():
    print("--- Week 4: AVX2 SIMD Benchmark ---")
    
    # 1. Setup Data
    N = 1000 # 100k vectors
    dim = 128
    print(f"Generating {N} vectors (dim={dim})...")
    data = np.random.rand(N, dim).astype(np.float32)
    
    db = veloxdb.VectorIndex()
    for i in range(N):
        db.add_vector(data[i])
    
    query = np.random.rand(dim).astype(np.float32)

    # 2. Run WITHOUT SIMD (Scalar)
    print("\n[Mode 1] Scalar Search (Standard C++)")
    db.set_simd(False)
    
    start = time.time()
    # Run 10 times to average
    for _ in range(10):
        db.search(query, metric = "cos")
    end = time.time()
    scalar_time = (end - start) / 10
    print(f"Average Scalar Time: {scalar_time*1000:.4f} ms")

    # 3. Run WITH SIMD (AVX2)
    print("\n[Mode 2] SIMD Search (AVX2 Enabled)")
    db.set_simd(True)
    
    start = time.time()
    for _ in range(10):
        db.search(query, metric = "cos")
    end = time.time()
    simd_time = (end - start) / 10
    print(f"Average SIMD Time:   {simd_time*1000:.4f} ms")

    # 4. Results
    speedup = scalar_time / simd_time
    print(f"\n🚀 Speedup Factor: {speedup:.2f}x")
    
    if speedup > 2.0:
        print("✅ SUCCESS: Significant performance boost detected!")
    else:
        print("⚠️ NOTE: Speedup might be low if dataset is too small or CPU throttled.")

if __name__ == "__main__":
    benchmark_simd()