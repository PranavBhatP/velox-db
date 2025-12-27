import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath("build"))

try:
    import veloxdb
except ImportError as e:
    print("Failed to import veloxdb module:", e)
    sys.exit(1)

def test_veloxdb():
    # Create a sample numpy array
    db = veloxdb.VectorIndex()
    v1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    v2 = [10.0, 20.0, 30.0, 40.0, 50.0]


    db.add_vector(v1)
    db.add_vector(v2)
    
    retrieved_v1 = db.get_vector(0)

    assert retrieved_v1 == v1, "Retrieved vector does not match the original"

    query = [0.1, 0.2, 0.3, 0.4, 0.5]

    start = time.time()
    results = db.search(query)
    end = time.time()

    assert results == 0, "Search did not return the expected result"

    print("All tests passed.")
if __name__ == "__main__":
    test_veloxdb()