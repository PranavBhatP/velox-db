"""Read .fvecs files into RAM without mmap."""

import struct
from pathlib import Path


def read_fvecs(path: str | Path) -> list[list[float]]:
    """Parse a .fvecs file: each record is int32 dim + dim float32 values."""
    path = Path(path)
    vectors: list[list[float]] = []
    with path.open("rb") as f:
        while True:
            header = f.read(4)
            if not header:
                break
            if len(header) < 4:
                raise ValueError(f"Truncated header in {path}")
            (dim,) = struct.unpack("i", header)
            payload = f.read(dim * 4)
            if len(payload) < dim * 4:
                raise ValueError(f"Truncated vector data in {path}")
            vec = struct.unpack(f"{dim}f", payload)
            vectors.append(list(vec))
    return vectors
