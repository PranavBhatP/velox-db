# VeloxDB Copilot Instructions

## Project Overview
VeloxDB is a high-performance embedded vector database implemented in C++17 with AVX2 optimizations, exposed to Python via pybind11. It features IVF indexing with K-Means clustering and memory-mapped file support.

## Architecture
- **Core (C++)**: Located in `src/` and `include/`. Implements vector storage, metric calculations (Euclidean), and indexing algorithms.
  - `VectorIndex`: Main class managing storage and search.
  - `veloxdb_core`: Shared library containing the logic.
- **Bindings (C++/Python)**: `bindings/python_bindings.cpp` defines the `veloxdb` Python module using `pybind11`.
- **Server (Python)**: `server/main.py` is a FastAPI application providing a REST API over the `veloxdb` module.
- **Build System**: Uses `scikit-build-core` and `CMake` to compile the C++ extension.

## Build & Development
- **Build Command**: `pip install .` or `pip install -e .` (editable install recommended for dev).
- **Dependencies**: `cmake`, `pybind11`, `scikit-build-core` (defined in `pyproject.toml`).
- **Compiler Flags**: AVX2 is enabled by default (`/arch:AVX2` on MSVC, `-mavx2` on others).

## Testing & Debugging
- **Test Scripts**: Located in `tests/`.
- **Running Tests**: `python tests/test_script.py`.
  - **Note**: Tests often manually append `build/` to `sys.path` to find the compiled `veloxdb` module if not installed in the environment.
- **Server**: Run with `python server/main.py`. Ensure `data/` directory exists for persistence.

## Coding Conventions
- **C++**:
  - Use C++17 standard.
  - Headers in `include/`, implementation in `src/`.
  - Optimize for performance (SIMD, memory mapping).
- **Python**:
  - Use type hints (e.g., `list[float]`).
  - Follow FastAPI patterns for the server.
  - Handle `ImportError` for `veloxdb` gracefully in scripts.

## Key Files
- `CMakeLists.txt`: Build configuration.
- `bindings/python_bindings.cpp`: Python API definition.
- `server/main.py`: REST API entry point.
- `src/index.cpp`: Core indexing logic.
