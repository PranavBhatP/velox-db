#include "storage.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

VectorStorage::~VectorStorage() {
    if (use_mmap_ && mmap_ptr_ != nullptr)
        munmap(mmap_ptr_, mmap_size_);
}

const float* VectorStorage::raw_vec_ptr(int index) const {
    if (!use_mmap_)
        return flat_database_.data() + index * dim_;

    const char* base = static_cast<const char*>(mmap_ptr_);
    size_t row_bytes = sizeof(int) + dim_ * sizeof(float);
    return reinterpret_cast<const float*>(base + index * row_bytes + sizeof(int));
}

std::vector<float> VectorStorage::get_vector(int index) const {
    if (index < 0 || index >= num_vectors_)
        throw std::out_of_range("Index out of bounds");
    const float* p = raw_vec_ptr(index);
    return std::vector<float>(p, p + dim_);
}

void VectorStorage::add_vector(const std::vector<float>& vec) {
    if (use_mmap_)
        throw std::runtime_error("Cannot add vectors to a read-only mmap index.");
    if (num_vectors_ == 0) {
        dim_ = static_cast<int>(vec.size());
    } else if (static_cast<int>(vec.size()) != dim_) {
        throw std::runtime_error("Vector dimension mismatch.");
    }
    flat_database_.insert(flat_database_.end(), vec.begin(), vec.end());
    num_vectors_++;
}

void VectorStorage::load_fvecs(const std::string& filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1)
        throw std::runtime_error("Could not open file: " + filename);

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Could not stat file: " + filename);
    }

    mmap_size_ = sb.st_size;
    mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mmap_ptr_ == MAP_FAILED)
        throw std::runtime_error("mmap failed.");

    const int* header = static_cast<const int*>(mmap_ptr_);
    dim_ = header[0];
    size_t row_bytes = sizeof(int) + dim_ * sizeof(float);
    num_vectors_ = static_cast<int>(mmap_size_ / row_bytes);
    use_mmap_ = true;

    std::cout << "[VeloxDB] Loaded " << num_vectors_
              << " vectors (dim=" << dim_ << ") via mmap.\n";
}

void VectorStorage::write_fvecs(const std::string& filename) const {
    if (use_mmap_)
        throw std::runtime_error("Already using mmap; cannot export in-memory data.");
    if (num_vectors_ == 0)
        throw std::runtime_error("No data to write.");

    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output file.");

    for (int i = 0; i < num_vectors_; i++) {
        out.write(reinterpret_cast<const char*>(&dim_), sizeof(int));
        out.write(reinterpret_cast<const char*>(flat_database_.data() + i * dim_),
                  dim_ * sizeof(float));
    }
    out.close();
    std::cout << "Wrote " << num_vectors_ << " vectors to " << filename << "\n";
}
