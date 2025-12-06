#include "vector_db.hpp"
#include <limits>
#include<cmath>
#include<stdexcept>
#include<sys/mman.h>
#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>


VectorIndex::VectorIndex(){
    std::cout << "VectorIndex Initialised!" << std::endl;
}

VectorIndex::~VectorIndex(){
    if(use_mmap && mmap_ptr != nullptr){
        munmap(mmap_ptr, mmap_size); // system call unmap pages of memory.
        std::cout<<"[VeloxDB] memory unmapped and cleaned!" << std::endl;
    }
}

void VectorIndex::add_vector(const std::vector<float> &vec){
    if(use_mmap){
        throw std::runtime_error("Cannot add vectors to a read-only mmap index!");
    }
    database.push_back(vec);
    std::cout << "Vector added. Total size: " << database.size() << std::endl;
}

void VectorIndex::load_fvecs(const std::string& filename){
    int fd = open(filename.c_str(), O_RDONLY);
    if(fd==-1){
        throw std::runtime_error("Could not open file:" + filename);
    }

    struct stat sb;
    if(fstat(fd,&sb)==-1){
        close(fd);
        throw std::runtime_error("Could not read file sizes: " + filename);
    }

    mmap_size = sb.st_size;
    // use PROT_READ for  read protections on the file.
    mmap_ptr = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);

    close(fd); //file is now in memory.

    if(mmap_ptr == MAP_FAILED){
        throw std::runtime_error("mmap failed!!");
    }

    // parse the header of a page (first 4 bytes (int) tells use the dimension)
    data_ptr = static_cast<float*>(mmap_ptr);
    int* header = static_cast<int*>(mmap_ptr);
    dim = header[0];

    //calculate the no. of vecs.
    // size of a single vector is =  bytes (header) + dim * 4 bytes (for each dim)
    size_t row_size_bytes = sizeof(int) + (dim * sizeof(float)); 
    num_vectors = mmap_size / row_size_bytes;
    
    use_mmap = true;
    std :: cout << "[VeloxDB] Loaded successfully! " << num_vectors << " vectors (dim = " << dim << ") using mmap." << std::endl;
}


std::vector<float> VectorIndex::get_vector(int index) {
    if(index < 0 || index >= num_vectors){
        throw std::out_of_range("Index out of bounds");
    }
    
    if(!use_mmap){
        return database[index];
    }

    // calculate the offset.
    char* base = static_cast<char*> (mmap_ptr); // convert void ptr to character for byte level access.
    size_t row_size_bytes = sizeof(int) + (dim * sizeof(float));

    // pointer to the start of the vector we want.
    char* vec_start = base + (index * row_size_bytes) + sizeof(int);
    float* vec_data = reinterpret_cast<float*>(vec_start);
    
    std::vector<float> result(vec_data, vec_data + dim);
    return result;
}

int VectorIndex::search(const std::vector<float> &query){
    int best_index = -1;
    float min_dist = std::numeric_limits<float>::max();

    int cnt = use_mmap ? num_vectors : database.size();

    for(size_t i = 0; i < cnt; i++){
        float dist = 0.0f;

        if(use_mmap){
            char* base = static_cast<char*>(mmap_ptr);
            size_t row_size_bytes = sizeof(int) + (dim*sizeof(float));
            char* vec_start = base + (i*row_size_bytes) + sizeof(int);
            float* vec_data = reinterpret_cast<float*>(vec_start);

            for(int j = 0; j < dim; j++){
                float diff = vec_data[j] - query[j];
                dist += diff*diff;
            }

        } else {
            const auto& vec = database[i];
            for(size_t j = 0; j < vec.size(); j++){
                float diff = vec[j]-query[j];
                dist += diff*diff;
            }
        }
        

        if(dist < min_dist){
            min_dist = dist;
            best_index = i;
        }
    }
    return best_index;
}