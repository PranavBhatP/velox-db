#include "vector_db.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <random>
#include <algorithm>
#include <numeric>

float euclidean_dist(const std::vector<float> &a, const std::vector<float> &b){
    float dist = 0.0f;
    for(size_t i = 0; i < a.size(); i++){
        float diff = a[i]-b[i];
        dist += diff * diff;
    }
    return dist;
}


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
    num_vectors++;
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


//build the db index using k-means clustering
void VectorIndex::build_index(int num_clusters, int epochs){
    if(num_vectors <  num_clusters){
        throw std::runtime_error("Not enough vectors to assign to clusters!");
    }

    std::cout << "Training the IVF Index with " << num_clusters << "clusters." << '\n';

    //pick k random vectors from your data to represent centroids.
    centroids.resize(num_clusters);
    std::vector<int> indices(num_vectors); // table for assigning indices to vectors.
    std::iota(indices.begin(), indices.end(), 0);


    //non-deterministic random number generator (truly unpredictable).
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(),indices.end(), g);
    
    for(int i = 0; i < num_clusters; i++){
        centroids[i] = get_vector(indices[i]);
    }

    std::vector<int> assignments(num_vectors); //maps vectors->centroid.
    
    for(int it = 0; it < epochs; ++it){
        std::vector<std::vector<float>> new_centroids(num_clusters, std::vector<float>(dim, 0.0f));
        std::vector<int> counts(num_clusters, 0);

        for(int i = 0; i < num_vectors; i++){
            std::vector<float> vec = get_vector(i);
            float min_d = std::numeric_limits<float>::max();

            int best_c = -1;
            for(int c = 0; c < num_clusters; c++){
                float d = euclidean_dist(vec, centroids[c]);
                if(d < min_d){ 
                    min_d = d;
                    best_c = c;
                }
            }

            assignments[i] = best_c;

            for(int d = 0;d < dim; d++){
                new_centroids[best_c][d] += vec[d]; // sum across all the vectors whose best_c is c.
            }
            counts[best_c]++;
        }
        //take average of all the sum and assign these values to centrodi.
        for(int c = 0; c < num_clusters; c++){
            if(counts[c] > 0){
                for(int d = 0; d < dim; d++) centroids[c][d] = new_centroids[c][d] / counts[c];
            }
        }

        std::cout << "KMeans epoch[" << it + 1 << "/" << epochs << "] completed." << '\n'; 
    }

    inverted_lists.clear();
    inverted_lists.resize(num_clusters);
    for(int i = 0; i < num_vectors; i++){
        inverted_lists[assignments[i]].push_back(i);
    }

    is_indexed=true;
    std::cout << "Indexing complete!\n";
}

// old code which uses brute force in two methods - in-memory vector and one using pointers and mmap.
// int VectorIndex::search(const std::vector<float> &query){
//     int best_index = -1;
//     float min_dist = std::numeric_limits<float>::max();

//     int cnt = use_mmap ? num_vectors : database.size();

//     for(size_t i = 0; i < cnt; i++){
//         float dist = 0.0f;

//         if(use_mmap){
//             char* base = static_cast<char*>(mmap_ptr);
//             size_t row_size_bytes = sizeof(int) + (dim*sizeof(float));
//             char* vec_start = base + (i*row_size_bytes) + sizeof(int);
//             float* vec_data = reinterpret_cast<float*>(vec_start);

//             for(int j = 0; j < dim; j++){
//                 float diff = vec_data[j] - query[j];
//                 dist += diff*diff;
//             }

//         } else {
//             const auto& vec = database[i];
//             for(size_t j = 0; j < vec.size(); j++){
//                 float diff = vec[j]-query[j];
//                 dist += diff*diff;
//             }
//         }
        

//         if(dist < min_dist){
//             min_dist = dist;
//             best_index = i;
//         }
//     }
//     return best_index;
// }

int VectorIndex::search(const std::vector<float> &query){
    if(!is_indexed){
        float min_dist = std::numeric_limits<float>::max();
        int best_idx = -1;
        for(int i = 0; i < num_vectors; i++){
            float d = euclidean_dist(get_vector(i), query);
            if(d < min_dist) {
                min_dist = d;
                best_idx = i;
            }
        }
        return best_idx;
    } else {
        //ivf search
        float min_c_dist = std::numeric_limits<float>::max();
        int best_c = -1;
        for(int c = 0; c < centroids.size(); c++){
            float d = euclidean_dist(centroids[c], query);
            if(d < min_c_dist){
                min_c_dist = c;
                best_c = c;
            }            
        }
        //fine search on best_c centroid's vectors.
        float min_dist = std::numeric_limits<float>::max();
        int best_idx = -1;
        
        const auto& bucket = inverted_lists[best_c];
        for (int vec_id : bucket) {
            float d = euclidean_dist(get_vector(vec_id), query);
            if (d < min_dist) { min_dist = d; best_idx = vec_id; }
        }
        return best_idx;
    }
    return -1;
}