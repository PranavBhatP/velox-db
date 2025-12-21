#include "vector_db.hpp"
#include "metrics.hpp"
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
#include<immintrin.h>
#include<fstream>


inline float compute_dist(const float* a, const float* b, int n, bool use_simd, const std::string& metric) {
    if (metric == "cos") {
        return use_simd ? cosine_dist_simd(a, b, n) : cosine_dist(a, b, n);
    } else {
        // Default to Euclidean
        return use_simd ? euclidean_dist_simd(a, b, n) : euclidean_dist(a, b, n);
    }
}

VectorIndex::VectorIndex(){
    std::cout << "VectorIndex Initialised!" << std::endl;
}

VectorIndex::~VectorIndex(){
    if(use_mmap && mmap_ptr != nullptr){
        munmap(mmap_ptr, mmap_size);
    }
}

void VectorIndex::add_vector(const std::vector<float> &vec){
    if(use_mmap){
        throw std::runtime_error("Cannot add vectors to a read-only mmap index!");
    }

    if(database.empty()){
        dim = vec.size();
    } else if (vec.size() != dim) {
        throw std::runtime_error("Vector dimension mismatch!");
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

void VectorIndex::write_fvecs(const std::string &filename){
    if(use_mmap) throw std::runtime_error("Already using mmap, cannot export to store in memory");
    if(database.empty()) throw std::runtime_error("No data to write");

    std::ofstream outfile(filename, std::ios::binary);
    if(!outfile) throw std::runtime_error("File cannot be opened");

    for(const auto& vec: database){
        int d = vec.size();
        outfile.write(reinterpret_cast<const char*>(&d), sizeof(int)); //header for the vector (denoting dimension)
        outfile.write(reinterpret_cast<const char*>(vec.data()), d* sizeof(float));//actual data stored in vector
    }

    outfile.close();
    std::cout << "File has been written successfully!" << std::endl;
}

void VectorIndex::save_index(const std::string &filename){
    if(!is_indexed) throw std::runtime_error("No index to save from!");

    std::ofstream out(filename, std::ios::binary);
    if(!out) throw std::runtime_error("o/p file cannot be opened.");

    //first 8 bytes are metadata.
    int num_clusters = centroids.size();
    //casting should not static - no valid conversion path from float to char.
    out.write(reinterpret_cast<const char*>(&num_clusters),sizeof(int)); 
    out.write(reinterpret_cast<const char*>(&dim), sizeof(int));

    for(const auto& c:centroids){
        out.write(reinterpret_cast<const char*>(c.data()), dim*sizeof(float));
    }

    for(const auto & lst: inverted_lists){
        int lst_size = lst.size();
        out.write(reinterpret_cast<const char*>(&lst_size), sizeof(int));
        out.write(reinterpret_cast<const char*>(lst.data()), lst_size * sizeof(int)); //int->int mapping.
    }
    out.close();
    std::cout << "Index saved to: " << filename << std::endl;
}

void VectorIndex::load_index(const std::string &filename){
    std::ifstream in(filename, std::ios::binary);
    if(!in) throw std::runtime_error("Cannot open index file!");

    int num_clusters;
    int loaded_dim;

    in.read(reinterpret_cast<char*>(&num_clusters), sizeof(int));
    in.read(reinterpret_cast<char*>(&loaded_dim), sizeof(int));

    if(loaded_dim == 0){
        throw std::runtime_error("Zero dims in index file!!");
    }
    std::cout << "Expected dim: " << dim << ", Loaded dim: " << loaded_dim << std::endl;
    if(loaded_dim == 0 || loaded_dim != dim){
        throw std::runtime_error("Dimension mismatch between data and index file!");
    }

    centroids.resize(num_clusters);
    for(int i = 0; i < num_clusters; i++){
        centroids[i].resize(loaded_dim);
        in.read(reinterpret_cast<char*>(centroids[i].data()), loaded_dim*sizeof(float));
    }
    inverted_lists.resize(num_clusters);
    for (int i = 0; i < num_clusters; ++i) {
        int list_size;
        in.read(reinterpret_cast<char*>(&list_size),sizeof(int));
        inverted_lists[i].resize(list_size);
        in.read(reinterpret_cast<char*>(inverted_lists[i].data()), list_size* sizeof(int));
    }

    is_indexed = true;
    std::cout << "Index loaded: " << num_clusters<< " clusters" << std::endl;
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
    float* vec_data = reinterpret_cast<float*>(vec_start); //use reintrepret cast for raw memory intrpretations - like reading a 
    // binary file or smthin - changes the way the compiler looks at the memory andf can be dangerous unless used properly.
    
    std::vector<float> result(vec_data, vec_data + dim);
    return result;
}

void VectorIndex::set_simd(bool enable){
    use_simd = enable;
    std::cout << "SIMD activation status: " << (use_simd ? "Active" : "Inactive") << "\n";
}

//build the db index using k-means clustering
void VectorIndex::build_index(int num_clusters, int epochs, const std::string &metric){
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
                float d = compute_dist(vec.data(), centroids[c].data(), dim, use_simd, metric);
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

int VectorIndex::search(const std::vector<float> &query, const std::string &metric){
    if(!is_indexed){
        float min_dist = std::numeric_limits<float>::max();
        int best_idx = -1;
        for(int i = 0; i < num_vectors; i++){
            float d = compute_dist(get_vector(i).data(), query.data(), dim, use_simd, metric);
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
            float d = compute_dist(centroids[c].data(), query.data(), dim, use_simd, metric);
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
            float d = compute_dist(get_vector(vec_id).data(), query.data(), dim, use_simd, metric);
            if (d < min_dist) { min_dist = d; best_idx = vec_id; }
        }
        return best_idx;
    }
    return -1;
}