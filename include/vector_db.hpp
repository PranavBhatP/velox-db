#pragma once
#include<vector>
#include<iostream>

class VectorIndex  {
    public:
        VectorIndex();
        ~VectorIndex();

        //add a vector into memory.
        void add_vector(const std::vector<float> &vec);

        // load a binary .fvecs file into memory
        // uses the fvecs standard.
        void load_fvecs(const std::string &filename);

        void build_index(int num_clusters, int epochs = 10);

        //get vector by index.
        std::vector<float> get_vector(int index);

        int search(const std::vector<float>& query);
    private:
        std::vector<std::vector<float>> database;

        bool use_mmap = false;
        void *mmap_ptr = nullptr;
        float* data_ptr = nullptr; //adding a typed pointer for direct acess.

        size_t mmap_size = 0;
        int dim = 0;
        int num_vectors = 0;

        bool is_indexed = false;

        // centers of clusters (k vectors per cluster).
        std::vector<std::vector<float>> centroids;

        // This represents the buckets
        // Maps [List_ID] -> [Vector ID, Vector ID, ...]
        //i.e inverted_lists[0] contains all vectors belonging to cluster 0.
        std::vector<std::vector<int>> inverted_lists; 
};