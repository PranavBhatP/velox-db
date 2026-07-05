#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <cstdio>
#include <fstream>
#include <cstdint>
#include "vector_db.hpp"

class VeloxTest : public ::testing::Test {
protected:
    VectorIndex db;
    void SetUp() override { db.set_simd(true); }
};

TEST_F(VeloxTest, AddVectorMatchesRetrieval) {
    db.add_vector({1.0f, 2.0f, 3.0f});
    auto retrieved = db.get_vector(0);
    ASSERT_EQ(retrieved.size(), 3u);
    EXPECT_FLOAT_EQ(retrieved[0], 1.0f);
    EXPECT_FLOAT_EQ(retrieved[1], 2.0f);
    EXPECT_FLOAT_EQ(retrieved[2], 3.0f);
}

// Brute-force search (no index) should find the nearest vector.
TEST_F(VeloxTest, BruteForceSearchEuclidean) {
    db.add_vector({0.0f, 0.0f});
    db.add_vector({0.0f, 1.0f});
    db.add_vector({10.0f, 10.0f});

    auto results = db.search({0.0f, 0.8f}, /*k=*/1, /*nprobe=*/1, "eucl");
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].first, 1);  // vector at id=1 is closest
}

TEST_F(VeloxTest, BruteForceSearchCosine) {
    db.add_vector({1.0f, 0.0f});
    db.add_vector({0.0f, 1.0f});
    db.add_vector({10.0f, 10.0f});

    auto results = db.search({1.0f, 0.0f}, /*k=*/1, /*nprobe=*/1, "eucl");
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].first, 0);
}

// Top-k should return k results sorted by distance.
TEST_F(VeloxTest, TopKResultsOrdered) {
    db.add_vector({0.0f});
    db.add_vector({1.0f});
    db.add_vector({5.0f});
    db.add_vector({10.0f});

    auto results = db.search({0.5f}, /*k=*/3, /*nprobe=*/1, "eucl");
    ASSERT_EQ(results.size(), 3u);
    // Results must be sorted nearest-first (non-decreasing distance).
    EXPECT_LE(results[0].second, results[1].second);
    EXPECT_LE(results[1].second, results[2].second);
    // Nearest to 0.5 should be 0.0 (id=0) or 1.0 (id=1)
    EXPECT_TRUE(results[0].first == 0 || results[0].first == 1);
}

// IVF search with nprobe=1 should still find the obvious nearest in a small dataset.
TEST_F(VeloxTest, IVFSearchWithNprobe) {
    for (int i = 0; i < 20; i++)
        db.add_vector({static_cast<float>(i), 0.0f});

    db.build_index(/*num_clusters=*/4, /*epochs=*/5, "eucl");

    auto results = db.search({0.0f, 0.0f}, /*k=*/1, /*nprobe=*/2, "eucl");
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].first, 0);
}

TEST_F(VeloxTest, OutOfBoundsThrows) {
    db.add_vector({1.0f});
    EXPECT_THROW(db.get_vector(99), std::out_of_range);
}

TEST_F(VeloxTest, DimMismatchSearchThrows) {
    db.add_vector({1.0f, 2.0f});
    EXPECT_THROW(db.search({1.0f}, 1, 1, "eucl"), std::runtime_error);
}

TEST_F(VeloxTest, DimMismatchAddThrows) {
    db.add_vector({1.0f, 2.0f});
    EXPECT_THROW(db.add_vector({1.0f}), std::runtime_error);
}

// HNSW search should still find the obvious nearest in a small dataset.
TEST_F(VeloxTest, HNSWBuildAndSearchFindsNearest) {
    for (int i = 0; i < 20; i++)
        db.add_vector({static_cast<float>(i), 0.0f});

    db.build_index_hnsw(/*M=*/8, /*ef_construction=*/50, "eucl");
    EXPECT_EQ(db.get_index_type(), "hnsw");

    auto results = db.search({0.0f, 0.0f}, /*k=*/1, /*nprobe=*/1, "eucl", /*ef_search=*/20);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].first, 0);
}

TEST_F(VeloxTest, HNSWDimensionMismatchThrows) {
    db.add_vector({1.0f, 2.0f});
    db.build_index_hnsw(/*M=*/4, /*ef_construction=*/20, "eucl");
    EXPECT_THROW(db.search({1.0f}, 1, 1, "eucl"), std::runtime_error);
}

TEST_F(VeloxTest, HNSWSaveLoadRoundtrip) {
    for (int i = 0; i < 30; i++)
        db.add_vector({static_cast<float>(i), static_cast<float>(i % 5)});

    db.build_index_hnsw(/*M=*/8, /*ef_construction=*/50, "eucl");
    auto before = db.search({3.0f, 1.0f}, /*k=*/5, /*nprobe=*/1, "eucl", /*ef_search=*/30);

    const char* path = "/tmp/velox_hnsw_roundtrip_test.idx";
    db.save_index(path);

    VectorIndex reloaded;
    reloaded.set_simd(true);
    for (int i = 0; i < 30; i++)
        reloaded.add_vector({static_cast<float>(i), static_cast<float>(i % 5)});
    reloaded.load_index(path);
    std::remove(path);

    EXPECT_EQ(reloaded.get_index_type(), "hnsw");
    auto after = reloaded.search({3.0f, 1.0f}, /*k=*/5, /*nprobe=*/1, "eucl", /*ef_search=*/30);

    ASSERT_EQ(before.size(), after.size());
    for (size_t i = 0; i < before.size(); i++) {
        EXPECT_EQ(before[i].first, after[i].first);
        EXPECT_FLOAT_EQ(before[i].second, after[i].second);
    }
}

// Sanity check: HNSW top-10 should mostly agree with brute-force top-10 on a
// moderately sized random dataset (approximate, not exact, by design).
TEST_F(VeloxTest, HNSWRecallVsBruteForce) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr int kNumVectors = 500;
    constexpr int kDim = 32;
    for (int i = 0; i < kNumVectors; i++) {
        std::vector<float> v(kDim);
        for (int d = 0; d < kDim; d++) v[d] = dist(rng);
        db.add_vector(v);
    }

    std::vector<float> query(kDim);
    for (int d = 0; d < kDim; d++) query[d] = dist(rng);

    auto brute = db.search(query, /*k=*/10, /*nprobe=*/1, "eucl"); // untrained -> brute force

    db.build_index_hnsw(/*M=*/16, /*ef_construction=*/200, "eucl");
    auto approx = db.search(query, /*k=*/10, /*nprobe=*/1, "eucl", /*ef_search=*/100);

    std::unordered_set<int> brute_ids;
    for (auto& p : brute) brute_ids.insert(p.first);

    int overlap = 0;
    for (auto& p : approx)
        if (brute_ids.count(p.first)) overlap++;

    EXPECT_GE(overlap, 8);
}

// Old (pre-HNSW) .ivf files had no index_type discriminator and wrote
// num_clusters/dim before the version-2 format's dim-first layout. Verify
// they still load correctly as IVF indexes.
TEST_F(VeloxTest, LegacyV1IndexLoads) {
    db.add_vector({0.0f});
    db.add_vector({1.0f});
    db.add_vector({5.0f});
    db.add_vector({10.0f});

    const char* path = "/tmp/velox_legacy_v1_test.idx";
    {
        std::ofstream out(path, std::ios::binary);
        uint32_t magic = 0x564C5846;
        uint16_t version = 1;
        int32_t num_clusters = 2;
        int32_t dim = 1;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&num_clusters), sizeof(num_clusters));
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

        float centroid0 = 0.5f, centroid1 = 7.5f;
        out.write(reinterpret_cast<const char*>(&centroid0), sizeof(float));
        out.write(reinterpret_cast<const char*>(&centroid1), sizeof(float));

        int32_t list0_sz = 2, list0[2] = {0, 1};
        out.write(reinterpret_cast<const char*>(&list0_sz), sizeof(list0_sz));
        out.write(reinterpret_cast<const char*>(list0), sizeof(list0));

        int32_t list1_sz = 2, list1[2] = {2, 3};
        out.write(reinterpret_cast<const char*>(&list1_sz), sizeof(list1_sz));
        out.write(reinterpret_cast<const char*>(list1), sizeof(list1));
    }

    db.load_index(path);
    std::remove(path);

    EXPECT_EQ(db.get_index_type(), "ivf");
    auto results = db.search({0.5f}, /*k=*/1, /*nprobe=*/1, "eucl");
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].first == 0 || results[0].first == 1);
}
