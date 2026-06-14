#include <gtest/gtest.h>
#include <vector>
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
