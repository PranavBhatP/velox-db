#include<gtest/gtest.h>
#include<iostream>
#include<vector>
#include "vector_db.hpp"

class VeloxTest: public::testing::Test {
    protected:
        VectorIndex db;
        void SetUp() override{
            db.set_simd(true);
        }
};

TEST_F(VeloxTest, AddVectorMatchesRetrieval){
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    db.add_vector(vec);

    std::vector<float> retrieved = db.get_vector(0);
    ASSERT_EQ(retrieved.size(), 3);
    EXPECT_FLOAT_EQ(retrieved[0], 1.0f);
    EXPECT_FLOAT_EQ(retrieved[1], 2.0f);
    EXPECT_FLOAT_EQ(retrieved[2], 3.0f);

}

TEST_F(VeloxTest, SearchLogicEuclidean) {
    db.add_vector({0.0f, 0.0f});
    db.add_vector({0.0f, 1.0f});
    db.add_vector({10.0f, 10.0f});
    std::vector<float> query = {0.0f, 0.8f};

    int idx = db.search(query, "eucl");
    ASSERT_EQ(idx, 1);
}


TEST_F(VeloxTest, SearchLogicCosine) {
    db.add_vector({1.0f, 0.0f});
    db.add_vector({0.0f, 1.0f});
    db.add_vector({10.0f, 10.0f});

    std::vector<float> query = {1.0f, 0.0f};

    int idx = db.search(query, "eucl");
    EXPECT_EQ(idx, 0);
}


TEST_F(VeloxTest, OutOfBoundsThrows) {
    db.add_vector({1.0f});
    EXPECT_THROW(db.get_vector(99), std::out_of_range);
}