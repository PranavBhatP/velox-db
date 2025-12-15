#include "metrics.hpp"
#include <immintrin.h>
#include <cmath>

float euclidean_dist(const float *a, const float *b, int n){
    float dist = 0.0f;
    for(int i = 0; i < n; i++){
        float diff = a[i]-b[i];
        dist += diff * diff;
    }
    return dist;
}


float euclidean_dist_simd(const float *a, const float *b, int n){
    float sum = 0.0f; 

    int i = 0;

    __m256 sum_vec = _mm256_setzero_ps();

    for(; i + 8 <= n; i+=8){
        __m256 va = _mm256_loadu_ps(a+i);
        __m256 vb = _mm256_loadu_ps(b+i);

        __m256 diff = _mm256_sub_ps(va,vb);
        __m256 sq = _mm256_mul_ps(diff, diff);

        sum_vec = _mm256_add_ps(sum_vec, sq);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, sum_vec);
    for(int k = 0; k < 8; ++k) sum += buffer[k];

    for(; i < n; i++){
        float diff = a[i]-b[i];
        sum += diff*diff;
    }

    return sum;
}

float cosine_dist(const float *a, const float *b, int n){
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for(int i = 0 ; i < n; i++){
        dot += a[i]*b[i];
        norm_a+= a[i]*a[i];
        norm_b += b[i]*b[i];
    }

    if(norm_a == 0 || norm_b == 0) return 1.0f; 

    return 1-(dot/(std::sqrt(norm_a)* std::sqrt(norm_b)));
}

float cosine_dist_simd(const float* a, const float *b, int n){
    int i = 0;

    __m256 dot_vec = _mm256_setzero_ps();
    __m256 na_vec = _mm256_setzero_ps();
    __m256 nb_vec = _mm256_setzero_ps();

    for(; i+8 <= n; i+=8){ 
        __m256 va = _mm256_loadu_ps(a+i);
        __m256 vb = _mm256_loadu_ps(b+i);

        dot_vec = _mm256_add_ps(dot_vec, _mm256_mul_ps(va,vb));
        na_vec = _mm256_add_ps(na_vec, _mm256_mul_ps(va,va));
        nb_vec = _mm256_add_ps(nb_vec, _mm256_mul_ps(vb,vb));
    }

    float buf_dot[8], buf_a[8], buf_b[8];
    float sum_dot = 0.0f, sum_a = 0.0f, sum_b = 0.0f;

    _mm256_storeu_ps(buf_dot, dot_vec);
    _mm256_storeu_ps(buf_a, na_vec);   
    _mm256_storeu_ps(buf_b, nb_vec);   

    for(int k = 0; k < 8; k++){        
        sum_dot += buf_dot[k];         
        sum_a += buf_a[k];             
        sum_b += buf_b[k];    
    }
    
    for (; i < n; ++i) {
        sum_dot += a[i] * b[i];
        sum_a += a[i] * a[i];
        sum_b += b[i] * b[i];
    }

    if (sum_a == 0 || sum_b == 0) return 1.0f;

    return 1 - (sum_dot/(std::sqrt(sum_a) * std::sqrt(sum_b)));
}