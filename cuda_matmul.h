//
// Created by qzero on 2025/10/20.
//

#ifndef NEEDLE_CUDA_MATMUL_H
#define NEEDLE_CUDA_MATMUL_H
#include <cstdint>
#include <chrono>
#include <vector>

static inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

typedef float scalar_t;
std::vector<double> MatmulProfile(const float *a, const float *b, float *out, uint32_t M, uint32_t N, uint32_t P, uint32_t repeat_num, uint32_t ver);

#endif //NEEDLE_CUDA_MATMUL_H