#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include "cuda_matmul.h"

static float max_rel_error(const float *ref, const float *got, size_t elems, bool show_diff_log) {
    float maxr = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        float a = ref[i], b = got[i];
        float denom = std::max(std::abs(a), 1e-8f);
        float r = std::abs(a - b) / denom;
        if (r > maxr) maxr = r;
        if (show_diff_log && r > 1e-6) {
            printf("Mismatch at %lu desired %f actual %f \n", i, ref[i], got[i]);
        }
    }
    return maxr;
}

// 简单参考实现（用于验证）
void reference_gemm(const float *A, const float *B, float *C, uint32_t M, uint32_t N, uint32_t P) {
    // std::fill(C, C + (size_t)M * P, 0.0f);
    // for (uint32_t i = 0; i < M; ++i) {
    //     for (uint32_t k = 0; k < N; ++k) {
    //         float aik = A[(size_t)i * N + k];
    //         for (uint32_t j = 0; j < P; ++j) {
    //             C[(size_t)i * P + j] += aik * B[(size_t)k * P + j];
    //         }
    //     }
    // }

    MatmulProfile(A, B, C, M, N, P, 1, 2);
}

int main(int argc, char **argv) {
    // 默认参数
    uint32_t M = 4096, N = 4096, P = 4096;
    int repeats = 10;
    int warmups = 1;
    unsigned int seed = 12345;
    int ver = 10;

    // 简单命令行解析： bench M N P [repeats]
    if (argc >= 4) {
        M = static_cast<uint32_t>(std::stoul(argv[1]));
        N = static_cast<uint32_t>(std::stoul(argv[2]));
        P = static_cast<uint32_t>(std::stoul(argv[3]));
    }
    if (argc >= 5) repeats = std::max(1, std::stoi(argv[4]));
    if (argc >= 6) ver = static_cast<unsigned int>(std::stoul(argv[5]));
    if (argc >= 7) warmups = std::max(0, std::stoi(argv[6]));
    if (argc >= 8) seed = static_cast<unsigned int>(std::stoul(argv[7]));

    size_t sizeA = (size_t)M * N;
    size_t sizeB = (size_t)N * P;
    size_t sizeC = (size_t)M * P;

    // 对齐分配（64 字节对齐）
    float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;
    if (posix_memalign((void**)&A, 64, sizeA * sizeof(float)) != 0) return 1;
    if (posix_memalign((void**)&B, 64, sizeB * sizeof(float)) != 0) return 1;
    if (posix_memalign((void**)&C, 64, sizeC * sizeof(float)) != 0) return 1;
    if (posix_memalign((void**)&C_ref, 64, sizeC * sizeof(float)) != 0) return 1;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) A[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) B[i] = dist(rng);

    // warmup
    MatmulProfile(A, B, C, M, N, P, warmups, ver);

    std::vector<double> times = MatmulProfile(A, B, C, M, N, P, repeats, ver);

    // 生成参考结果（用于校验）
    reference_gemm(A, B, C_ref, M, N, P);
    std::cout << "已计算参考结果" << std::endl;
    float err = max_rel_error(C_ref, C, sizeC, false);
    std::cout << "运行相对最大误差: " << err << "\n";
    if (err > 1e-1) {
        std::cerr << "实现可能有误！！！！！！！！！！！！！！！！！！！！！\n";
    }
    assert(err < 1e-1);

    std::sort(times.begin(), times.end());
    double best_ms = times.front();
    double median_ms = times[times.size()/2];

    // 计算 FLOPS: 一次 MxN * NxP 需要 2*M*N*P 流水线操作（乘加计 2）
    double flops = 2.0 * (double)M * (double)N * (double)P;
    double best_s = best_ms / 1e3;
    double gflops = (flops / best_s) / 1e9;

    // 简单估算内存带宽（读A, readB, writeC）
    double bytes = ((double)sizeA + (double)sizeB + (double)sizeC) * sizeof(float);
    double bw_gb_s = (bytes / best_s) / 1e9;

    std::cout << "内核版本号：" << ver << "\n";
    std::cout << "矩阵尺寸: A(" << M << "x" << N << "), B(" << N << "x" << P << ")\n";
    std::cout << "数据类型: float\n";
    std::cout << "重复次数: " << repeats << " (warmups: " << warmups << ")\n";
    std::cout << "最好耗时: " << best_ms << " ms\n";
    std::cout << "中位耗时: " << median_ms << " ms\n";
    std::cout << "性能: " << gflops << " GFLOPS\n";
    std::cout << "估算内存带宽: " << bw_gb_s << " GB/s\n";

    free(A); free(B); free(C); free(C_ref);
    return 0;
}