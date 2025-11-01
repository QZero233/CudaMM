//
// Created by qzero on 2025/10/20.
//

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

#include "cuda_matmul.h"

#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void MatmulKernelV1(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < P) {
        scalar_t tmp = 0;
        for (uint32_t k = 0; k < N; k++) {
            // out[i][j] = a[i][k] * b[k][j]
            tmp += a[x * N + k] * b[k * P + y];
        }

        out[x * P + y] = tmp;
    }
}

constexpr uint32_t BLOCK_SIZE = 32;
__global__ void MatmulKernelV2(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    // 相当于首先把OUT分成若干个32*32的Block，V1和V2都是如此，它们的区别在于Block内部的分配方式
    // 这里blockIdx.x * BLOCK_SIZE, blockIdx.y * BLOCK_SIZE 就是在定位Block的起始x和y
    const uint32_t x = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    const uint32_t y = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;

    /*
        还有另一种实现方式：
        uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t x = blockIdx.y * blockDim.y + threadIdx.y;
        V1直接反转x和y就行了，其余保持一致
        这种实现理论上要比上面的要好，因为不涉及到除法和求余数
     */

    if (x < M && y < P) {
        scalar_t tmp = 0;
        for (uint32_t k = 0; k < N; k++) {
            // C[x][y] += A[x][k] * B[k][y]
            tmp += a[x * N + k] * b[k * P + y];
        }
        out[x * P + y] = tmp;
    }
}

__global__ void MatmulKernelV3(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    __shared__ scalar_t As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ scalar_t Bs[BLOCK_SIZE * BLOCK_SIZE];

    const uint32_t x = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    const uint32_t y = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;

    if (x >= M || y >= P) {
        return;
    }

    scalar_t tmp = 0;
    for (int32_t k = 0; k < N; k += BLOCK_SIZE) {
        // 记 threadX, threadY 为 (x, y) 在OUT块中的位置
        // threadX = threadIdx.x / BLOCK_SIZE, threadY = threadIdx.x % BLOCK_SIZE

        // 加载 A[x][k] - A[x + BLOCK_SIZE][k + BLOCK_SIZE] 到 As
        // 加载 B[k][y] - B[k + BLOCK_SIZE][y + BLOCK_SIZE] 到 Bs

        // 计算 SUM(As[threadX][:] * Bs[:][threadY]) 存储到 OUT[x][y]

        uint32_t threadX = threadIdx.x / BLOCK_SIZE;
        uint32_t threadY = threadIdx.x % BLOCK_SIZE;

        As[threadX * BLOCK_SIZE + threadY] = a[x * N + (k + threadY)];
        Bs[threadX * BLOCK_SIZE + threadY] = b[(k + threadX) * P + y];

        __syncthreads();

        // 注意：这里在矩阵大小<32时会出问题，因为As实际上没装满
        for (int32_t i = 0; i < BLOCK_SIZE; i++) {
            tmp += As[threadX * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + threadY];
        }

        __syncthreads();
    }
    out[x * P + y] = tmp;

}

namespace V4 {
    constexpr uint32_t ROW_BLOCK_SIZE = 64;
    constexpr uint32_t COL_BLOCK_SIZE = 64;
    constexpr uint32_t K_STEP = 8;
    constexpr uint32_t TILE_ROW_SIZE = 8;

    __device__ inline uint32_t idx(uint32_t x, uint32_t y, uint32_t row_num, uint32_t col_num) {
        return x * col_num + y;
    }

    __global__ void MatmulKernelV4(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
        const uint32_t outTopLeftX = blockIdx.x * ROW_BLOCK_SIZE;
        const uint32_t outTopLeftY = blockIdx.y * COL_BLOCK_SIZE;
        // 需要注意的是，这里的每个 (threadX, threadY) 都需要计算 TILE_ROW_SIZE 个OUT
        const uint32_t threadX = threadIdx.x / COL_BLOCK_SIZE;
        const uint32_t threadY = threadIdx.x % COL_BLOCK_SIZE;

        if ((outTopLeftX + threadX) >= M || (outTopLeftY + threadY) >= P) {
            return;
        }

        // 用lambda表达式计算二维坐标对应的下标
        const auto aIdx = [M, N](uint32_t x, uint32_t y) {
            return idx(x, y, M, N);
        };
        const auto bIdx = [N, P](uint32_t x, uint32_t y) {
            return idx(x, y, N, P);
        };
        const auto outIdx = [M, P](uint32_t x, uint32_t y) {
            return idx(x, y, M, P);
        };
        const auto asIdx = [](uint32_t x, uint32_t y) {
            return idx(x, y, ROW_BLOCK_SIZE, K_STEP);
        };
        const auto bsIdx = [](uint32_t x, uint32_t y) {
            return idx(x, y, K_STEP, COL_BLOCK_SIZE);
        };

        // 每个Block计算 ROW_BLOCK_SIZE * COL_BLOCK_SIZE 个OUT
        // 其中每个线程计算 ROW_TILE_SIZE * 1 个OUT
        // 每个Block有 (ROW_BLOCK_SIZE / ROW_TILE_SIZE) * COL_BLOCK_SIZE 个线程
        // 移动步长由 BLOCK_SIZE 改为了 K_STEP
        __shared__ scalar_t As[ROW_BLOCK_SIZE * K_STEP];
        __shared__ scalar_t Bs[K_STEP * COL_BLOCK_SIZE];

        scalar_t tmp[TILE_ROW_SIZE] = {0.0f};
        for (uint32_t k = 0; k < N; k += K_STEP) {
            // 加载As和Bs

            // 这里As的大小为 ROW_BLOCK_SIZE * K_STEP，线程数量为 (ROW_BLOCK_SIZE / ROW_TILE_SIZE) * COL_BLOCK_SIZE
            // 两者应该恰好相等，否则无法把As装载完成
            // 如果相等，就能通过编排的方式完成装载
            assert((ROW_BLOCK_SIZE * K_STEP) == ((ROW_BLOCK_SIZE / TILE_ROW_SIZE) * COL_BLOCK_SIZE)); //会常量展开的，不影响性能
            // 这里计算每个线程负责装载到As的哪个位置
            const uint32_t asLoadX = threadIdx.x / K_STEP;
            const uint32_t asLoadY = threadIdx.x % K_STEP;
            // 该线程负责搬运 A[outTopLeftX + asLoadX][k + asLoadY] -> As[asLoadX][asLoadY]
            As[asIdx(asLoadX, asLoadY)] = a[aIdx(outTopLeftX + asLoadX, k + asLoadY)];

            // Bs同理
            assert((K_STEP * COL_BLOCK_SIZE) == ((ROW_BLOCK_SIZE / TILE_ROW_SIZE) * COL_BLOCK_SIZE));
            const uint32_t bsLoadX = threadIdx.x / COL_BLOCK_SIZE;
            const uint32_t bsLoadY = threadIdx.x % COL_BLOCK_SIZE;
            // 这里搬运 B[k + bsLoadX][outTopLeftY + bsLoadY] -> Bs[bsLoadX][bsLoadY]
            Bs[bsIdx(bsLoadX, bsLoadY)] = b[bIdx(k + bsLoadX, outTopLeftY + bsLoadY)];

            __syncthreads();

            // 循环计算 tmp[tile] += As[threadX * TILE_ROW_SIZE + tile][:] * Bs[:][threadY] ，其中 tile: 0 -> ROW_TILE_SIZE
            // 计算完成的 tmp[tile] 就是 OUT[x * TILE_ROW_SIZE + tile][threadY]
            // 这一步可以缓存 Bs[:][threadY] 以复用
            // 这里循环把i提到外面，一方面是为了复用B，另一方面也是为了防止Bank Conflict（一个warp里每个线程都在访问同一个x对应不同的y，并且是顺序的）
            for (uint32_t i = 0; i < K_STEP; i++) {
                const scalar_t tmpB = Bs[bsIdx(i, threadY)];
                for (uint32_t tile = 0; tile < TILE_ROW_SIZE; tile++) {
                    tmp[tile] += As[asIdx(threadX * TILE_ROW_SIZE + tile, i)] * tmpB;
                }
            }

            __syncthreads();
        }

        // 写回结果
        for (uint32_t tile = 0; tile < TILE_ROW_SIZE; tile++) {
            out[outIdx(outTopLeftX + threadX * TILE_ROW_SIZE + tile, outTopLeftY + threadY)] = tmp[tile];
        }
    }

}

namespace V5 {
    constexpr uint32_t ROW_BLOCK_SIZE = 128;
    constexpr uint32_t COL_BLOCK_SIZE = 128;
    constexpr uint32_t K_STEP = 8;
    constexpr uint32_t TILE_ROW_SIZE = 8;
    constexpr uint32_t TILE_COL_SIZE = 8;

    __device__ inline uint32_t idx(uint32_t x, uint32_t y, uint32_t row_num, uint32_t col_num) {
        return x * col_num + y;
    }

    __global__ void MatmulKernelV5(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
        // 每个线程计算OUT的 TILE_ROW_SIZE * TILE_ROW_SIZE 个值
        // 一共启动 (ROW_BLOCK_SIZE / TILE_ROW_SIZE) * (COL_BLOCK_SIZE / TILE_COL_SIZE) 个线程
        const uint32_t outTopLeftX = blockIdx.x * ROW_BLOCK_SIZE;
        const uint32_t outTopLeftY = blockIdx.y * COL_BLOCK_SIZE;
        const uint32_t threadX = threadIdx.x / (COL_BLOCK_SIZE / TILE_COL_SIZE);
        const uint32_t threadY = threadIdx.x % (COL_BLOCK_SIZE / TILE_COL_SIZE);

        if ((outTopLeftX + threadX) >= M || (outTopLeftY + threadY) >= P) {
            return;
        }

        // 这个线程需要计算的小OUT块的左上角坐标
        const uint32_t threadOutTopLeftX = outTopLeftX + threadX * TILE_ROW_SIZE;
        const uint32_t threadOutTopLeftY = outTopLeftY + threadY * TILE_COL_SIZE;

        // 用lambda表达式计算二维坐标对应的下标
        const auto aIdx = [M, N](uint32_t x, uint32_t y) {
            return idx(x, y, M, N);
        };
        const auto bIdx = [N, P](uint32_t x, uint32_t y) {
            return idx(x, y, N, P);
        };
        const auto outIdx = [M, P](uint32_t x, uint32_t y) {
            return idx(x, y, M, P);
        };
        const auto asIdx = [](uint32_t x, uint32_t y) {
            return idx(x, y, ROW_BLOCK_SIZE, K_STEP);
        };
        const auto bsIdx = [](uint32_t x, uint32_t y) {
            return idx(x, y, K_STEP, COL_BLOCK_SIZE);
        };

        // 每个Block计算 ROW_BLOCK_SIZE * COL_BLOCK_SIZE 个OUT
        // 其中每个线程计算 TILE_ROW_SIZE * TILE_COL_SIZE 个OUT
        __shared__ scalar_t As[ROW_BLOCK_SIZE * K_STEP];
        __shared__ scalar_t Bs[K_STEP * COL_BLOCK_SIZE];

        constexpr  uint32_t totalThreadNum = (ROW_BLOCK_SIZE / TILE_ROW_SIZE) * (COL_BLOCK_SIZE / TILE_COL_SIZE);
        constexpr uint32_t strideA = (ROW_BLOCK_SIZE * K_STEP) / totalThreadNum;
        constexpr uint32_t strideB = (K_STEP * COL_BLOCK_SIZE) / totalThreadNum;

        scalar_t tmp[TILE_ROW_SIZE * TILE_COL_SIZE] = {0.0f};
        scalar_t a_reg[TILE_ROW_SIZE];
        scalar_t b_reg[TILE_COL_SIZE];

        for (uint32_t k = 0; k < N; k += K_STEP) {
            // 当前线程数量小于SMEM Size,需要用一层循环来加载
            // 这里要小心Bank Conflict
            // 但是本身目标内存区域As和Bs也是不连续的，怎么防止Bank Conflict还需要仔细研究一下
            for (uint32_t i = 0; i < strideA; i++) {
                const uint32_t loadIndex = i * totalThreadNum + threadIdx.x;
                const uint32_t loadX = loadIndex / K_STEP;
                const uint32_t loadY = loadIndex % K_STEP;
                // 加载 A[outTopLeftX + loadX][k + loadY] -> As[loadIndex]
                As[loadIndex] = a[aIdx(outTopLeftX + loadX, k + loadY)];
            }

            for (uint32_t i = 0; i < strideB; i++) {
                const uint32_t loadIndex = i * totalThreadNum + threadIdx.x;
                const uint32_t loadX = loadIndex / COL_BLOCK_SIZE;
                const uint32_t loadY = loadIndex % COL_BLOCK_SIZE;
                // 加载 B[k + loadX][outTopLeftY + loadY] -> Bs[loadIndex]
                Bs[loadIndex] = b[bIdx(k + loadX, outTopLeftY + loadY)];
            }

            __syncthreads();


            // 计算OUT
            for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row++) {
                for (uint32_t tile_col = 0; tile_col < TILE_COL_SIZE; tile_col++) {
                    for (uint32_t i = 0; i < K_STEP; i++) {
                        tmp[tile_row * TILE_COL_SIZE + tile_col] += As[asIdx(threadX * TILE_ROW_SIZE + tile_row, i)] * Bs[bsIdx(i, threadY * TILE_COL_SIZE + tile_col)];
                    }
                }
            }

            // 这里经过优化，把SMEM访问次数从 TILE_ROW_SIZE * TILE_COL_SIZE * K_STEP
            // 变为了 K_STEP * (TILE_ROW_SIZE + TILE_COL_SIZE)
            // 能这么做的本质原因在于，计算OUT中的元素时，是存在复用的，比如(0, 0)和(0, 1)会共用As的第0行
            // 这里巧妙构造之后将这个复用实现了
            // 但是我这边实测下来，下面这个实现比上面的要慢接近50%
            // for (uint32_t i = 0; i < K_STEP; i++) {
            //     for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row++) {
            //         a_reg[tile_row] = As[asIdx(threadX * TILE_ROW_SIZE + tile_row, i)];
            //     }
            //
            //     for (uint32_t tile_col = 0; tile_col < TILE_COL_SIZE; tile_col++) {
            //         b_reg[tile_col] = Bs[bsIdx(i, threadY * TILE_COL_SIZE + tile_col)];
            //     }
            //
            //     for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row++) {
            //         for (uint32_t tile_col = 0; tile_col < TILE_ROW_SIZE; tile_col++) {
            //             tmp[tile_row * TILE_COL_SIZE + tile_col] += a_reg[tile_row] * b_reg[tile_col];
            //         }
            //     }
            // }


            __syncthreads();
        }

        // 写回结果
        for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row++) {
            for (uint32_t tile_col = 0; tile_col < TILE_COL_SIZE; tile_col++) {
                out[outIdx(threadOutTopLeftX + tile_row, threadOutTopLeftY + tile_col)] = tmp[tile_row * TILE_COL_SIZE + tile_col];
            }
        }
    }
}

namespace V6 {
    constexpr uint32_t ROW_BLOCK_SIZE = 128;
    constexpr uint32_t COL_BLOCK_SIZE = 128;
    constexpr uint32_t K_STEP = 8;
    constexpr uint32_t TILE_ROW_SIZE = 8;
    constexpr uint32_t TILE_COL_SIZE = 8;

    __device__ inline uint32_t idx(uint32_t x, uint32_t y, uint32_t row_num, uint32_t col_num) {
        return x * col_num + y;
    }

    __global__ void MatmulKernelV6(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
        // 每个线程计算OUT的 TILE_ROW_SIZE * TILE_ROW_SIZE 个值
        // 一共启动 (ROW_BLOCK_SIZE / TILE_ROW_SIZE) * (COL_BLOCK_SIZE / TILE_COL_SIZE) 个线程
        const uint32_t outTopLeftX = blockIdx.x * ROW_BLOCK_SIZE;
        const uint32_t outTopLeftY = blockIdx.y * COL_BLOCK_SIZE;
        const uint32_t threadX = threadIdx.x / (COL_BLOCK_SIZE / TILE_COL_SIZE);
        const uint32_t threadY = threadIdx.x % (COL_BLOCK_SIZE / TILE_COL_SIZE);

        if ((outTopLeftX + threadX) >= M || (outTopLeftY + threadY) >= P) {
            return;
        }

        // 这个线程需要计算的小OUT块的左上角坐标
        const uint32_t threadOutTopLeftX = outTopLeftX + threadX * TILE_ROW_SIZE;
        const uint32_t threadOutTopLeftY = outTopLeftY + threadY * TILE_COL_SIZE;

        // 用lambda表达式计算二维坐标对应的下标
        const auto aIdx = [M, N](uint32_t x, uint32_t y) {
            return idx(x, y, M, N);
        };
        const auto bIdx = [N, P](uint32_t x, uint32_t y) {
            return idx(x, y, N, P);
        };
        const auto outIdx = [M, P](uint32_t x, uint32_t y) {
            return idx(x, y, M, P);
        };
        const auto asIdx = [](uint32_t x, uint32_t y) {
            return idx(x, y, ROW_BLOCK_SIZE, K_STEP);
        };
        const auto bsIdx = [](uint32_t x, uint32_t y) {
            return idx(x, y, K_STEP, COL_BLOCK_SIZE);
        };

        // 每个Block计算 ROW_BLOCK_SIZE * COL_BLOCK_SIZE 个OUT
        // 其中每个线程计算 TILE_ROW_SIZE * TILE_COL_SIZE 个OUT
        __shared__ scalar_t As[ROW_BLOCK_SIZE * K_STEP];
        __shared__ scalar_t Bs[K_STEP * COL_BLOCK_SIZE];

        constexpr uint32_t totalThreadNum = (ROW_BLOCK_SIZE / TILE_ROW_SIZE) * (COL_BLOCK_SIZE / TILE_COL_SIZE);
        constexpr uint32_t strideA = (ROW_BLOCK_SIZE * K_STEP) / totalThreadNum;
        constexpr uint32_t strideB = (K_STEP * COL_BLOCK_SIZE) / totalThreadNum;

        scalar_t tmp[TILE_ROW_SIZE * TILE_COL_SIZE] = {0.0f};
        scalar_t a_reg[TILE_ROW_SIZE];
        scalar_t b_reg[TILE_COL_SIZE];

        for (uint32_t k = 0; k < N; k += K_STEP) {
            // 使用float4向量化加载必须确保每个线程处理4个元素的加载
            assert(strideA == 4);
            assert(strideB == 4);

            // 不对As做转置的版本
            // 要求K_STEP必须是4对齐的
            assert((K_STEP & 3) == 0);
            // 计算该线程要加载的元素在As的起始位置
            const uint32_t as_load_x = threadIdx.x / (K_STEP / 4);
            const uint32_t as_load_y = (threadIdx.x % (K_STEP / 4)) * 4;
            // 计算A要加载的部分在整个A数组的坐标
            const uint32_t a_load_x = outTopLeftX + as_load_x;
            const uint32_t a_load_y = as_load_y + k;
            reinterpret_cast<float4 *>(&As[asIdx(as_load_x, as_load_y)])[0] =
                reinterpret_cast<const float4 *>(&a[aIdx(a_load_x, a_load_y)])[0];

            // 加载Bs，同理
            // 要求COL_BLOCK_SIZE必须是4对齐的，这样才能一个线程加载一行不出错
            assert((COL_BLOCK_SIZE & 3) == 0);
            const uint32_t bs_load_x = threadIdx.x / (COL_BLOCK_SIZE / 4);
            const uint32_t bs_load_y = (threadIdx.x % (COL_BLOCK_SIZE / 4)) * 4;
            const uint32_t b_load_x = k + bs_load_x;
            const uint32_t b_load_y = outTopLeftY + bs_load_y;
            reinterpret_cast<float4 *>(&Bs[bsIdx(bs_load_x, bs_load_y)])[0] =
                reinterpret_cast<const float4 *>(&b[bIdx(b_load_x, b_load_y)])[0];

            __syncthreads();

            // 计算OUT
            for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row++) {
                for (uint32_t tile_col = 0; tile_col < TILE_COL_SIZE; tile_col++) {
                    for (uint32_t i = 0; i < K_STEP; i++) {
                        tmp[tile_row * TILE_COL_SIZE + tile_col] += As[asIdx(threadX * TILE_ROW_SIZE + tile_row, i)] * Bs[bsIdx(i, threadY * TILE_COL_SIZE + tile_col)];
                    }
                }
            }

            __syncthreads();
        }

        // 写回结果
        // 这里同样可以向量化
        assert((TILE_COL_SIZE & 3) == 0);
        for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row++) {
            for (uint32_t tile_col = 0; tile_col < TILE_COL_SIZE; tile_col += 4) {
                reinterpret_cast<float4 *>(&out[outIdx(threadOutTopLeftX + tile_row, threadOutTopLeftY + tile_col)])[0] =
                    reinterpret_cast<float4 *>(&tmp[tile_row * TILE_COL_SIZE + tile_col])[0];
            }
        }
    }

    __global__ void MatmulKernelV6Transpose(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
        // 每个线程计算OUT的 TILE_ROW_SIZE * TILE_ROW_SIZE 个值
        // 一共启动 (ROW_BLOCK_SIZE / TILE_ROW_SIZE) * (COL_BLOCK_SIZE / TILE_COL_SIZE) 个线程
        const uint32_t outTopLeftX = blockIdx.x * ROW_BLOCK_SIZE;
        const uint32_t outTopLeftY = blockIdx.y * COL_BLOCK_SIZE;
        const uint32_t threadX = threadIdx.x / (COL_BLOCK_SIZE / TILE_COL_SIZE);
        const uint32_t threadY = threadIdx.x % (COL_BLOCK_SIZE / TILE_COL_SIZE);

        if ((outTopLeftX + threadX) >= M || (outTopLeftY + threadY) >= P) {
            return;
        }

        // 这个线程需要计算的小OUT块的左上角坐标
        const uint32_t threadOutTopLeftX = outTopLeftX + threadX * TILE_ROW_SIZE;
        const uint32_t threadOutTopLeftY = outTopLeftY + threadY * TILE_COL_SIZE;

        // 用lambda表达式计算二维坐标对应的下标
        const auto aIdx = [M, N](uint32_t x, uint32_t y) {
            return idx(x, y, M, N);
        };
        const auto bIdx = [N, P](uint32_t x, uint32_t y) {
            return idx(x, y, N, P);
        };
        const auto outIdx = [M, P](uint32_t x, uint32_t y) {
            return idx(x, y, M, P);
        };
        const auto asIdxTranspose = [](uint32_t x, uint32_t y) {
            return idx(x, y, K_STEP, ROW_BLOCK_SIZE);
        };
        const auto bsIdx = [](uint32_t x, uint32_t y) {
            return idx(x, y, K_STEP, COL_BLOCK_SIZE);
        };

        // 每个Block计算 ROW_BLOCK_SIZE * COL_BLOCK_SIZE 个OUT
        // 其中每个线程计算 TILE_ROW_SIZE * TILE_COL_SIZE 个OUT
        __shared__ scalar_t As[ROW_BLOCK_SIZE * K_STEP];
        __shared__ scalar_t Bs[K_STEP * COL_BLOCK_SIZE];

        constexpr uint32_t totalThreadNum = (ROW_BLOCK_SIZE / TILE_ROW_SIZE) * (COL_BLOCK_SIZE / TILE_COL_SIZE);
        constexpr uint32_t strideA = (ROW_BLOCK_SIZE * K_STEP) / totalThreadNum;
        constexpr uint32_t strideB = (K_STEP * COL_BLOCK_SIZE) / totalThreadNum;

        scalar_t tmp[TILE_ROW_SIZE * TILE_COL_SIZE] = {0.0f};
        scalar_t a_reg[TILE_ROW_SIZE];
        scalar_t b_reg[TILE_COL_SIZE];

        for (uint32_t k = 0; k < N; k += K_STEP) {
            // 使用float4向量化加载必须确保每个线程处理4个元素的加载
            assert(strideA == 4);
            assert(strideB == 4);

            // 要求K_STEP必须是4对齐的
            assert((K_STEP & 3) == 0);
            // 计算该线程要加载的元素在As的起始位置
            const uint32_t as_load_x = threadIdx.x / (K_STEP / 4);
            const uint32_t as_load_y = (threadIdx.x % (K_STEP / 4)) * 4;
            // 计算A要加载的部分在整个A数组的坐标
            const uint32_t a_load_x = outTopLeftX + as_load_x;
            const uint32_t a_load_y = as_load_y + k;
            // 转置加载到As
            auto *a_tmp = reinterpret_cast<const float4 *>(&a[aIdx(a_load_x, a_load_y)]);
            As[asIdxTranspose(as_load_y + 0, as_load_x)] = a_tmp->x;
            As[asIdxTranspose(as_load_y + 1, as_load_x)] = a_tmp->y;
            As[asIdxTranspose(as_load_y + 2, as_load_x)] = a_tmp->z;
            As[asIdxTranspose(as_load_y + 3, as_load_x)] = a_tmp->w;

            // 加载Bs，同理
            // 要求COL_BLOCK_SIZE必须是4对齐的，这样才能一个线程加载一行不出错
            assert((COL_BLOCK_SIZE & 3) == 0);
            const uint32_t bs_load_x = threadIdx.x / (COL_BLOCK_SIZE / 4);
            const uint32_t bs_load_y = (threadIdx.x % (COL_BLOCK_SIZE / 4)) * 4;
            const uint32_t b_load_x = k + bs_load_x;
            const uint32_t b_load_y = outTopLeftY + bs_load_y;
            reinterpret_cast<float4 *>(&Bs[bsIdx(bs_load_x, bs_load_y)])[0] =
                reinterpret_cast<const float4 *>(&b[bIdx(b_load_x, b_load_y)])[0];

            __syncthreads();

            // 使用V5里提到的优化方法，提前把数据加载到reg里，减少SMEM的访问
            // 这里As转置后，可以向量化地提取，可以进一步提升性能
            // 但是这个实现还是比原版V6慢，不理解为什么
            for (uint32_t i = 0; i < K_STEP; i++) {
                // 这里会向量化加载a_reg，所以要求其长度4对齐
                assert((TILE_ROW_SIZE & 3) == 0);
                for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row += 4) {
                    reinterpret_cast<float4 *>(&a_reg[tile_row])[0] =
                        reinterpret_cast<float4 *>(&As[asIdxTranspose(i, threadX * TILE_ROW_SIZE + tile_row)])[0];
                }

                // 同理，向量化加载b_reg
                assert((TILE_COL_SIZE & 3) == 0);
                for (uint32_t tile_col = 0; tile_col < TILE_COL_SIZE; tile_col += 4) {
                    reinterpret_cast<float4 *>(&b_reg[tile_col])[0] =
                        reinterpret_cast<float4 *>(&Bs[bsIdx(i, threadY * TILE_COL_SIZE + tile_col)])[0];
                }

                for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row++) {
                    for (uint32_t tile_col = 0; tile_col < TILE_ROW_SIZE; tile_col++) {
                        tmp[tile_row * TILE_COL_SIZE + tile_col] += a_reg[tile_row] * b_reg[tile_col];
                    }
                }
            }


            __syncthreads();
        }

        // 写回结果
        // 这里同样可以向量化
        assert((TILE_COL_SIZE & 3) == 0);
        for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row++) {
            for (uint32_t tile_col = 0; tile_col < TILE_COL_SIZE; tile_col += 4) {
                reinterpret_cast<float4 *>(&out[outIdx(threadOutTopLeftX + tile_row, threadOutTopLeftY + tile_col)])[0] =
                    reinterpret_cast<float4 *>(&tmp[tile_row * TILE_COL_SIZE + tile_col])[0];
            }
        }
    }

}

namespace V6_Author {
    __global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                                   float *B, float beta, float *C) {
      const int BM = 128;
      const int BN = 128;
      const int BK = 8;
      const int TM = 8;
      const int TN = 8;

      const uint cRow = blockIdx.y;
      const uint cCol = blockIdx.x;

      const uint totalResultsBlocktile = BM * BN;
      // A thread is responsible for calculating TM*TN elements in the blocktile
      const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

      // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
      assert(numThreadsBlocktile == blockDim.x);

      // BN/TN are the number of threads to span a column
      const int threadCol = threadIdx.x % (BN / TN);
      const int threadRow = threadIdx.x / (BN / TN);

      // allocate space for the current blocktile in smem
      __shared__ float As[BM * BK];
      __shared__ float Bs[BK * BN];

      // Move blocktile to beginning of A's row and B's column
      A += cRow * BM * K;
      B += cCol * BN;
      C += cRow * BM * N + cCol * BN;

      // calculating the indices that this thread will load into SMEM
      // we'll load 128bit / 32bit = 4 elements per thread at each step
      const uint innerRowA = threadIdx.x / (BK / 4);
      const uint innerColA = threadIdx.x % (BK / 4);
      // calculates the number of rows of As that are being loaded in a single step
      // by a single block
      const uint rowStrideA = (numThreadsBlocktile * 4) / BK;
      const uint innerRowB = threadIdx.x / (BN / 4);
      const uint innerColB = threadIdx.x % (BN / 4);
      // for both As and Bs we want each load to span the full column-width, for
      // better GMEM coalescing (as opposed to spanning full row-width and iterating
      // across columns)
      const uint rowStrideB = numThreadsBlocktile / (BN / 4);

      // allocate thread-local cache for results in registerfile
      float threadResults[TM * TN] = {0.0};
      float regM[TM] = {0.0};
      float regN[TN] = {0.0};

      // outer-most loop over block tiles
      for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        // transpose A while loading it
        float4 tmp =
            reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        /*
         * Writing it as an unrolled loop has worse performance, because it doesn't
         * guarantee alignment of the loads
         * Bs[innerRowB * BN + innerColB * 4 + 0] =
         *    B[innerRowB * N + innerColB * 4 + 0];
         * Bs[innerRowB * BN + innerColB * 4 + 1] =
         *    B[innerRowB * N + innerColB * 4 + 1];
         * Bs[innerRowB * BN + innerColB * 4 + 2] =
         *    B[innerRowB * N + innerColB * 4 + 2];
         * Bs[innerRowB * BN + innerColB * 4 + 3] =
         *    B[innerRowB * N + innerColB * 4 + 3];
         */

        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // block into registers
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
          }
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[resIdxM * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
          }
        }
        __syncthreads();
      }

      // write out the results
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
          // perform GEMM update in reg
          tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
          tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
              tmp;
        }
      }
    }
}

namespace V10 {
    // 每个Block计算 ROW_BLOCK_SIZE * COL_BLOCK_SIZE 个OUT
    constexpr uint32_t ROW_BLOCK_SIZE = 64;
    constexpr uint32_t COL_BLOCK_SIZE = 128;
    constexpr uint32_t K_STEP = 8;
    // 每个Warp计算 WARP_ROW_SIZE * WARP_COL_SIZE 个OUT
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t WARP_ROW_SIZE = 32;
    constexpr uint32_t WARP_COL_SIZE = 64;
    // 所以一共需要 WARP_NUM = (ROW_BLOCK_SIZE * COL_BLOCK_SIZE) / (WARP_ROW_SIZE * WARP_COL_SIZE)
    // 也就是需要启动 WARP_NUM * WARP_SIZE 个线程
    constexpr uint32_t THREAD_NUM_PER_BLOCK = ((ROW_BLOCK_SIZE * COL_BLOCK_SIZE) / (WARP_ROW_SIZE * WARP_COL_SIZE)) * WARP_SIZE;

    // 一个Warp里有 THREAD_ROW_SIZE_IN_WARP * THREAD_COL_SIZE_IN_WARP 个Thread
    constexpr uint32_t THREAD_ROW_SIZE_IN_WARP = 4;
    constexpr uint32_t THREAD_COL_SIZE_IN_WARP = 8;
    // 所以一个Thread需要计算 (WARP_ROW_SIZE / THREAD_ROW_SIZE_IN_WARP) * (WARP_COL_SIZE / THREAD_COL_SIZE_IN_WARP) 个OUT
    constexpr uint32_t TILE_ROW_SIZE = WARP_ROW_SIZE / THREAD_ROW_SIZE_IN_WARP;
    constexpr uint32_t TILE_COL_SIZE = WARP_COL_SIZE / THREAD_COL_SIZE_IN_WARP;


    __device__ inline uint32_t idx(uint32_t x, uint32_t y, uint32_t row_num, uint32_t col_num) {
        return x * col_num + y;
    }

    __global__ void MatmulKernelV10(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
        // 每个线程计算OUT的 TILE_ROW_SIZE * TILE_ROW_SIZE 个值
        const uint32_t outTopLeftX = blockIdx.x * ROW_BLOCK_SIZE;
        const uint32_t outTopLeftY = blockIdx.y * COL_BLOCK_SIZE;

        // FIXME 这里判断条件要加上
        // const uint32_t threadX = threadIdx.x / (COL_BLOCK_SIZE / TILE_COL_SIZE);
        // const uint32_t threadY = threadIdx.x % (COL_BLOCK_SIZE / TILE_COL_SIZE);
        //
        // if ((outTopLeftX + threadX) >= M || (outTopLeftY + threadY) >= P) {
        //     return;
        // }

        // 用lambda表达式计算二维坐标对应的下标
        const auto aIdx = [M, N](uint32_t x, uint32_t y) {
            return idx(x, y, M, N);
        };
        const auto bIdx = [N, P](uint32_t x, uint32_t y) {
            return idx(x, y, N, P);
        };
        const auto outIdx = [M, P](uint32_t x, uint32_t y) {
            return idx(x, y, M, P);
        };
        const auto asIdxTranspose = [](uint32_t x, uint32_t y) {
            return idx(x, y, K_STEP, ROW_BLOCK_SIZE);
        };
        const auto bsIdx = [](uint32_t x, uint32_t y) {
            return idx(x, y, K_STEP, COL_BLOCK_SIZE);
        };

        // 每个Block计算 ROW_BLOCK_SIZE * COL_BLOCK_SIZE 个OUT
        // 其中每个线程计算 TILE_ROW_SIZE * TILE_COL_SIZE 个OUT
        __shared__ scalar_t As[ROW_BLOCK_SIZE * K_STEP];
        __shared__ scalar_t Bs[K_STEP * COL_BLOCK_SIZE];

        constexpr uint32_t strideA = (ROW_BLOCK_SIZE * K_STEP) / THREAD_NUM_PER_BLOCK;
        constexpr uint32_t strideB = (K_STEP * COL_BLOCK_SIZE) / THREAD_NUM_PER_BLOCK;

        scalar_t tmp[TILE_ROW_SIZE * TILE_COL_SIZE] = {0.0f};
        scalar_t a_reg[TILE_ROW_SIZE];
        scalar_t b_reg[TILE_COL_SIZE];

        constexpr uint32_t WARP_COL_SIZE_IN_BLOCK = COL_BLOCK_SIZE / WARP_COL_SIZE;
        const uint32_t warp_index_in_block = threadIdx.x / WARP_SIZE;
        const uint32_t warp_block_x = warp_index_in_block / WARP_COL_SIZE_IN_BLOCK;
        const uint32_t warp_block_y = warp_index_in_block % WARP_COL_SIZE_IN_BLOCK;

        const uint32_t thread_index_in_warp = threadIdx.x % WARP_SIZE;
        const uint32_t thread_x_in_warp = thread_index_in_warp / THREAD_COL_SIZE_IN_WARP;
        const uint32_t thread_y_in_warp = thread_index_in_warp % THREAD_COL_SIZE_IN_WARP;

        const uint32_t thread_x_in_block = warp_block_x * THREAD_ROW_SIZE_IN_WARP + thread_x_in_warp;
        const uint32_t thread_y_in_block = warp_block_y * THREAD_COL_SIZE_IN_WARP + thread_y_in_warp;

        for (uint32_t k = 0; k < N; k += K_STEP) {
            // 使用float4向量化加载必须确保每个线程处理的元素个数是4对齐的
            // 这里会直接在编译阶段展开，所以放for循环里问题不大
            static_assert((strideA & 3) == 0);
            static_assert((strideB & 3) == 0);

            // As里，从 as_start_load_index 开始，向后加载 strideA 个元素
            const uint32_t as_start_load_index = threadIdx.x * strideA;
            for (uint32_t i = 0; i < strideA; i += 4) {
                // 这个循环里，从 as_current_start_load_index 向后加载4个元素
                const uint32_t as_current_start_load_index = as_start_load_index + i;

                // 首先计算当前元素在As里的坐标
                // 这里As的形状是 ROW_BLOCK_SIZE * K_STEP
                // 需要保证K_STEP是4对齐的
                static_assert((K_STEP & 3) == 0);
                const uint32_t as_load_x = as_current_start_load_index / K_STEP;
                const uint32_t as_load_y = as_current_start_load_index % K_STEP;

                // 把这个坐标映射到A上
                const uint32_t a_x = outTopLeftX + as_load_x;
                const uint32_t a_y = k + as_load_y;

                // 使用向量化复制
                // 并且要转置
                auto a_tmp = reinterpret_cast<const float4 *>(&a[aIdx(a_x, a_y)]);
                As[asIdxTranspose(as_load_y + 0, as_load_x)] = a_tmp->x;
                As[asIdxTranspose(as_load_y + 1, as_load_x)] = a_tmp->y;
                As[asIdxTranspose(as_load_y + 2, as_load_x)] = a_tmp->z;
                As[asIdxTranspose(as_load_y + 3, as_load_x)] = a_tmp->w;
            }

            // 加载Bs，同理
            const uint32_t bs_start_load_index = threadIdx.x * strideB;
            for (uint32_t i = 0; i < strideB; i += 4) {
                const uint32_t bs_current_start_load_index = bs_start_load_index + i;

                static_assert((COL_BLOCK_SIZE & 3) == 0);
                const uint32_t bs_load_x = bs_current_start_load_index / COL_BLOCK_SIZE;
                const uint32_t bs_load_y = bs_current_start_load_index % COL_BLOCK_SIZE;

                const uint32_t b_x = bs_load_x + k;
                const uint32_t b_y = outTopLeftY + bs_load_y;

                reinterpret_cast<float4 *>(&Bs[bsIdx(bs_load_x, bs_load_y)])[0] =
                    reinterpret_cast<const float4 *>(&b[bIdx(b_x, b_y)])[0];
            }

            __syncthreads();

            for (uint32_t i = 0; i < K_STEP; i++) {
                // 这里会向量化加载a_reg，所以要求其长度4对齐
                static_assert((TILE_ROW_SIZE & 3) == 0);
                for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row += 4) {
                    reinterpret_cast<float4 *>(&a_reg[tile_row])[0] =
                        reinterpret_cast<float4 *>(&As[asIdxTranspose(i, thread_x_in_block * TILE_ROW_SIZE + tile_row)])[0];
                }

                // 同理，向量化加载b_reg
                static_assert((TILE_COL_SIZE & 3) == 0);
                for (uint32_t tile_col = 0; tile_col < TILE_COL_SIZE; tile_col += 4) {
                    reinterpret_cast<float4 *>(&b_reg[tile_col])[0] =
                        reinterpret_cast<float4 *>(&Bs[bsIdx(i, thread_y_in_block * TILE_COL_SIZE + tile_col)])[0];
                }

                for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row++) {
                    for (uint32_t tile_col = 0; tile_col < TILE_ROW_SIZE; tile_col++) {
                        tmp[tile_row * TILE_COL_SIZE + tile_col] += a_reg[tile_row] * b_reg[tile_col];
                    }
                }
            }

            __syncthreads();
        }

        // 写回结果
        // 这个线程需要计算的小OUT块的左上角坐标
        const uint32_t thread_out_top_left_x = outTopLeftX + thread_x_in_block * TILE_ROW_SIZE;
        const uint32_t thread_out_top_left_y = outTopLeftY + thread_y_in_block * TILE_COL_SIZE;

        static_assert((TILE_COL_SIZE & 3) == 0);
        for (uint32_t tile_row = 0; tile_row < TILE_ROW_SIZE; tile_row++) {
            for (uint32_t tile_col = 0; tile_col < TILE_COL_SIZE; tile_col += 4) {
                reinterpret_cast<float4 *>(&out[outIdx(thread_out_top_left_x + tile_row, thread_out_top_left_y + tile_col)])[0] =
                    reinterpret_cast<float4 *>(&tmp[tile_row * TILE_COL_SIZE + tile_col])[0];
            }
        }
    }

}

// 文章作者实现的V10版本
namespace V10_Author {
    const int WARPSIZE = 32; // warpSize is not constexpr

    /*
     * @tparam BM The threadblock size for M dimension SMEM caching.
     * @tparam BN The threadblock size for N dimension SMEM caching.
     * @tparam BK The threadblock size for K dimension SMEM caching.
     * @tparam WM M dim of continuous tile computed by each warp
     * @tparam WN N dim of continuous tile computed by each warp
     * @tparam WMITER The number of subwarp tiling steps in M dimension.
     * @tparam WNITER The number of subwarp tiling steps in N dimension.
     * @tparam TM The per-thread tile size for M dimension.
     * @tparam TN The per-thread tile size for N dimension.
     */

    const uint NUM_THREADS = 128;

    __global__ void __launch_bounds__(NUM_THREADS)
        sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
      const uint BN = 128;
      const uint BM = 64;
      const uint BK = 8;
      const uint WN = 64;
      const uint WM = 32;
      const uint WNITER = 2;
      const uint TN = 4;
      const uint TM = 4;

      const uint cRow = blockIdx.y;
      const uint cCol = blockIdx.x;

      // Placement of the warp in the threadblock tile
      const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
      const uint warpCol = warpIdx % (BN / WN);
      const uint warpRow = warpIdx / (BN / WN);

      // size of the warp subtile
      constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
      constexpr uint WSUBM = WM / WMITER; // 64/2=32
      constexpr uint WSUBN = WN / WNITER; // 32/2=16

      // Placement of the thread in the warp subtile
      const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
      const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
      const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

      // allocate space for the current blocktile in SMEM
      __shared__ float As[BM * BK];
      __shared__ float Bs[BK * BN];

      // Move blocktile to beginning of A's row and B's column
      A += cRow * BM * K;
      B += cCol * BN;
      // Move C_ptr to warp's output tile
      C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

      // calculating the indices that this thread will load into SMEM
      // we'll load 128bit / 32bit = 4 elements per thread at each step
      const uint innerRowA = threadIdx.x / (BK / 4);
      const uint innerColA = threadIdx.x % (BK / 4);
      constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
      const uint innerRowB = threadIdx.x / (BN / 4);
      const uint innerColB = threadIdx.x % (BN / 4);
      constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

      // allocate thread-local cache for results in registerfile
      float threadResults[WMITER * TM * WNITER * TN] = {0.0};
      // we cache into registers on the warptile level
      float regM[WMITER * TM] = {0.0};
      float regN[WNITER * TN] = {0.0};

      // outer-most loop over block tiles
      for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
          float4 tmp = reinterpret_cast<float4 *>(
              &A[(innerRowA + offset) * K + innerColA * 4])[0];
          // transpose A while storing it
          As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
          As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
          As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
          As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
          reinterpret_cast<float4 *>(
              &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
              reinterpret_cast<float4 *>(
                  &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // populate registers for whole warptile
          for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint i = 0; i < TM; ++i) {
              regM[wSubRowIdx * TM + i] =
                  As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                     threadRowInWarp * TM + i];
            }
          }
          for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint i = 0; i < TN; ++i) {
              regN[wSubColIdx * TN + i] =
                  Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                     threadColInWarp * TN + i];
            }
          }

          // execute warptile matmul
          for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
              // calculate per-thread results
              for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                  threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                (wSubColIdx * TN) + resIdxN] +=
                      regM[wSubRowIdx * TM + resIdxM] *
                      regN[wSubColIdx * TN + resIdxN];
                }
              }
            }
          }
        }
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down
        __syncthreads();
      }

      // write out the results
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
          // move C pointer to current warp subtile
          float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
          for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
            for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
              // load C vector into registers
              float4 tmp = reinterpret_cast<float4 *>(
                  &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                             threadColInWarp * TN + resIdxN])[0];
              // perform GEMM update in reg
              const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            wSubColIdx * TN + resIdxN;
              tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
              tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
              tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
              tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
              // write back
              reinterpret_cast<float4 *>(
                  &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                             threadColInWarp * TN + resIdxN])[0] = tmp;
            }
          }
        }
      }
    }
}

namespace cuBlasImpl {

    #include <type_traits>

    // 将 cuBLAS 的列优先输出转成行优先
    __global__ void TransposeFromCublasKernel(const scalar_t* __restrict__ cublas_out,
                                              scalar_t* __restrict__ out,
                                              uint32_t M, uint32_t P) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; // row in [0..M)
        uint32_t j = blockIdx.y * blockDim.y + threadIdx.y; // col in [0..P)
        if (i < M && j < P) {
            // cublas_out 存储的是 P x M 矩阵的列优先数据，目标是 M x P 行优先
            out[i * P + j] = cublas_out[j + i * P];
        }
    }


    void MatmulCoreV42(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cublasCreate failed: %d\n", status);
            return;
        }

        // 临时缓冲，cuBLAS 会把结果按 (m=P, n=M) 的列优先格式写入此缓冲区
        scalar_t *cublas_out = nullptr;
        cudaError_t cerr = cudaMalloc(&cublas_out, sizeof(scalar_t) * static_cast<size_t>(M) * static_cast<size_t>(P));
        if (cerr != cudaSuccess) {
            printf("cudaMalloc cublas_out failed: %d %s\n", cerr, cudaGetErrorString(cerr));
            cublasDestroy(handle);
            return;
        }

        const scalar_t alpha = static_cast<scalar_t>(1.0);
        const scalar_t beta  = static_cast<scalar_t>(0.0);

        if (std::is_same<scalar_t, float>::value) {
            status = cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 static_cast<int>(P), static_cast<int>(M), static_cast<int>(N),
                                 reinterpret_cast<const float*>(&alpha),
                                 reinterpret_cast<const float*>(b), static_cast<int>(P),
                                 reinterpret_cast<const float*>(a), static_cast<int>(N),
                                 reinterpret_cast<const float*>(&beta),
                                 reinterpret_cast<float*>(cublas_out), static_cast<int>(P));
        } else if (std::is_same<scalar_t, double>::value) {
            status = cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 static_cast<int>(P), static_cast<int>(M), static_cast<int>(N),
                                 reinterpret_cast<const double*>(&alpha),
                                 reinterpret_cast<const double*>(b), static_cast<int>(P),
                                 reinterpret_cast<const double*>(a), static_cast<int>(N),
                                 reinterpret_cast<const double*>(&beta),
                                 reinterpret_cast<double*>(cublas_out), static_cast<int>(P));
        } else {
            printf("MatmulCoreV42: unsupported scalar_t\n");
            cudaFree(cublas_out);
            cublasDestroy(handle);
            return;
        }

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cublasGemm failed: %d\n", status);
            cudaFree(cublas_out);
            cublasDestroy(handle);
            return;
        }

        // 把 cublas_out (P x M, 列优先存储) 转成 out (M x P, 行优先)
        // 确认结果正确之后，实际测实现性能时需要注释掉这一块
        // constexpr int TX = 16;
        // constexpr int TY = 16;
        // dim3 block(TX, TY);
        // dim3 grid((M + TX - 1) / TX, (P + TY - 1) / TY);
        // TransposeFromCublasKernel<<<grid, block>>>(cublas_out, out, M, P);
        //
        // cudaError_t syncErr = cudaDeviceSynchronize();
        // if (syncErr != cudaSuccess) {
        //     printf("Transpose kernel failed: %d %s\n", syncErr, cudaGetErrorString(syncErr));
        // }

        cudaFree(cublas_out);
        cublasDestroy(handle);
    }
}

// Core的指针都是Device指针
void MatmulCoreV1(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    dim3 grid(std::ceil(M / 32.0), std::ceil(P / 32.0), 1);
    dim3 block(32, 32, 1);
    MatmulKernelV1<<<grid, block>>>(a, b, out, M, N, P);
}

void MatmulCoreV2(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    dim3 grid(std::ceil(M / 32.0), std::ceil(P / 32.0));
    dim3 block(32 * 32);
    MatmulKernelV2<<<grid, block>>>(a, b, out, M, N, P);
}

void MatmulCoreV3(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    dim3 grid(std::ceil(M / 32.0), std::ceil(P / 32.0));
    dim3 block(32 * 32);
    MatmulKernelV3<<<grid, block>>>(a, b, out, M, N, P);
}

void MatmulCoreV4(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    dim3 grid(std::ceil(static_cast<double>(M) / (V4::ROW_BLOCK_SIZE)), std::ceil(static_cast<double>(P) / V4::COL_BLOCK_SIZE));
    dim3 block((V4::ROW_BLOCK_SIZE / V4::TILE_ROW_SIZE) * V4::COL_BLOCK_SIZE);
    V4::MatmulKernelV4<<<grid, block>>>(a, b, out, M, N, P);
}

void MatmulCoreV5(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    dim3 grid(std::ceil(static_cast<double>(M) / (V5::ROW_BLOCK_SIZE)), std::ceil(static_cast<double>(P) / V5::COL_BLOCK_SIZE));
    dim3 block((V5::ROW_BLOCK_SIZE / V5::TILE_ROW_SIZE) * (V5::COL_BLOCK_SIZE / V5::TILE_COL_SIZE));
    V5::MatmulKernelV5<<<grid, block>>>(a, b, out, M, N, P);
}

void MatmulCoreV6(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    dim3 grid(std::ceil(static_cast<double>(M) / (V6::ROW_BLOCK_SIZE)), std::ceil(static_cast<double>(P) / V6::COL_BLOCK_SIZE));
    dim3 block((V6::ROW_BLOCK_SIZE / V6::TILE_ROW_SIZE) * (V6::COL_BLOCK_SIZE / V6::TILE_COL_SIZE));
    V6::MatmulKernelV6<<<grid, block>>>(a, b, out, M, N, P);
}

void MatmulCoreV6_Author(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    dim3 grid(std::ceil(static_cast<double>(P) / 128), std::ceil(static_cast<double>(M) / 128));
    dim3 block(256);
    V6_Author::sgemmVectorize<<<grid, block>>>(M, P, N, 1, (float*)a, (float*)b, 0, out);
}

void MatmulCoreV10(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    dim3 grid(std::ceil(static_cast<double>(M) / (V10::ROW_BLOCK_SIZE)), std::ceil(static_cast<double>(P) / V10::COL_BLOCK_SIZE));
    dim3 block(V10::THREAD_NUM_PER_BLOCK);
    V10::MatmulKernelV10<<<grid, block>>>(a, b, out, M, N, P);
}

void MatmulCoreV10_Author(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    dim3 grid(std::ceil(static_cast<double>(P) / (128)), std::ceil(static_cast<double>(M) / 64));
    dim3 block(128);
    V10_Author::sgemmWarptiling<<<grid, block>>>(M, P, N, 1, (float*)a, (float*)b, 0, out);
}

// 这里传的是Host指针
std::vector<double> MatmulProfile(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P, uint32_t repeat_num, uint32_t ver) {
    scalar_t *cuda_a, *cuda_b, *cuda_out;
    cudaMalloc(&cuda_a, sizeof(scalar_t) * M * N);
    cudaMalloc(&cuda_b, sizeof(scalar_t) * N * P);
    cudaMalloc(&cuda_out, sizeof(scalar_t) * M * P);

    cudaMemcpy(cuda_a, a, sizeof(scalar_t) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(scalar_t) * N * P, cudaMemcpyHostToDevice);

    // Invoke kernel
    std::vector<double> ret;
    ret.reserve(repeat_num);
    for (int i = 0; i < repeat_num; i++) {
        // cudaMemset(cuda_out, 0, sizeof(scalar_t) * M * P);
        uint64_t t0 = now_ns();
        switch (ver) {
            case 1:
                MatmulCoreV1(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            case 2:
                MatmulCoreV2(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            case 3:
                MatmulCoreV3(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            case 4:
                MatmulCoreV4(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            case 5:
                MatmulCoreV5(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            case 6:
                MatmulCoreV6(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            case 10:
                MatmulCoreV10(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            case 40:
                MatmulCoreV6_Author(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            case 41:
                MatmulCoreV10_Author(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            case 42:
                cuBlasImpl::MatmulCoreV42(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            default:
                assert(false);
        }

        cudaError_t lastErr = cudaPeekAtLastError();
        if (lastErr != cudaSuccess) {
            printf("Cuda Last Err %d %s\n", lastErr, cudaGetErrorString(lastErr));
        }
        assert(lastErr == cudaSuccess);

        cudaError_t err = cudaDeviceSynchronize();
        printf("Cuda Return %d %s\n", err, cudaGetErrorString(err));

        assert(err == cudaSuccess);
        uint64_t t1 = now_ns();

        ret.push_back((t1 - t0) * 1e-6);
    }

    cudaMemcpy(out, cuda_out, sizeof(scalar_t) * M * P, cudaMemcpyDeviceToHost);
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_out);

    return ret;
}