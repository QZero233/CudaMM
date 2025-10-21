//
// Created by qzero on 2025/10/20.
//

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

#include "cuda_matmul.h"

#include <cublas_v2.h>

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
        cudaMemset(cuda_out, 0, sizeof(scalar_t) * M * P);
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
            case 42:
                cuBlasImpl::MatmulCoreV42(cuda_a, cuda_b, cuda_out, M, N, P);
                break;
            default:
                assert(false);
        }
        cudaError_t err = cudaDeviceSynchronize();
        printf("Cuda Return %d %s\n", err, cudaGetErrorString(err));

        assert(err == cudaSuccess);
        uint64_t t1 = now_ns();

        ret.push_back((t1 - t0) * 1e-6);
    }

    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_out);

    return ret;
}