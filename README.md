# CUDA 矩阵乘法性能测试

简介
---
本项目实现并比较了多种基于 CUDA 的矩阵乘法（GEMM）内核（从简单基准到多种 tile / warp / 向量化优化）。提供性能测试、正确性验证与 cuBLAS 对比实现。

构建
---
直接使用make指令即可

```bash
make
```

编译后的二进制文件在`build/cuda_perf`

运行
---
基本用法：

```bash
build/cuda_perf M N P [repeats] [ver] [warmups] [seed]
```

参数说明：
- `M N P`：矩阵维度，计算 A(MxN) * B(NxP) -> C(MxP)。
- `repeats`：测时重复次数（默认 10）。
- `ver`：内核版本号（默认 10）。可选值包括：
    - `1,2,3`：基础实现/演进版本
    - `4,5,6,10`：作者实现的不同优化版本
    - `40`：作者的 V6 向量化实现（示例）
    - `41`：作者的 V10 Warp-tiling 实现
    - `42`：使用 cuBLAS（用于性能上界对比）
- `warmups`：预热次数（默认 1）
- `seed`：随机数种子（默认 12345）

示例：

```bash
# 运行 4096x4096 矩阵，10 次重复，使用内核版本 10
./perf 4096 4096 4096 10 10 1 12345
```

输出说明
---
程序会打印每次运行的耗时（ms），并在完成后输出：
- 最佳耗时与中位耗时（ms）
- 性能（GFLOPS）
- 估算内存带宽（GB/s）
- 相对最大误差（相对于参考实现），并在误差过大时触发断言或报警

依赖
---
- CUDA Toolkit（含 nvcc、cuBLAS）
- C++17 编译器
- CMake
- 推荐在具备足够显存的 NVIDIA GPU 上运行

文件结构
---
- `perf.cpp`：基准程序入口，构造矩阵并调用 `MatmulProfile`。
- `cuda_matmul.h` / `perf` 可调用的实现文件：包含多个版本的内核实现（V1..V10、作者版本、cuBLAS 封装等）。
- `README.md`：本文件。

验证
---
基准程序会使用 `reference_gemm`（基于 MatmulProfile 简化方式）生成参考结果并比较相对误差。误差阈值为 1e-1（可根据需要在代码中调整）。


