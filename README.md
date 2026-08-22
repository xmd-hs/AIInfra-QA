# GPUForge

GPUForge 是一个面向 Transformer 推理的 C++17/CUDA AI Infra 原型项目，
将编译器 IR、Shape 推导、Kernel 规划、CPU/CUDA 算子、Paged KV Cache、
内存管理、请求调度、Telemetry 和 Benchmark 组织在一条可运行的工程链路中。

## 项目解析

```text
计算图 IR
  -> 输入、Shape、算子数量校验
  -> Constant Folding / DCE
  -> Fusion、Layout 推导
  -> Tile Schedule / Cost Model
  -> Launch Report / CUDA Codegen
  -> Runtime 执行、Tracing 和 Benchmark
```

不计算 `vendor_memorypool` SDK 和构建产物，核心源码规模如下：

| 模块 | 文件数 | 行数 |
|---|---:|---:|
| `src/` | 18 | 1,363 |
| `include/` | 18 | 617 |
| `cuda/` | 1 | 24* |
| `tests/` | 1 | 209 |
| `examples/` | 1 | 34 |
| **合计** | **39** | **2,247** |

\* CUDA 文件采用较紧凑的设备代码布局，物理行数不能完全反映 Kernel 复杂度。

## 主要模块

- 编译器 IR：Typed Shape/Layout、输入数量校验、Shape 校验、Constant
  Folding、显式输出、Dead Code Elimination、Fusion、Tile Planner 和
  Launch Report。
- 算子：CPU Tiled GEMM、Softmax、Scaled/Causal Attention、LayerNorm，
  以及 CUDA Tiled GEMM、WMMA Tensor Core GEMM 和 CUDA Attention。
- Runtime：Paged KV Cache、容量受限 MemoryPool、Prefill/Decode 调度、
  Stream、TraceRecorder、Metrics 和 AutoTune。
- 工程能力：CMake、MSVC/CUDA 集成、确定性测试、CPU/CUDA 一致性检查和
  Linux CI 配置。

## 测试数据

测试环境：NVIDIA GeForce RTX 4060 Laptop、NVIDIA Driver 555.97、CUDA 12.5。

| 后端 | 矩阵规模 | 迭代次数 | 平均延迟 | 吞吐 |
|---|---|---:|---:|---:|
| CPU FP32 GEMM | 256x256x256 | 5 | 212.482 ms | 0.158 GFLOP/s |
| CUDA Tiled GEMM | 256x256x256 | 5 | 0.706 ms | 47.498 GFLOP/s |

以上是本机工程测试数据，不是跨设备的通用性能承诺。实际结果会受功耗、
驱动、时钟频率和后台负载影响。

测试覆盖：

- 随机 GEMM 与独立参考实现逐元素比对。
- CUDA 可用时的 CPU/CUDA GEMM 一致性检查。
- 非方形 Causal Attention 和 CUDA Attention。
- 大正数/负数 Softmax 数值稳定性。
- LayerNorm 数值和参数边界。
- 非法 IR、算子输入数量、Shape、Constant Folding 和 DCE。
- Paged KV Cache 跨页读写、容量耗尽和失败 append 原子性。
- MemoryPool 容量、所有权、并发分配和释放。
- Graph 校验、Scheduler 阶段隔离、Telemetry 合并和 JSON 输出。

## 构建与测试

CPU：

```bash
cmake -S . -B build -DGPUFORGE_ENABLE_CUDA=OFF -DGPUFORGE_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CUDA：

```bash
cmake -S . -B build-cuda -DGPUFORGE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DGPUFORGE_BUILD_TESTS=ON
cmake --build build-cuda --config Release
ctest --test-dir build-cuda -C Release --output-on-failure
build-cuda/Release/gpuforge-bench.exe 256 256 256 5
```

## 当前边界

GPUForge 当前定位为 AI 编译器与推理 Runtime 原型。编译器已经支持
Elementwise CUDA 模板生成；通用 MatMul、Attention、LayerNorm 的完整
自动 Codegen 仍在完善。分布式模块目前是通信规划接口，尚未接入
NCCL/RDMA。这里明确列出边界，避免把原型能力误写成生产级功能。

## 目录结构

```text
include/gpuforge/  C++ 公共接口
src/               CPU 算子、编译器、Runtime、内存和 Telemetry
cuda/              CUDA 与 WMMA Kernel
tests/             集成测试和边界测试
examples/          Benchmark 可执行程序
vendor_memorypool/ Kama MemoryPool v3 依赖
.github/           Linux CPU、ASan 和 CUDA CI 配置
```
