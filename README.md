# GPUForge

GPUForge 是一个面向 Transformer 推理的 C++17/CUDA 基础设施原型，重点
展示三条工程链路：AI 编译器、GPU 算子和推理 Runtime。

## 项目定位

```text
计算图 IR
  -> 校验与 Shape 推导
  -> Constant Folding / DCE / Fusion
  -> Tile Planner / Cost Model
  -> Launch Report / CUDA Codegen
  -> 算子执行、KV Cache、调度和性能分析
```

项目适合用于学习和展示：

- C++ 编译器和 IR 设计。
- CUDA GEMM、WMMA 和 Attention Kernel。
- Transformer 推理中的 KV Cache、MemoryPool 和请求调度。
- CMake、跨平台构建、测试和 Benchmark 工程实践。

## 核心实现

### 编译器

- Typed `Shape`、`Layout`、`Module`、`Node` 和 `Value`。
- IR 输入引用、算子输入数量和 Shape 合法性校验。
- MatMul、Broadcast、Reshape、Reduce 的 Shape 推导。
- Constant Folding，支持常量 Add、Mul、Relu、Gelu、Cast、Reshape。
- 基于显式 outputs 的 Dead Code Elimination，并自动重映射节点 ID。
- Fusion Group、Layout Inference、Tile Schedule、Cost Model 和 Launch Report。
- Elementwise CUDA Kernel 模板生成。

### 算子

CPU 路径位于 `src/kernels.cpp`，CUDA 路径位于 `cuda/kernels.cu`。

| 算子 | CPU | CUDA |
|---|---|---|
| Tiled FP32 GEMM | 支持 | 支持 |
| WMMA Tensor Core GEMM | fallback | 支持 |
| Softmax | 支持 | 暂无独立 Kernel |
| Scaled/Causal Attention | 支持 | 支持 |
| LayerNorm | 支持 | 暂无独立 Kernel |

### Runtime 与内存

- Paged KV Cache：跨页追加、读取、定位、容量检测和 Sequence 回收。
- MemoryPool：容量限制、对齐分配、当前池字节统计、所有权跟踪和并发释放。
- Scheduler：Prefill/Decode 阶段隔离、请求取消和 Batch 生成。
- Executor、Stream、TraceRecorder、Metrics 和 AutoTune 基础组件。

## 测试与数据

测试环境：

```text
Windows 10/11
MSVC 19.44
CUDA 12.5
NVIDIA GeForce RTX 4060 Laptop
```

已验证内容包括：

- CPU Release 构建和 CTest。
- CUDA Release 构建和 CTest。
- CPU/CUDA GEMM 数值一致性。
- 非方形 CPU/CUDA Attention。
- 随机 GEMM 参考实现比对。
- Softmax 数值稳定性和 LayerNorm 边界。
- IR、Shape、DCE、Graph、Scheduler、KV Cache 和 MemoryPool 边界测试。

### GEMM Benchmark

实测配置：`256 x 256 x 256`，5 次迭代。

| 后端 | 平均延迟 | 吞吐 |
|---|---:|---:|
| CPU FP32 | 212.482 ms | 0.158 GFLOP/s |
| CUDA Tiled | 0.706 ms | 47.498 GFLOP/s |

该数据是 RTX 4060 Laptop 上的本地工程测试结果，不代表所有设备的性能。

### 性能分析

按照同一台机器、同一矩阵规模和同一迭代次数计算：

```text
CPU latency / CUDA latency = 212.482 / 0.706 ≈ 300.7x
CUDA throughput / CPU throughput = 47.498 / 0.158 ≈ 300.6x
```

这组结果说明当前 CUDA Tiled GEMM 已经真正进入 GPU 执行路径，而不是
CPU fallback。性能提升主要来自：

- 32x32 Shared Memory tiling，减少 Global Memory 重复读取。
- 线程块并行计算矩阵的 Row/Column tile。
- 边界条件在 Kernel 内处理，支持非对齐矩阵尺寸。
- CUDA Runtime 使用独立 device buffer，并在 Kernel 完成后同步取回结果。
- AutoTuner 根据矩阵工作量选择 Tile 和 Warps 配置。

需要注意：当前 benchmark 包含 Host 到 Device 拷贝、Kernel 执行、同步和
Device 到 Host 拷贝，不是只测 Kernel 时间。因此它更接近一次完整算子调用
的端到端延迟。CPU 结果会受到 Windows 电源模式、线程调度和后台负载影响，
正式对比时应固定功耗模式、预热次数、矩阵布局、数据类型和迭代次数。

建议在其他 GPU 上使用相同命令重新记录数据：

```bash
gpuforge-bench.exe 128 128 128 5
gpuforge-bench.exe 256 256 256 5
gpuforge-bench.exe 512 512 512 5
```

## 构建

### CPU

```bash
cmake -S . -B build -DGPUFORGE_ENABLE_CUDA=OFF -DGPUFORGE_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

### CUDA

```bash
cmake -S . -B build-cuda \
  -DGPUFORGE_ENABLE_CUDA=ON \
  -DGPUFORGE_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --config Release
ctest --test-dir build-cuda -C Release --output-on-failure
build-cuda/Release/gpuforge-bench.exe 256 256 256 5
```

## 目录结构

```text
include/gpuforge/  公共 C++ 接口
src/               编译器、CPU 算子、Runtime、内存和 Telemetry
cuda/              CUDA 与 WMMA Kernel
tests/             集成和边界测试
examples/          Benchmark 程序
benchmarks/        CUDA 数据记录模板
vendor_memorypool/ Kama MemoryPool v3 依赖
.github/           Linux CPU、ASan 和 CUDA CI
```

## 当前边界

GPUForge 当前是可运行的基础设施原型，不是完整生产级推理框架。通用
MatMul、Attention、LayerNorm 自动 Codegen、独立 CUDA Softmax/LayerNorm、
NCCL/RDMA 通信和完整图执行器仍需要继续扩展。README 中只记录已经构建、
测试或实测验证过的能力。
