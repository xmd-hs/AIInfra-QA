# GPUForge

GPUForge 是一个面向大模型训练与推理的 C++17/CUDA AI Infra 性能工程项目，围绕两条主线组织实现：

- **Tensor Core 与 GPU 算子优化**：理解 GPU 执行模型、显存层次和矩阵计算路径，编写并选择 CUDA/WMMA Kernel。
- **编译器式图优化与调度**：从计算图 IR 出发，完成 Shape/Layout 推理、算子融合、代价分析、Tile Schedule 和 Kernel Launch 计划生成。

项目将编译器、算子、显存、推理调度和性能分析组织成一条可运行的工程链路。

## 核心模块

### Tensor Core 与 CUDA Kernel

- CPU 参考 Tensor 运行时。
- Shared Memory Tiled FP32 GEMM。
- WMMA FP16 输入、FP32 累加 Tensor Core GEMM。
- 融合 QK-Softmax-V Attention 参考 Kernel。
- Tile Planner：矩阵维度对齐、padding、grid、shared memory 和 workspace 估算。
- Tensor Core 合法性检查与数值误差比较工具。
- 无 CUDA 环境下的 CPU fallback。

### 编译器与 Kernel 计划

- `Module`、`Node`、`Value`、`Shape` 组成的计算图 IR。
- DType、Layout 和 Shape 表达。
- MatMul、Broadcast、Reshape、Reduce 的 Shape Inference。
- Canonicalization、Fusion、Dead Code 分析接口。
- Cost Model、Tile/Warp Schedule 和 Kernel 选择。
- `CompilePipeline` 生成 LaunchPlan，包括线程数、shared memory 和预估延迟。
- TuneDatabase 保存 GEMM 调优结果，支持配置复用和文件持久化。

### 推理运行时

- Paged KV Cache：跨 page 写入、读回、逻辑 offset 到物理 page 索引、容量耗尽检查和 sequence 回收。
- Prefill/Decode 两阶段请求调度器。
- Continuous batching 风格的 Batch 规划、请求取消和完成回收。
- Executor、Stream、TraceRecorder 和 Metrics。
- BenchmarkSuite 输出延迟、GFLOP/s、GB/s 和 Markdown 表格。

### 内存与通信

- `gpuforge::MemoryPool` 是 Kama MemoryPool v3 的兼容适配层。
- Kama SDK 提供 ThreadCache、CentralCache、PageCache 三级内存池。
- AllReduce、AllGather、ReduceScatter、Broadcast 通信规划。
- 通信分块及计算/通信 overlap 规划。

## 构建与测试

CPU fallback：

```bash
cmake -S . -B build -DGPUFORGE_ENABLE_CUDA=OFF
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CUDA：

```bash
cmake -S . -B build-cuda -DGPUFORGE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --config Release
ctest --test-dir build-cuda -C Release --output-on-failure
```

Kama MemoryPool SDK 位于 `vendor_memorypool/v3`，通过 CMake 子目录参与构建。`build/`、`build-cuda/` 等目录均为生成产物。

## 实测数据

已在以下环境完成 CUDA 构建和测试：

- Windows 10
- Visual Studio 2022 / MSVC 19.44
- CMake 4.4.2
- CUDA 12.5
- NVIDIA GeForce RTX 4060 Laptop

256 x 256 x 256 FP32 GEMM：

```text
CPU 参考实现：48.5712 ms，0.690829 GFLOP/s
CUDA Tiled：   0.58876 ms，56.9917 GFLOP/s
相对 CPU 参考实现：约 82 倍加速
CTest：1/1 通过
```

## 测试内容

核心测试覆盖：

- GEMM、Softmax 和数值误差。
- Paged KV 跨 page append/read、索引定位、容量耗尽和 sequence 回收。
- Kama MemoryPool 分配与释放。
- Scheduler 请求提交、取消、Batch 生成和完成回收。
- Tensor Core 对齐矩阵与非对齐矩阵的 Tile Planner 选择。
- Compiler Pipeline Launch Report。
- Trace Event 和 Chrome Trace JSON 导出。

## 项目架构

```text
计算图 IR
  -> Shape/Layout 推理
  -> Canonicalization 与算子融合
  -> Cost Model 与 Tile Schedule
  -> Tensor Core / CUDA Tiled Kernel 选择
  -> LaunchPlan 与 Executor
  -> Paged KV Cache 与请求调度
  -> Benchmark、Trace 和 Auto-Tuning Cache
```

## 目录结构

```text
include/gpuforge/     C++ 公共接口
src/                  CPU Runtime、编译器和调度实现
cuda/                 CUDA Kernel
examples/             Benchmark 示例
tests/                集成测试和边界测试
vendor_memorypool/    Kama MemoryPool SDK
```
