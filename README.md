# GPUForge

GPUForge is a C++17/CUDA prototype for Transformer inference infrastructure.
It connects typed compiler IR, shape validation, kernel planning, CPU fallback
operators, CUDA kernels, paged KV cache, memory management, scheduling,
telemetry, and benchmarks.

## Project Analysis

```text
IR Module -> validation -> constant folding/DCE -> fusion/layout inference
  -> tile schedule/cost model -> launch report/codegen -> runtime and telemetry
```

Core source size, excluding the vendored Kama MemoryPool SDK and build output:

| Area | Files | Lines |
|---|---:|---:|
| `src/` | 18 | 1,363 |
| `include/` | 18 | 617 |
| `cuda/` | 1 | 24* |
| `tests/` | 1 | 209 |
| `examples/` | 1 | 34 |
| **Total** | **39** | **2,247** |

\* Physical line count; device code is intentionally dense.

## Components

- Compiler IR: typed Shape/Layout, arity and shape validation, constant
  folding, explicit outputs, DCE, fusion, tile planning, and launch reports.
- Operators: tiled CPU GEMM, Softmax, scaled/causal Attention, LayerNorm,
  CUDA tiled GEMM, WMMA GEMM, and CUDA Attention.
- Runtime: Paged KV Cache, capacity-enforced MemoryPool, request scheduling,
  streams, tracing, metrics, and AutoTune.
- Engineering: CMake, MSVC/CUDA integration, deterministic tests, CPU/CUDA
  consistency checks, and Linux CI definitions.

## Verification Data

Measured on NVIDIA GeForce RTX 4060 Laptop, driver 555.97, CUDA 12.5:

| Backend | Shape | Iterations | Mean latency | Throughput |
|---|---|---:|---:|---:|
| CPU FP32 GEMM | 256x256x256 | 5 | 212.482 ms | 0.158 GFLOP/s |
| CUDA tiled GEMM | 256x256x256 | 5 | 0.706 ms | 47.498 GFLOP/s |

These are local engineering measurements, not portable performance claims.

Tests cover randomized GEMM reference comparison, CPU/CUDA consistency,
non-square causal Attention, stable Softmax, LayerNorm, invalid IR and arity,
constant folding, DCE, KV Cache boundaries, MemoryPool capacity/concurrency,
Graph validation, Scheduler phase isolation, telemetry merge, and JSON output.

## Build and Test

```bash
cmake -S . -B build -DGPUFORGE_ENABLE_CUDA=OFF -DGPUFORGE_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CUDA:

```bash
cmake -S . -B build-cuda -DGPUFORGE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --config Release
ctest --test-dir build-cuda -C Release --output-on-failure
build-cuda/Release/gpuforge-bench.exe 256 256 256 5
```

## Current Boundaries

GPUForge is an infrastructure prototype. Elementwise CUDA templates are
available through the compiler codegen; general-purpose MatMul, Attention and
LayerNorm code generation is not claimed as complete. Distributed communication
is a planning interface, not an NCCL/RDMA transport. These boundaries are kept
explicit so benchmark and interview claims remain reproducible.

## Layout

```text
include/gpuforge/  public C++ interfaces
src/               CPU operators, compiler, runtime, memory, telemetry
cuda/              CUDA and WMMA kernels
tests/             integration and boundary tests
examples/          benchmark executable
vendor_memorypool/ Kama MemoryPool v3 dependency
.github/           Linux CPU, ASan, and CUDA CI workflows
```
