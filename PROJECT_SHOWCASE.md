# GPUForge Project Showcase

GPUForge is a C++17 runtime and compiler prototype for Transformer inference.
It is designed to demonstrate compiler IR validation, shape-aware kernel
planning, CPU operator fallbacks, paged KV cache management, and memory-aware
runtime components.

## Architecture

```text
Module / IR
  -> IR validation and shape checks
  -> canonicalization and layout inference
  -> fusion groups and tile schedule
  -> launch report and cost model
  -> CPU fallback or CUDA kernel path
  -> executor, telemetry, KV cache, and benchmark
```

## Compiler Focus

- Typed `Shape` and `Layout` values for compiler IR.
- MatMul and elementwise broadcast validation before launch planning.
- Tile planner with boundary padding, shared-memory estimates, and WMMA path selection.
- Fusion groups, schedule selection, tuning database integration, and launch reports.
- Human-readable IR and compile diagnostics for invalid graphs.

## Operator Focus

- Tiled CPU GEMM with configurable tile size.
- Numerically stable row-wise Softmax.
- Scaled and causal Attention with non-square query/key lengths.
- LayerNorm with parameter and epsilon validation.
- CUDA entry points with CPU fallback when CUDA is unavailable.

## Runtime and C++ Engineering

- Paged KV cache with cross-page append/read, page location, capacity checks,
  and sequence reclamation.
- Capacity-enforced `MemoryPool` wrapper over Kama MemoryPool v3.
- Thread-safe allocation tracking and concurrent allocate/release coverage.
- Scheduler, executor, trace recorder, metrics, and benchmark helpers.

## Verification

The current Windows/MSVC verification covers:

- GPUForge integration build and core test.
- Kama MemoryPool v3 unit test and performance smoke test.
- Deterministic randomized GEMM reference comparisons.
- Non-square Attention and Softmax/LayerNorm numerical tests.
- Invalid IR, shape mismatch, KV cache boundary, and scheduler cancellation tests.

Build and test:

```text
cmake -S . -B build -DGPUFORGE_ENABLE_CUDA=OFF
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Benchmark Snapshot

The checked-in benchmark is a correctness-oriented CPU baseline. On the test
machine (RTX 4060 Laptop, MSVC Release), a 256x256x256 FP32 GEMM measured
about 48.7 ms and 0.69 GFLOP/s. CUDA numbers must be regenerated on a machine
with a working CUDA toolchain; they should not be treated as portable claims.

## Resume Positioning

Suggested positioning:

> Built a C++17/CUDA Transformer inference infrastructure prototype with a
> shape-aware compiler pipeline, tiled operators, paged KV cache, and a
> capacity-enforced memory pool; added invalid-IR diagnostics, numerical
> reference tests, and concurrency coverage.

This project is best suited for C++ infrastructure, CUDA kernel optimization,
inference runtime, and compiler engineering interviews.
