# CUDA Benchmark Results

This file is intentionally maintained from real hardware runs. Do not copy
CPU numbers into the CUDA section or claim a result without recording the GPU,
CUDA, driver, compiler, matrix shape, dtype, tile configuration, warmup count,
iteration count, latency, and throughput.

## Command

```text
cmake -S . -B build-cuda -DGPUFORGE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --config Release
build-cuda/Release/gpuforge-bench.exe
```

## Required Record

| GPU | CUDA | Shape | Dtype | Kernel | Warmup | Iters | Latency (ms) | GFLOP/s |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| pending GPU run | pending | 256x256x256 | FP32 | CUDA tiled | 2 | 5 | pending | pending |

The current repository has a CPU baseline in `PROJECT_SHOWCASE.md`. CUDA
performance numbers remain pending until this benchmark is run on a machine
with an NVIDIA GPU and a working CUDA toolchain.
