# gb10-gups

Random-access memory characterization for NVIDIA GB10 using a
GUPS-derived workload.

Target platform:

- NVIDIA GB10 (SM 12.1)
- DGX Spark and Spark-class systems
- Hardware-coherent unified memory
- Shared LPDDR5X memory subsystem

## Purpose

Previous GB10 characterization focused primarily on sequential,
streaming, and fixed-buffer access patterns. Representative
measurements include:

- SYS-scope vs GPU-scope atomic latency ratio: 1.00x
- CPU-write / GPU-read contention throughput loss: ~2.2%
- No measurable migration-related cold-start penalty observed

Reference: https://forums.developer.nvidia.com/t/gb10-hardware-baseline-first-direct-measurements-and-findings/367851

Those measurements characterize coherent memory behavior under
structured access patterns. This project extends that work to
randomized update workloads derived from HPCC RandomAccess (GUPS).

The objective is to characterize how GB10 behaves when CPU and GPU
access patterns defeat caches, prefetchers, and TLB locality, creating
a substantially different memory-access regime than previous GB10
measurements.

## What this measures

| Variant           | Memory model                         | Purpose                           |
| ----------------- | ------------------------------------ | --------------------------------- |
| `cpu_gups.c`      | Host memory (`malloc`)               | CPU random-update baseline        |
| `cuda_gups.cu`    | Device memory (`cudaMalloc`)         | GPU random-update baseline        |
| `managed_gups.cu` | Unified memory (`cudaMallocManaged`) | Concurrent CPU/GPU random updates |

The managed-memory variant measures aggregate behavior under
simultaneous CPU and GPU updates to a shared table. Observed
performance reflects the combined effects of coherence traffic,
memory-controller arbitration, cache ownership transitions, atomic
serialization, and random-access bandwidth limitations.

This benchmark does not isolate any individual mechanism. It
characterizes overall platform behavior under randomized concurrent
access.

If the aggregate rate remains close to the sum of the isolated CPU and
GPU baselines, that indicates the combined overhead is small enough
that throughput remains close to the isolated baseline sum under
randomized concurrent access.

If the aggregate rate is significantly lower, the suite can distinguish
driver-reported throttle events from ordinary memory-bound contention
using NVML throttle-state tracking. Additional targeted measurements
would still be required to attribute the underlying cause precisely.

## Table sizing

`cpu_gups.c` determines a default table size at startup using
`MemAvailable` from `/proc/meminfo` and the original HPCC RandomAccess
sizing rule: the largest power-of-two table not exceeding half of
available memory.

`cuda_gups.cu` and `managed_gups.cu` take table size as an explicit
argument. This keeps all three variants directly comparable and avoids
reliance on CUDA memory-availability queries for benchmark sizing.

## Build

```bash
# CPU variant — any Linux platform
gcc -O2 -Wall -o cpu_gups cpu_gups.c -lpthread -ldl

# CUDA variants — GB10 target
nvcc -O3 -arch=sm_121 -o cuda_gups cuda_gups.cu
nvcc -O3 -arch=sm_121 -Xcompiler -pthread -o managed_gups managed_gups.cu

# CUDA variants — Pascal toolchain-verification target
nvcc -O3 -arch=sm_61 -o cuda_gups cuda_gups.cu
nvcc -O3 -arch=sm_61 -Xcompiler -pthread -o managed_gups managed_gups.cu
```

CUDA 13.0 is the recommended GB10 build target.

Prior GB10 characterization observed incorrect PTX `%clock64`
behavior under CUDA 13.1 and 13.2 on SM 12.1.
Source: https://forums.developer.nvidia.com/t/gb10-hardware-baseline-first-direct-measurements-and-findings/367851/9

This suite uses CUDA event timing rather than direct `%clock64`
timing, but CUDA 13.0 is retained for consistency with the broader
GB10 diagnostic toolchain.

## Usage

```bash
# CPU baseline — auto-sizes from live /proc/meminfo
./cpu_gups

# Force a specific size for direct comparison across all three:
./cpu_gups 29 4.0
./cuda_gups 29 4.0
./managed_gups 29 4.0 0.5    # 0.5 = 50/50 CPU/GPU update split

# Pass isolated rates from prior cpu_gups/cuda_gups runs at the same
# table size to get coherence_efficiency in the output and JSON log
# (managed_gups GUP/s divided by the sum of the two isolated rates —
# omitted entirely, not fabricated as zero, if not provided):
./managed_gups 29 4.0 0.5 0.0233 0.590
```

All three write a structured JSON log to `results/`, including the
full per-sample thermal/power/throttle-bitmask time series, not just
summary statistics — see `results/<binary>_log2_<n>_<timestamp>.json`
after any run.

## Toolchain verification

No GB10 unit was available during development, so all three binaries
were compiled and executed on a GTX 1080 (SM 6.1, driver 570.195.03)
as a compiler/runtime validation target.

This verified:

- C/CUDA compilation
- CUDA kernel execution
- CPU threading
- JSON logging
- progress reporting
- NVML integration
- throttle-state decoding
- power-telemetry fallback behavior

The resulting GUP/s measurements are specific to a Pascal discrete-GPU
memory architecture and are not reported as GB10 results. The purpose
of this testing was to validate the implementation, not characterize
GB10 behavior.

## Verification status

| File | Compiled | Executed | Verified on GB10 |
|---|---|---|---|
| `cpu_gups.c` | Yes | Yes | **Not yet** |
| `cuda_gups.cu` | Yes | Yes | **Not yet** |
| `managed_gups.cu` | Yes | Yes | **Not yet** |

## Related GB10 Work

- https://github.com/parallelArchitect/cuda-unified-memory-analyzer
- https://github.com/parallelArchitect/sparkview
- https://github.com/parallelArchitect/spark-gpu-throttle-check
- https://github.com/parallelArchitect/nvidia-uma-fault-probe
