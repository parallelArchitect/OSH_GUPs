# Changelog

All notable changes to gb10-gups are documented here.

### Added

**Initial three-file suite**
- `cpu_gups.c` — CPU-only GUPS baseline, plain `malloc`, auto-sizes
  table from live `/proc/meminfo` MemAvailable.
- `cuda_gups.cu` — GPU-only GUPS baseline, `cudaMalloc`, explicit
  table size argument (CUDA memory queries unreliable on GB10).
- `managed_gups.cu` — CPU+GPU concurrent GUPS, `cudaMallocManaged`,
  configurable CPU/GPU update split.
- LFSR random generator, `starts()` seed-jump, and XOR
  read-modify-write update logic ported from HPCC RandomAccess /
  OpenSHMEM GUPS (Copyright 2011-2015 University of Houston System
  and UT-Battelle, LLC, BSD-licensed). SHMEM multi-process
  partitioning machinery intentionally not reused — does not apply
  to single-chip unified memory.

**License compliance fix**
- Added the verbatim BSD 3-Clause notice (full text, not paraphrased)
  to all three files' headers, per that license's own redistribution
  terms. Original provenance description alone was not sufficient.

**Runtime platform classification**
- Ported `query_um_paradigm()` byte-for-byte from
  `cuda-unified-memory-analyzer`'s `um_analyzer_v8.cu`, verified
  against the original source via direct diff (zero logic
  differences). Both `.cu` files now self-report `FULL_EXPLICIT` /
  `FULL_HARDWARE_COHERENT` / `FULL_SOFTWARE_COHERENT` / `LIMITED` /
  `UNKNOWN` at runtime, correctly identifying whichever GPU
  architecture they are actually run on rather than assuming the
  build target.

**Thermal and power sampling**
- Added background-thread NVML sampling (`ThermalSampler`,
  `thermal_sample_thread`) to all three binaries, sampling every 0.5s
  for the full run duration — catches real climb on long runs,
  correctly shows nothing meaningful on sub-second runs.
- Added `read_power_w()` implementing the spbm_hwmon-first,
  NVML-fallback, explicit-"unavailable"-third chain, ported from the
  real `sparkview/power.py`. Power source is recorded and reported so
  readers know which sensor produced a given number.
- Added average power tracking alongside peak (matching the
  `gb10-kernel-probe` forum baseline's avg+peak reporting pattern,
  not peak alone).

**NVML throttle-reason bitmask tracking**
- Added `nvmlDeviceGetCurrentClocksThrottleReasons` reading and
  accumulation (`throttle_flags_seen`), with bitmask constants ported
  from `spark-gpu-throttle-check.py`.
- Fixed the misleading generic "possible throttle" message: the
  output now distinguishes a real driver-reported throttle flag from
  a clock drop with no flag set. On Pascal, confirmed the latter case
  in practice — see Fixed section below.

**Structured JSON results logging**
- Added `write_results_json()` to all three binaries — writes
  `results/<binary>_log2_<n>_<timestamp>.json` per run, including the
  full per-sample thermal/power/throttle time series (not just
  summary stats), table size, GUP/s, and UM paradigm classification.

**Progress display**
- Added live progress reporting to `cpu_gups.c`'s update loop and
  `managed_gups.cu`'s CPU thread (the only loops instrumentable from
  the host side — a running GPU kernel cannot report progress
  mid-flight). `cuda_gups.cu` uses `cudaEventQuery` polling instead
  of a blocking sync, printing elapsed time while the async kernel
  runs.
- `managed_gups.cu`'s progress line additionally shows live
  temp/clock/power read from the shared `ThermalSampler`, so there is
  real machine state to look at during multi-minute runs, not just a
  percentage.

### Fixed

- **Forward-declaration ordering bug**: `wall_seconds()` was used
  inside `thermal_sample_thread` before its definition later in
  `managed_gups.cu` (pre-existing in that file for the CPU thread).
  Caused a real compile failure (`identifier "wall_seconds" is
  undefined`), confirmed and fixed with a forward declaration.
- **ETA calculation noise**: dividing by a near-zero `pct` in the
  first few progress samples produced wildly inflated ETA estimates
  (observed: "ETA 29046s" / ~8 hours at 0.2% complete on a run that
  actually completed in ~250-300s). Fixed by suppressing the ETA
  display until at least 2% complete and 5 seconds elapsed.
- **Premature `min_clock_mhz` display**: the live progress line was
  reading `min_clock_mhz` before it had been set past its
  `0xFFFFFFFF` sentinel, printing a value that looked real but was
  not yet meaningful. Fixed by gating the display on an explicit
  "has this been set" check; shows "MHz(measuring)" until a real
  minimum exists.
- **Stray trailing text on completion line**: the "100.0% complete |
  done" message did not fully overwrite the longest possible prior
  progress line, leaving fragments like "peak)" visible after
  completion. Fixed by padding the completion message with
  sufficient trailing whitespace.
- **Mislabeled "GB10 GUPS" banner on Pascal hardware**: initial
  versions of `cuda_gups.cu` / `managed_gups.cu` printed a static
  "GB10 GUPS" banner regardless of actual hardware. Fixed by adding
  the `query_um_paradigm()` port (see Added) so the banner and
  classification reflect the real detected hardware.

### Verified

All three binaries compiled (`gcc -Wall` zero warnings; `nvcc
-arch=sm_61` zero errors) and ran to completion repeatedly on real
hardware (NVIDIA GTX 1080, SM 6.1, driver 570.195.03):

- `cpu_gups`: multiple runs at 2^22, 2^27, 2^29 (auto-sized).
- `cuda_gups`: multiple runs at 2^24, 2^29.
- `managed_gups`: multiple runs at 2^24, 2^29 (50/50 CPU/GPU split).

Confirmed working as designed: throttle-flag discrimination
correctly distinguished Pascal's idle-clock-state cycling (observed:
clock dropping to 139 MHz during long `managed_gups` runs, zero NVML
throttle flags set throughout) from a real hardware throttle event,
cross-checked against an independent clean `PASS` from
`spark-gpu-throttle-check.py` run minutes apart on the same card.

No run has been performed on GB10 hardware. All numbers produced to
date are from the Pascal control case described above.
