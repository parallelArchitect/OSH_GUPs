/* cuda_gups.cu
 *
 * GB10 GUPS — CUDA cudaMalloc (GPU-only) variant
 *
 * Same LFSR generator and XOR read-modify-write update as cpu_gups.c,
 * but the table lives in cudaMalloc'd device memory and the entire
 * update loop runs inside a single CUDA kernel, GPU-only.
 *
 * Purpose in the gb10-gups comparison set:
 *   cpu_gups.c     — CPU-only baseline, plain malloc, single thread
 *   cuda_gups.c    — THIS FILE: GPU-only, explicit device allocation
 *   managed_gups.c — cudaMallocManaged, CPU and GPU both touch the table
 *
 * cuda_gups establishes the GPU's own raw random-update rate in
 * isolation, with no CPU involvement and no coherence question at
 * all (device memory, device-only access — the simplest case).
 * This is the reference point managed_gups is compared against:
 * if managed_gups (CPU+GPU sharing one coherent table) performs close
 * to this GPU-only number, that is evidence GB10's hardware coherence
 * imposes little or no random-access penalty for sharing. If it
 * performs much worse, that is evidence of a real coherence cost
 * specific to the random-access pattern that uma_bw's sequential
 * tests did not capture.
 *
 * See cpu_gups.c for full provenance notes on the LFSR generator and
 * update kernel (derived from HPCC RandomAccess / OpenSHMEM GUPS,
 * Copyright 2011-2015 University of Houston System and UT-Battelle,
 * LLC, BSD-licensed) and for the table-sizing rationale.
 *
 * ---------------------------------------------------------------------
 * ORIGINAL LICENSE NOTICE — reproduced verbatim per its own terms
 * (this file reuses the same LFSR generator and update-kernel concept
 * as cpu_gups.c, so the same notice applies independently of whether
 * this file is distributed alongside cpu_gups.c or on its own)
 * ---------------------------------------------------------------------
 *
 * OpenSHMEM version:
 *
 * Copyright (c) 2011 - 2015
 *   University of Houston System and UT-Battelle, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * o Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * o Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * o Neither the name of the University of Houston System,
 *   UT-Battelle, LLC. nor the names of its contributors may be used to
 *   endorse or promote products derived from this software without specific
 *   prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Per HPCC's own BSD license (http://icl.cs.utk.edu/hpcc/faq/index.html#263):
 * "All original copyrights are retained." This file's modifications
 * are original work layered on top of the above, not a claim of
 * ownership over the reused generator/update-kernel concept.
 *
 * Table sizing here is necessarily smaller than the CPU variant's
 * MemAvailable-based calculation: GB10 GPU kernels allocate from the
 * SAME unified LPDDR5X pool via cudaMalloc, so in principle the same
 * sizing rule applies, but cudaMemGetInfo is known to be unreliable
 * on GB10 (see forums.developer.nvidia.com/t/gb10-hardware-baseline-
 * first-direct-measurements-and-findings/367851 — "memory clock N/A
 * on this platform", peak bandwidth reported as 0 by design rather
 * than fabricated). This program therefore takes table size as an
 * explicit command-line argument rather than attempting to auto-size
 * from a CUDA memory query that prior measurement has shown is not
 * trustworthy on this platform. Cross-check against cpu_gups's live
 * /proc/meminfo-derived size when choosing a value for direct
 * comparison.
 *
 * Build:
 *   nvcc -O3 -arch=sm_121 -o cuda_gups cuda_gups.cu
 *   (sm_121 = GB10 / Blackwell SM 12.1. Use CUDA 13.0 — prior GB10
 *   measurement found CUDA 13.1 produces broken event timing on this
 *   platform; this program does not use event timing, only host-side
 *   wall clock around a synchronized kernel launch, but build with
 *   13.0 for consistency with the rest of the gb10-gups toolchain.)
 *
 * Usage:
 *   ./cuda_gups <log2_table_size> [num_updates_multiplier]
 *
 *   No live-memory auto-sizing (see rationale above) — log2 size is
 *   required, not optional, unlike cpu_gups.c.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <cstring>
#include <pthread.h>
#include <dlfcn.h>
#include <dirent.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

#define POLY   0x0000000000000007ULL
#define PERIOD 1317624576693539401ULL

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                              \
            exit(1);                                                       \
        }                                                                    \
    } while (0)

/* ---------------------------------------------------------------------
 * NVML/spbm_hwmon thermal+power sampling — ported verbatim (same logic,
 * same ordering) from cpu_gups.c, which itself ports from the real,
 * working spark-gpu-throttle-check.py (NVML calls) and sparkview/
 * power.py (spbm_hwmon-first fallback chain). See cpu_gups.c for the
 * full provenance/rationale comments — not repeated here to avoid
 * duplication; the logic below is identical to that file's version.
 * ---------------------------------------------------------------------
 */
typedef int (*nvmlInit_t)(void);
typedef int (*nvmlShutdown_t)(void);
typedef int (*nvmlDeviceGetHandleByIndex_t)(unsigned int, void **);
typedef int (*nvmlDeviceGetClockInfo_t)(void *, int, unsigned int *);
typedef int (*nvmlDeviceGetPowerUsage_t)(void *, unsigned int *);
typedef int (*nvmlDeviceGetTemperature_t)(void *, int, unsigned int *);
typedef int (*nvmlDeviceGetPerformanceState_t)(void *, unsigned int *);
typedef int (*nvmlDeviceGetCurrentClocksThrottleReasons_t)(void *, unsigned long long *);

typedef struct {
    void *lib;
    void *device;
    nvmlShutdown_t shutdown;
    nvmlDeviceGetClockInfo_t get_clock;
    nvmlDeviceGetPowerUsage_t get_power;
    nvmlDeviceGetTemperature_t get_temp;
    nvmlDeviceGetPerformanceState_t get_pstate;
    nvmlDeviceGetCurrentClocksThrottleReasons_t get_throttle;
    int available;
} NvmlHandle;

static int nvml_init(NvmlHandle *h, unsigned int device_index)
{
    memset(h, 0, sizeof(*h));
    h->lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!h->lib) h->lib = dlopen("libnvidia-ml.so", RTLD_NOW);
    if (!h->lib) { h->available = 0; return 0; }

    nvmlInit_t nvml_init_fn = (nvmlInit_t)dlsym(h->lib, "nvmlInit_v2");
    if (!nvml_init_fn) nvml_init_fn = (nvmlInit_t)dlsym(h->lib, "nvmlInit");
    nvmlDeviceGetHandleByIndex_t get_handle =
        (nvmlDeviceGetHandleByIndex_t)dlsym(h->lib, "nvmlDeviceGetHandleByIndex_v2");
    if (!get_handle)
        get_handle = (nvmlDeviceGetHandleByIndex_t)dlsym(h->lib, "nvmlDeviceGetHandleByIndex");

    h->shutdown = (nvmlShutdown_t)dlsym(h->lib, "nvmlShutdown");
    h->get_clock = (nvmlDeviceGetClockInfo_t)dlsym(h->lib, "nvmlDeviceGetClockInfo");
    h->get_power = (nvmlDeviceGetPowerUsage_t)dlsym(h->lib, "nvmlDeviceGetPowerUsage");
    h->get_temp = (nvmlDeviceGetTemperature_t)dlsym(h->lib, "nvmlDeviceGetTemperature");
    h->get_pstate = (nvmlDeviceGetPerformanceState_t)dlsym(h->lib, "nvmlDeviceGetPerformanceState");
    h->get_throttle = (nvmlDeviceGetCurrentClocksThrottleReasons_t)dlsym(h->lib, "nvmlDeviceGetCurrentClocksThrottleReasons");

    if (!nvml_init_fn || !get_handle || nvml_init_fn() != 0) {
        h->available = 0;
        return 0;
    }
    if (get_handle(device_index, &h->device) != 0) {
        h->available = 0;
        return 0;
    }
    h->available = 1;
    return 1;
}

static void nvml_shutdown(NvmlHandle *h)
{
    if (h->available && h->shutdown) h->shutdown();
    if (h->lib) dlclose(h->lib);
}

/* Throttle reason bits — ported from spark-gpu-throttle-check.py.
 * See cpu_gups.c for the full rationale comment. */
#define NVML_THROTTLE_SW_POWER_CAP        0x0000000000000004ULL
#define NVML_THROTTLE_HW_SLOWDOWN         0x0000000000000008ULL
#define NVML_THROTTLE_SW_THERMAL_SLOWDOWN 0x0000000000000020ULL
#define NVML_THROTTLE_HW_THERMAL_SLOWDOWN 0x0000000000000040ULL
#define NVML_THROTTLE_HW_POWER_BRAKE      0x0000000000000080ULL
#define NVML_THROTTLE_PROBLEM_MASK \
    (NVML_THROTTLE_SW_POWER_CAP | NVML_THROTTLE_HW_SLOWDOWN | \
     NVML_THROTTLE_SW_THERMAL_SLOWDOWN | NVML_THROTTLE_HW_THERMAL_SLOWDOWN | \
     NVML_THROTTLE_HW_POWER_BRAKE)

#define POWER_SOURCE_LEN 16

static int read_power_w(NvmlHandle *nvml, double *out_power_w,
                         char *out_source, size_t source_len)
{
    DIR *hwmon_dir = opendir("/sys/class/hwmon");
    if (hwmon_dir) {
        struct dirent *entry;
        while ((entry = readdir(hwmon_dir)) != NULL) {
            if (entry->d_name[0] == '.') continue;

            char name_path[512];
            snprintf(name_path, sizeof(name_path),
                     "/sys/class/hwmon/%s/name", entry->d_name);
            FILE *nf = fopen(name_path, "r");
            if (!nf) continue;

            char name_buf[128] = {0};
            if (fgets(name_buf, sizeof(name_buf), nf)) {
                for (char *p = name_buf; *p; p++) *p = (char)tolower(*p);
                if (strstr(name_buf, "spbm")) {
                    char power_path[512];
                    snprintf(power_path, sizeof(power_path),
                             "/sys/class/hwmon/%s/power1_input",
                             entry->d_name);
                    FILE *pf = fopen(power_path, "r");
                    if (pf) {
                        long microwatts = 0;
                        if (fscanf(pf, "%ld", &microwatts) == 1) {
                            fclose(pf);
                            fclose(nf);
                            closedir(hwmon_dir);
                            *out_power_w = microwatts / 1000000.0;
                            snprintf(out_source, source_len, "spbm_hwmon");
                            return 1;
                        }
                        fclose(pf);
                    }
                }
            }
            fclose(nf);
        }
        closedir(hwmon_dir);
    }

    if (nvml->available && nvml->get_power) {
        unsigned int milliwatts = 0;
        if (nvml->get_power(nvml->device, &milliwatts) == 0) {
            *out_power_w = milliwatts / 1000.0;
            snprintf(out_source, source_len, "nvml");
            return 1;
        }
    }

    *out_power_w = 0.0;
    snprintf(out_source, source_len, "unavailable");
    return 0;
}

#define MAX_THERMAL_SAMPLES 4096

typedef struct {
    double elapsed_sec;
    unsigned int temp_c;
    unsigned int clock_mhz;
    double power_w;
    unsigned long long throttle_bitmask;
    char power_source[POWER_SOURCE_LEN];
} ThermalSample;

typedef struct {
    NvmlHandle *nvml;
    volatile int *stop_flag;
    double interval_sec;
    double t_start;
    unsigned int start_temp_c;
    unsigned int peak_temp_c;
    unsigned int end_temp_c;
    unsigned int start_clock_mhz;
    unsigned int min_clock_mhz;
    unsigned long long throttle_flags_seen;
    double start_power_w;
    double peak_power_w;
    double power_sum_w;
    char power_source[POWER_SOURCE_LEN];
    int sample_count;
    int got_first_sample;
    ThermalSample history[MAX_THERMAL_SAMPLES];
} ThermalSampler;

static double wall_seconds(void)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec + tp.tv_usec / 1.0e6;
}

static void *thermal_sample_thread(void *arg)
{
    ThermalSampler *s = (ThermalSampler *)arg;

    while (!(*s->stop_flag)) {
        unsigned int temp = 0, clock_mhz = 0;
        unsigned long long throttle_bits = 0;
        double power_w = 0.0;
        char power_source[POWER_SOURCE_LEN] = "unavailable";

        if (s->nvml->available) {
            if (s->nvml->get_temp)
                s->nvml->get_temp(s->nvml->device, 0, &temp);
            if (s->nvml->get_clock)
                s->nvml->get_clock(s->nvml->device, 0, &clock_mhz);
            if (s->nvml->get_throttle)
                s->nvml->get_throttle(s->nvml->device, &throttle_bits);
        }

        read_power_w(s->nvml, &power_w, power_source, sizeof(power_source));

        s->throttle_flags_seen |= (throttle_bits & NVML_THROTTLE_PROBLEM_MASK);

        if (!s->got_first_sample) {
            s->start_temp_c = temp;
            s->start_clock_mhz = clock_mhz;
            s->start_power_w = power_w;
            s->peak_temp_c = temp;
            s->peak_power_w = power_w;
            s->min_clock_mhz = 0xFFFFFFFFu;
            strncpy(s->power_source, power_source, POWER_SOURCE_LEN);
            s->power_source[POWER_SOURCE_LEN - 1] = '\0';
            s->got_first_sample = 1;
        } else {
            if (temp > s->peak_temp_c) s->peak_temp_c = temp;
            if (power_w > s->peak_power_w) s->peak_power_w = power_w;
            if (clock_mhz > 0 && clock_mhz < s->min_clock_mhz)
                s->min_clock_mhz = clock_mhz;
        }
        s->power_sum_w += power_w;
        s->end_temp_c = temp;

        if (s->sample_count < MAX_THERMAL_SAMPLES) {
            ThermalSample *hs = &s->history[s->sample_count];
            hs->elapsed_sec = wall_seconds() - s->t_start;
            hs->temp_c = temp;
            hs->clock_mhz = clock_mhz;
            hs->power_w = power_w;
            hs->throttle_bitmask = throttle_bits;
            strncpy(hs->power_source, power_source, POWER_SOURCE_LEN);
            hs->power_source[POWER_SOURCE_LEN - 1] = '\0';
        }
        s->sample_count++;

        struct timespec ts;
        ts.tv_sec  = (time_t)s->interval_sec;
        ts.tv_nsec = (long)((s->interval_sec - ts.tv_sec) * 1e9);
        nanosleep(&ts, NULL);
    }
    return NULL;
}

/* query_um_paradigm() — ported directly from parallelArchitect/
 * cuda-unified-memory-analyzer (um_analyzer_v8.cu, line 849), no
 * logic changed. Classifies the actual Unified Memory paradigm of
 * the device this binary is running on, at runtime, via CUDA device
 * attribute queries — not assumed from the build target.
 *
 * Returns one of:
 *   FULL_EXPLICIT          — discrete GPU, PCIe-managed (e.g. Pascal)
 *   FULL_HARDWARE_COHERENT — hardware-coherent UMA (e.g. GB10, GH200)
 *   FULL_SOFTWARE_COHERENT — pageable but no host-page-table coherence
 *   LIMITED                — no concurrent managed access
 *   UNKNOWN                — device attribute query failed
 *
 * This means this exact binary, unmodified, correctly self-reports
 * its real classification whether run on this Pascal GTX 1080 or on
 * real GB10 hardware later — no separately-named Pascal/GB10 files
 * needed, matching the architecture-aware pattern already
 * established and shipped in cuda-unified-memory-analyzer v8. */
static std::string query_um_paradigm(int device) {
    int concurrent = 0, pageable = 0, uses_host_pt = 0;
    bool c_ok = (cudaDeviceGetAttribute(&concurrent,
                   cudaDevAttrConcurrentManagedAccess, device) == cudaSuccess);
    bool p_ok = (cudaDeviceGetAttribute(&pageable,
                   cudaDevAttrPageableMemoryAccess, device) == cudaSuccess);
    bool h_ok = (cudaDeviceGetAttribute(&uses_host_pt,
                   cudaDevAttrPageableMemoryAccessUsesHostPageTables, device) == cudaSuccess);
    if (!c_ok) return "UNKNOWN";
    if (concurrent == 0) return "LIMITED";
    if (!p_ok || pageable == 0) return "FULL_EXPLICIT";
    if (!h_ok) return "FULL_SOFTWARE_COHERENT";  // conservative fallback
    return (uses_host_pt == 1) ? "FULL_HARDWARE_COHERENT" : "FULL_SOFTWARE_COHERENT";
}

/* starts(): identical jump-ahead LFSR seed algorithm to cpu_gups.c
 * and the original SHMEMRandomAccess.c starts(). Computed on host
 * before launch — only the per-update recurrence runs on device. */
static int64_t starts(uint64_t n)
{
    int i, j;
    uint64_t m2[64];
    uint64_t temp, ran;

    while ((int64_t)n < 0)  n += PERIOD;
    while (n > PERIOD)      n -= PERIOD;
    if (n == 0) return 0x1;

    temp = 0x1;
    for (i = 0; i < 64; i++) {
        m2[i] = temp;
        temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY : 0);
        temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY : 0);
    }

    for (i = 62; i >= 0; i--)
        if ((n >> i) & 1) break;

    ran = 0x2;
    while (i > 0) {
        temp = 0;
        for (j = 0; j < 64; j++)
            if ((ran >> j) & 1) temp ^= m2[j];
        ran = temp;
        i -= 1;
        if ((n >> i) & 1)
            ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY : 0);
    }
    return (int64_t)ran;
}

/* GPU kernel: each thread runs its own independent LFSR stream
 * (seeded via starts() at a unique offset per thread, computed on
 * host and passed in) and performs its share of the random updates
 * against the single shared device table. This mirrors the
 * "each thread permitted to look ahead" structure described in the
 * original benchmark's own comments, but with NO remote-process
 * messaging — every thread updates the same device-resident table
 * directly, since all threads already share the same physical
 * device memory (no partitioning required, unlike the SHMEM version
 * which partitions across separate processes/nodes). */
__global__ void gups_kernel(uint64_t *table, uint64_t nlocalm1,
                             uint64_t updates_per_thread,
                             int64_t *thread_seeds)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t ran = thread_seeds[tid];

    for (uint64_t iter = 0; iter < updates_per_thread; iter++) {
        ran = (ran << 1) ^ ((int64_t)ran < 0 ? (int64_t)POLY : 0);
        uint64_t index = (uint64_t)ran & nlocalm1;
        /* Same XOR read-modify-write as the original kernel.
         * Note: concurrent threads may race on the same index —
         * the original SHMEM version also tolerates "a small
         * (less than 1%) percentage of missed updates" per its
         * own correctness rules, so this is consistent with the
         * benchmark's documented tolerance rather than a bug. */
        table[index] ^= (uint64_t)ran;
    }
}

/* write_results_json() — identical to cpu_gups.c's version, ported
 * verbatim. See that file for the rationale comment. */
static void write_results_json(const char *binary_name,
                                const char *gpu_paradigm,
                                int log_table_size,
                                uint64_t table_size,
                                uint64_t num_updates,
                                double real_time_sec,
                                double gups,
                                const ThermalSampler *sampler)
{
    struct stat st;
    if (stat("results", &st) != 0) {
        if (mkdir("results", 0755) != 0) {
            fprintf(stderr, "[results] could not create results/ directory "
                             "— skipping JSON log\n");
            return;
        }
    }

    time_t now = time(NULL);
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm_buf);

    char filename[256];
    snprintf(filename, sizeof(filename), "results/%s_log2_%d_%s.json",
             binary_name, log_table_size, timestamp);

    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "[results] could not open %s for writing\n", filename);
        return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"binary\": \"%s\",\n", binary_name);
    fprintf(f, "  \"timestamp\": \"%s\",\n", timestamp);
    fprintf(f, "  \"um_paradigm\": \"%s\",\n", gpu_paradigm);
    fprintf(f, "  \"log2_table_size\": %d,\n", log_table_size);
    fprintf(f, "  \"table_size_entries\": %lu,\n", (unsigned long)table_size);
    fprintf(f, "  \"table_size_bytes\": %lu,\n",
            (unsigned long)(table_size * sizeof(uint64_t)));
    fprintf(f, "  \"num_updates\": %lu,\n", (unsigned long)num_updates);
    fprintf(f, "  \"real_time_sec\": %.6f,\n", real_time_sec);
    fprintf(f, "  \"gups\": %.9f,\n", gups);
    fprintf(f, "  \"thermal_power\": {\n");
    fprintf(f, "    \"sample_count\": %d,\n", sampler->sample_count);
    fprintf(f, "    \"interval_sec\": %.1f,\n", sampler->interval_sec);
    fprintf(f, "    \"power_source\": \"%s\",\n", sampler->power_source);
    fprintf(f, "    \"start_temp_c\": %u,\n", sampler->start_temp_c);
    fprintf(f, "    \"peak_temp_c\": %u,\n", sampler->peak_temp_c);
    fprintf(f, "    \"end_temp_c\": %u,\n", sampler->end_temp_c);
    fprintf(f, "    \"start_clock_mhz\": %u,\n", sampler->start_clock_mhz);
    fprintf(f, "    \"min_clock_mhz\": %u,\n",
            (sampler->min_clock_mhz == 0xFFFFFFFFu) ? 0 : sampler->min_clock_mhz);
    fprintf(f, "    \"throttle_flags_seen_hex\": \"0x%llx\",\n",
            sampler->throttle_flags_seen);
    fprintf(f, "    \"start_power_w\": %.2f,\n", sampler->start_power_w);
    fprintf(f, "    \"peak_power_w\": %.2f,\n", sampler->peak_power_w);
    fprintf(f, "    \"avg_power_w\": %.2f,\n",
            (sampler->sample_count > 0)
                ? sampler->power_sum_w / sampler->sample_count : 0.0);
    fprintf(f, "    \"samples\": [\n");

    int n = sampler->sample_count;
    if (n > MAX_THERMAL_SAMPLES) n = MAX_THERMAL_SAMPLES;
    for (int i = 0; i < n; i++) {
        const ThermalSample *hs = &sampler->history[i];
        fprintf(f, "      {\"t\": %.2f, \"temp_c\": %u, \"clock_mhz\": %u, "
                   "\"throttle_bitmask\": \"0x%llx\", "
                   "\"power_w\": %.2f, \"source\": \"%s\"}%s\n",
                hs->elapsed_sec, hs->temp_c, hs->clock_mhz,
                hs->throttle_bitmask, hs->power_w,
                hs->power_source, (i < n - 1) ? "," : "");
    }

    fprintf(f, "    ]\n");
    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);
    printf("\n[results] wrote %s\n", filename);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <log2_table_size> [num_updates_multiplier]\n",
                argv[0]);
        fprintf(stderr, "Example: %s 24 4.0   (16M-entry table, 128MB, "
                         "4x updates)\n", argv[0]);
        return 1;
    }

    int log_table_size = atoi(argv[1]);
    double update_multiplier = (argc > 2) ? atof(argv[2]) : 4.0;

    if (log_table_size < 1 || log_table_size > 33) {
        fprintf(stderr, "log2_table_size must be between 1 and 33\n");
        return 1;
    }

    uint64_t table_size = (uint64_t)1 << log_table_size;
    uint64_t nlocalm1 = table_size - 1;
    uint64_t total_updates = (uint64_t)(update_multiplier * (double)table_size);

    size_t table_bytes = table_size * sizeof(uint64_t);

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::string um_paradigm = query_um_paradigm(device);

    printf("=== CUDA GUPS (cudaMalloc, GPU-only variant) — %s detected at runtime ===\n",
           prop.name);
    printf("GPU              : %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("UM paradigm      : %s\n", um_paradigm.c_str());
    if (um_paradigm == "FULL_HARDWARE_COHERENT") {
        printf("                   (hardware-coherent UMA — GB10/GH200-class)\n");
    } else if (um_paradigm == "FULL_EXPLICIT") {
        printf("                   (discrete GPU, PCIe-managed — Pascal-class)\n");
    }
    printf("Table size       : 2^%d = %lu entries\n", log_table_size,
           (unsigned long)table_size);
    printf("Table memory     : %.3f GB\n", table_bytes / 1.0e9);
    printf("Total updates    : %lu (%.1fx table size)\n",
           (unsigned long)total_updates, update_multiplier);

    uint64_t *d_table = NULL;
    cudaError_t alloc_err = cudaMalloc(&d_table, table_bytes);
    if (alloc_err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for %.3f GB: %s\n",
                table_bytes / 1.0e9, cudaGetErrorString(alloc_err));
        fprintf(stderr, "Try a smaller log2_table_size.\n");
        return 1;
    }

    /* Initialize table on device: table[i] = i, same pattern as
     * cpu_gups.c and the original benchmark's init loop. */
    {
        uint64_t *h_init = (uint64_t *)malloc(table_bytes);
        if (!h_init) {
            fprintf(stderr, "Host staging buffer allocation failed\n");
            cudaFree(d_table);
            return 1;
        }
        for (uint64_t i = 0; i < table_size; i++) h_init[i] = i;
        CUDA_CHECK(cudaMemcpy(d_table, h_init, table_bytes,
                               cudaMemcpyHostToDevice));
        free(h_init);
    }

    int threads_per_block = 256;
    int num_blocks = 64; /* fixed grid — adjust per GPU if needed */
    int total_threads = threads_per_block * num_blocks;
    uint64_t updates_per_thread = total_updates / total_threads;
    if (updates_per_thread == 0) updates_per_thread = 1;

    /* Seed each thread's independent LFSR stream via starts(),
     * computed on host (matches original benchmark's GlobalStartMyProc
     * seeding pattern, but per-thread instead of per-process). */
    int64_t *h_seeds = (int64_t *)malloc(total_threads * sizeof(int64_t));
    for (int t = 0; t < total_threads; t++)
        h_seeds[t] = starts((uint64_t)(4 * t * updates_per_thread));

    int64_t *d_seeds = NULL;
    CUDA_CHECK(cudaMalloc(&d_seeds, total_threads * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds, total_threads * sizeof(int64_t),
                           cudaMemcpyHostToDevice));
    free(h_seeds);

    printf("Grid             : %d blocks x %d threads = %d total threads\n",
           num_blocks, threads_per_block, total_threads);
    printf("Updates/thread   : %lu\n\n", (unsigned long)updates_per_thread);

    /* NVML/spbm_hwmon thermal+power sampling — see top-of-file comment.
     * Started before the kernel launch so it catches the full duration,
     * including any ramp-up. */
    NvmlHandle nvml;
    int nvml_ok = nvml_init(&nvml, device);
    if (!nvml_ok) {
        fprintf(stderr, "[thermal] NVML unavailable — temp/clock sampling "
                         "disabled; power will still be attempted via "
                         "spbm_hwmon if present\n");
    }
    volatile int sampler_stop = 0;
    ThermalSampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    sampler.nvml = &nvml;
    sampler.stop_flag = &sampler_stop;
    sampler.interval_sec = 0.5;
    sampler.t_start = wall_seconds();
    pthread_t sampler_thread;
    pthread_create(&sampler_thread, NULL, thermal_sample_thread, &sampler);

    printf("Running update kernel...\n");

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    gups_kernel<<<num_blocks, threads_per_block>>>(
        d_table, nlocalm1, updates_per_thread, d_seeds);

    CUDA_CHECK(cudaEventRecord(stop));

    /* Poll instead of a single blocking sync, so elapsed time can be
     * shown while the async kernel runs. The kernel itself reports no
     * internal progress (no way to instrument a running GPU kernel
     * from the host mid-flight) — this shows elapsed wall-clock only,
     * which is still far better than zero feedback on a multi-minute
     * run, same usability problem cpu_gups.c's progress bar solves
     * for the CPU-loop case. */
    {
        double poll_start = wall_seconds();
        double last_print = poll_start;
        while (cudaEventQuery(stop) == cudaErrorNotReady) {
            double now = wall_seconds();
            if (now - last_print >= 1.0) {
                fprintf(stderr, "\r  kernel running...  elapsed %.0fs   ",
                        now - poll_start);
                fflush(stderr);
                last_print = now;
            }
            struct timespec sleep_ts = {0, 100000000L}; /* 100ms */
            nanosleep(&sleep_ts, NULL);
        }
        fprintf(stderr, "\r  kernel done.                              \n");
    }
    CUDA_CHECK(cudaEventSynchronize(stop));

    sampler_stop = 1;
    pthread_join(sampler_thread, NULL);

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double real_time = elapsed_ms / 1000.0;

    uint64_t actual_updates = (uint64_t)updates_per_thread * total_threads;
    double gups = 1e-9 * (double)actual_updates / real_time;

    printf("\n=== Results ===\n");
    printf("Real time        : %.6f seconds\n", real_time);
    printf("Actual updates   : %lu\n", (unsigned long)actual_updates);
    printf("GUP/s            : %.9f\n", gups);

    if (sampler.sample_count > 0) {
        printf("\n=== Thermal / Power (%d samples @ %.1fs interval) ===\n",
               sampler.sample_count, sampler.interval_sec);
        printf("Power source     : %s\n", sampler.power_source);
        if (nvml_ok) {
            printf("Start temp       : %u C\n", sampler.start_temp_c);
            printf("Peak temp        : %u C  (rise: +%d C)\n",
                   sampler.peak_temp_c,
                   (int)sampler.peak_temp_c - (int)sampler.start_temp_c);
            printf("End temp         : %u C\n", sampler.end_temp_c);
            if (real_time > 0.0) {
                double rise_rate = ((double)sampler.peak_temp_c
                                     - (double)sampler.start_temp_c) / real_time;
                printf("Temp rise rate   : %.3f C/sec\n", rise_rate);
            }
            printf("Start clock      : %u MHz\n", sampler.start_clock_mhz);
            if (sampler.min_clock_mhz != 0xFFFFFFFFu) {
                const char *clock_note = "";
                if (sampler.min_clock_mhz < sampler.start_clock_mhz) {
                    clock_note = (sampler.throttle_flags_seen != 0)
                        ? "  (clock dropped — REAL throttle flag set, see below)"
                        : "  (clock dropped, no throttle flag — Pascal idle-clock\n"
                          "                    cycling under memory-bound stalls, "
                          "not GB10-relevant)";
                }
                printf("Min clock seen   : %u MHz%s\n", sampler.min_clock_mhz,
                       clock_note);
            }
            if (sampler.throttle_flags_seen != 0) {
                printf("Throttle flags   : 0x%llx (", sampler.throttle_flags_seen);
                if (sampler.throttle_flags_seen & NVML_THROTTLE_SW_POWER_CAP)
                    printf("SW_POWER_CAP ");
                if (sampler.throttle_flags_seen & NVML_THROTTLE_HW_SLOWDOWN)
                    printf("HW_SLOWDOWN ");
                if (sampler.throttle_flags_seen & NVML_THROTTLE_SW_THERMAL_SLOWDOWN)
                    printf("SW_THERMAL_SLOWDOWN ");
                if (sampler.throttle_flags_seen & NVML_THROTTLE_HW_THERMAL_SLOWDOWN)
                    printf("HW_THERMAL_SLOWDOWN ");
                if (sampler.throttle_flags_seen & NVML_THROTTLE_HW_POWER_BRAKE)
                    printf("HW_POWER_BRAKE ");
                printf(")\n");
            } else {
                printf("Throttle flags   : none\n");
            }
        }
        if (strcmp(sampler.power_source, "unavailable") != 0) {
            double avg_power_w = sampler.power_sum_w / sampler.sample_count;
            printf("Start power      : %.1f W\n", sampler.start_power_w);
            printf("Avg power        : %.1f W\n", avg_power_w);
            printf("Peak power       : %.1f W\n", sampler.peak_power_w);
        } else {
            printf("Power            : unavailable\n");
        }
    } else {
        printf("\n(Run too short to capture a thermal sample at %.1fs interval)\n",
               sampler.interval_sec);
    }

    /* Sanity check sample entries, same pattern as cpu_gups.c */
    {
        uint64_t *h_check = (uint64_t *)malloc(table_bytes);
        CUDA_CHECK(cudaMemcpy(h_check, d_table, table_bytes,
                               cudaMemcpyDeviceToHost));
        printf("\nSanity check — sample entries:\n");
        for (int s = 0; s < 4; s++) {
            uint64_t idx = (table_size / 4) * s;
            printf("  table[%lu] = 0x%016lx\n", (unsigned long)idx,
                   (unsigned long)h_check[idx]);
        }
        free(h_check);
    }

    write_results_json("cuda_gups", um_paradigm.c_str(), log_table_size,
                        table_size, actual_updates, real_time, gups, &sampler);

    if (nvml_ok) nvml_shutdown(&nvml);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_seeds);
    cudaFree(d_table);

    return 0;
}
