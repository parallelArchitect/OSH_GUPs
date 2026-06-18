/* managed_gups.cu
 *
 * cudaMallocManaged GUPS (CPU+GPU coherent-memory variant) — %s detected at runtime
 *
 * THIS is the variant that actually answers the open question stated
 * in cpu_gups.c's header: does GB10's confirmed near-zero coherence
 * overhead (uma_atomic SYS/GPU ratio 1.00x; uma_bw contention sweep
 * 2.2% drop under sequential cpu-write+gpu-read — see forums.developer.
 * nvidia.com/t/gb10-hardware-baseline-first-direct-measurements-and-
 * findings/367851) hold up under the GUPS-specific fully-randomized
 * giant-table access pattern, not just sequential/small-buffer access.
 *
 * Same LFSR generator and XOR update as cpu_gups.c and cuda_gups.cu.
 * The table here is allocated with cudaMallocManaged — on GB10's
 * hardware-coherent UMA this is the SAME physical LPDDR5X pool the
 * CPU malloc and cudaMalloc variants also use; what differs is that
 * THIS table is touched by BOTH a CPU thread and a GPU kernel
 * CONCURRENTLY, on overlapping random addresses, with no explicit
 * synchronization barrier between every update (deliberately — adding
 * a barrier every update would just re-introduce the SHMEM-style
 * cost we are explicitly trying to avoid measuring, see cpu_gups.c's
 * provenance notes on why that machinery was excluded).
 *
 * Three-way comparison this enables:
 *   cpu_gups.c     — CPU-only random GUP/s, no GPU involved
 *   cuda_gups.cu   — GPU-only random GUP/s, no CPU involved
 *   managed_gups.cu (this file) — CPU and GPU BOTH randomly updating
 *                    the same coherent table at the same time
 *
 * If managed_gups's aggregate GUP/s (CPU updates/sec + GPU updates/sec)
 * is close to the SUM of the two isolated baselines, that is strong
 * evidence GB10 hardware coherence imposes near-zero cost even under
 * concurrent random contention — consistent with, and extending, the
 * uma_bw sequential-contention finding. If it is significantly lower
 * than the sum, that demonstrates a real random-access-specific
 * coherence cost this is the first tool to isolate on GB10.
 *
 * On a discrete GPU, this same program would be expected to show a
 * large drop from page-fault-driven migration as the CPU and GPU
 * pull pages back and forth between separate physical pools — that
 * is the classic UVM "ping-pong" cost. GB10 has no separate pools to
 * migrate between, so if that cost appears here anyway, it would be
 * a genuinely new finding rather than an expected result.
 *
 * ---------------------------------------------------------------------
 * ORIGINAL LICENSE NOTICE — reproduced verbatim per its own terms
 * (this file reuses the same LFSR generator and update-kernel concept
 * as cpu_gups.c and cuda_gups.cu, originally from HPCC RandomAccess /
 * OpenSHMEM GUPS, so the same notice applies independently of whether
 * this file is distributed alongside the others or on its own)
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
 * Build:
 *   nvcc -O3 -arch=sm_121 -Xcompiler -pthread -o managed_gups managed_gups.cu
 *   (CUDA 13.0 — see cuda_gups.cu header for why 13.1 is avoided on GB10)
 *
 * Usage:
 *   ./managed_gups <log2_table_size> [num_updates_multiplier] [cpu_fraction]
 *
 *   cpu_fraction (default 0.5): fraction of total updates assigned to
 *   the CPU thread; the remainder goes to the GPU kernel. Both run
 *   concurrently via a host thread + an async kernel launch.
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
 * NVML/spbm_hwmon thermal+power sampling — ported verbatim from
 * cpu_gups.c / cuda_gups.cu. See cpu_gups.c for full provenance.
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

static double wall_seconds(void);  /* forward decl — real definition is
                                     * later in this file (used by the
                                     * pre-existing CPU thread); needed
                                     * here too for thermal sample
                                     * timestamps */

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
 * logic changed. See cuda_gups.cu for full explanation. */
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

static double wall_seconds(void)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec + tp.tv_usec / 1.0e6;
}

/* Identical starts() to cpu_gups.c / cuda_gups.cu — see those files
 * for the algorithm description. Kept identical across all three
 * variants so seed derivation is consistent for direct comparison. */
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

/* GPU kernel — identical update logic to cuda_gups.cu's gups_kernel,
 * operating on the managed pointer instead of a cudaMalloc pointer.
 * On GB10 these are the same physical memory class; the pointer type
 * difference is purely a CUDA API-level distinction here. */
__global__ void gups_kernel_managed(uint64_t *table, uint64_t nlocalm1,
                                     uint64_t updates_per_thread,
                                     int64_t *thread_seeds)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t ran = thread_seeds[tid];

    for (uint64_t iter = 0; iter < updates_per_thread; iter++) {
        ran = (ran << 1) ^ ((int64_t)ran < 0 ? (int64_t)POLY : 0);
        uint64_t index = (uint64_t)ran & nlocalm1;
        table[index] ^= (uint64_t)ran;
    }
}

/* Argument bundle for the CPU update thread, run concurrently with
 * the GPU kernel via pthread + async kernel launch. */
typedef struct {
    uint64_t *table;
    uint64_t  nlocalm1;
    uint64_t  num_updates;
    int64_t   seed;
    double    elapsed_seconds; /* output */
    const ThermalSampler *sampler; /* read-only, for live progress display —
                                     * background thread writes it, this
                                     * thread only reads, no lock needed
                                     * since we only ever read the latest
                                     * already-written summary fields */
} cpu_thread_args_t;

/* CPU-side update loop — identical logic to cpu_gups.c's main loop,
 * extracted into a thread function so it can run concurrently with
 * the GPU kernel against the SAME managed table.
 *
 * Progress display shows CPU thread % (a reasonable proxy for overall
 * progress since CPU and GPU each process exactly 50% of num_updates
 * concurrently) plus live temp/clock/power from the thermal sampler,
 * so there is something to look at besides a bare percentage during
 * a multi-minute run — not just "trust me, it's still running." */
static void *cpu_update_thread(void *arg)
{
    cpu_thread_args_t *a = (cpu_thread_args_t *)arg;
    int64_t ran = a->seed;

    double t0 = wall_seconds();
    const uint64_t PROGRESS_CHECK_MASK = (1ULL << 20) - 1;
    double last_progress_print = t0;

    for (uint64_t iter = 0; iter < a->num_updates; iter++) {
        ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY : 0);
        uint64_t index = (uint64_t)ran & a->nlocalm1;
        a->table[index] ^= (uint64_t)ran;

        if ((iter & PROGRESS_CHECK_MASK) == 0 && iter > 0) {
            double now = wall_seconds();
            if (now - last_progress_print >= 1.0) {
                double pct = 100.0 * (double)iter / (double)a->num_updates;
                double elapsed = now - t0;
                int eta_ready = (pct >= 2.0 && elapsed >= 5.0);
                double eta_sec = eta_ready
                    ? elapsed * (100.0 - pct) / pct : 0.0;

                /* Live machine state, read from the shared sampler —
                 * actual signal to look at during a multi-minute run,
                 * not just a percentage ticking up.
                 *
                 * min_clock_mhz is a running minimum across the WHOLE
                 * run so far — only print it once it has actually been
                 * set past its 0xFFFFFFFF sentinel (i.e. at least one
                 * full sample cycle has completed). Printing it before
                 * that showed garbage/sentinel-adjacent values that
                 * looked like a real number but were not. */
                unsigned int live_temp = 0, live_clock_min = 0;
                int have_clock_min = 0;
                double live_power = 0.0;
                if (a->sampler) {
                    live_temp  = a->sampler->end_temp_c;
                    live_power = a->sampler->peak_power_w;
                    if (a->sampler->min_clock_mhz != 0xFFFFFFFFu) {
                        live_clock_min = a->sampler->min_clock_mhz;
                        have_clock_min = 1;
                    }
                }

                char clock_str[24];
                if (have_clock_min)
                    snprintf(clock_str, sizeof(clock_str),
                             "%uMHz(min)", live_clock_min);
                else
                    snprintf(clock_str, sizeof(clock_str), "MHz(measuring)");

                char eta_str[32];
                if (eta_ready)
                    snprintf(eta_str, sizeof(eta_str), "ETA %.0fs", eta_sec);
                else
                    snprintf(eta_str, sizeof(eta_str), "ETA measuring...");

                fprintf(stderr,
                    "\r  %5.1f%% complete  |  elapsed %.0fs  |  %-16s  |  "
                    "%uC  %s  %.0fW(peak)   ",
                    pct, elapsed, eta_str, live_temp, clock_str, live_power);
                fflush(stderr);
                last_progress_print = now;
            }
        }
    }
    fprintf(stderr, "\r  100.0%% complete  |  done"
                     "                                                            "
                     "                              \n");
    a->elapsed_seconds = wall_seconds() - t0;
    return NULL;
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
        fprintf(stderr, "Usage: %s <log2_table_size> "
                         "[num_updates_multiplier] [cpu_fraction]\n", argv[0]);
        fprintf(stderr, "Example: %s 24 4.0 0.5\n", argv[0]);
        return 1;
    }

    int log_table_size = atoi(argv[1]);
    double update_multiplier = (argc > 2) ? atof(argv[2]) : 4.0;
    double cpu_fraction = (argc > 3) ? atof(argv[3]) : 0.5;

    if (log_table_size < 1 || log_table_size > 33) {
        fprintf(stderr, "log2_table_size must be between 1 and 33\n");
        return 1;
    }
    if (cpu_fraction < 0.0 || cpu_fraction > 1.0) {
        fprintf(stderr, "cpu_fraction must be between 0.0 and 1.0\n");
        return 1;
    }

    uint64_t table_size = (uint64_t)1 << log_table_size;
    uint64_t nlocalm1 = table_size - 1;
    uint64_t total_updates = (uint64_t)(update_multiplier * (double)table_size);
    uint64_t cpu_updates = (uint64_t)(cpu_fraction * (double)total_updates);
    uint64_t gpu_updates = total_updates - cpu_updates;

    size_t table_bytes = table_size * sizeof(uint64_t);

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::string um_paradigm = query_um_paradigm(device);

    printf("=== cudaMallocManaged GUPS (CPU+GPU coherent-memory variant) — %s detected at runtime ===\n",
           prop.name);
    printf("GPU              : %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("UM paradigm      : %s\n", um_paradigm.c_str());
    if (um_paradigm == "FULL_HARDWARE_COHERENT") {
        printf("                   (hardware-coherent UMA — GB10/GH200-class. No\n");
        printf("                    separate physical pools — concurrent CPU/GPU\n");
        printf("                    access does not require page migration.)\n");
    } else if (um_paradigm == "FULL_EXPLICIT") {
        printf("                   (discrete GPU, PCIe-managed — Pascal-class.\n");
        printf("                    Concurrent CPU/GPU access on cudaMallocManaged\n");
        printf("                    triggers page-fault-driven migration between\n");
        printf("                    separate physical CPU/GPU memory pools.)\n");
    }
    printf("Table size       : 2^%d = %lu entries\n", log_table_size,
           (unsigned long)table_size);
    printf("Table memory     : %.3f GB\n", table_bytes / 1.0e9);
    printf("Total updates    : %lu\n", (unsigned long)total_updates);
    printf("CPU share        : %lu (%.0f%%)\n", (unsigned long)cpu_updates,
           cpu_fraction * 100.0);
    printf("GPU share        : %lu (%.0f%%)\n", (unsigned long)gpu_updates,
           (1.0 - cpu_fraction) * 100.0);

    uint64_t *table = NULL;
    cudaError_t alloc_err = cudaMallocManaged(&table, table_bytes);
    if (alloc_err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for %.3f GB: %s\n",
                table_bytes / 1.0e9, cudaGetErrorString(alloc_err));
        fprintf(stderr, "Try a smaller log2_table_size.\n");
        return 1;
    }

    /* Initialize on host — valid directly since this is managed
     * memory; no explicit copy needed before first CPU touch. */
    for (uint64_t i = 0; i < table_size; i++) table[i] = i;

    int threads_per_block = 256;
    int num_blocks = 64;
    int total_gpu_threads = threads_per_block * num_blocks;
    uint64_t gpu_updates_per_thread = gpu_updates / total_gpu_threads;
    if (gpu_updates_per_thread == 0) gpu_updates_per_thread = 1;

    int64_t *h_seeds = (int64_t *)malloc(total_gpu_threads * sizeof(int64_t));
    for (int t = 0; t < total_gpu_threads; t++)
        h_seeds[t] = starts((uint64_t)(4 * t * gpu_updates_per_thread));

    int64_t *d_seeds = NULL;
    CUDA_CHECK(cudaMalloc(&d_seeds, total_gpu_threads * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds,
                           total_gpu_threads * sizeof(int64_t),
                           cudaMemcpyHostToDevice));
    free(h_seeds);

    cpu_thread_args_t cpu_args;
    cpu_args.table = table;
    cpu_args.nlocalm1 = nlocalm1;
    cpu_args.num_updates = cpu_updates;
    cpu_args.seed = starts(4); /* distinct seed offset from GPU threads */
    cpu_args.elapsed_seconds = 0.0;

    printf("\nLaunching CPU thread and GPU kernel CONCURRENTLY "
           "against the same table...\n");

    /* NVML/spbm_hwmon thermal+power sampling — started before the
     * concurrent section so it captures the full duration, including
     * any ramp during the page-fault-migration-heavy contention this
     * specific test is designed to stress. */
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

    cpu_args.sampler = &sampler;

    double t_start = sampler.t_start;

    pthread_t cpu_thread;
    pthread_create(&cpu_thread, NULL, cpu_update_thread, &cpu_args);

    /* Async kernel launch — does not block while the CPU thread runs.
     * This is the actual concurrent-contention condition: both the
     * CPU thread above and this kernel are touching `table` at the
     * same time, on independent random streams, with no barrier
     * between them until the join/sync below. */
    gups_kernel_managed<<<num_blocks, threads_per_block>>>(
        table, nlocalm1, gpu_updates_per_thread, d_seeds);

    pthread_join(cpu_thread, NULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    sampler_stop = 1;
    pthread_join(sampler_thread, NULL);

    double t_end = wall_seconds();
    double wall_clock_total = t_end - t_start;

    uint64_t actual_gpu_updates = gpu_updates_per_thread * total_gpu_threads;
    uint64_t actual_total = cpu_updates + actual_gpu_updates;

    double cpu_gups_rate = 1e-9 * (double)cpu_updates / cpu_args.elapsed_seconds;
    double aggregate_gups = 1e-9 * (double)actual_total / wall_clock_total;

    printf("\n=== Results ===\n");
    printf("Wall clock (concurrent CPU+GPU) : %.6f seconds\n", wall_clock_total);
    printf("CPU thread elapsed               : %.6f seconds\n",
           cpu_args.elapsed_seconds);
    printf("CPU-only GUP/s (within this run) : %.9f\n", cpu_gups_rate);
    printf("Aggregate GUP/s (CPU+GPU combined, wall clock): %.9f\n",
           aggregate_gups);
    printf("\nCompare this aggregate figure against the SUM of\n");
    printf("cpu_gups.c's isolated rate + cuda_gups.cu's isolated rate\n");
    printf("at the same table size, to assess concurrent-contention cost.\n");

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
            if (wall_clock_total > 0.0) {
                double rise_rate = ((double)sampler.peak_temp_c
                                     - (double)sampler.start_temp_c)
                                    / wall_clock_total;
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

    printf("\nSanity check — sample entries:\n");
    for (int s = 0; s < 4; s++) {
        uint64_t idx = (table_size / 4) * s;
        printf("  table[%lu] = 0x%016lx\n", (unsigned long)idx,
               (unsigned long)table[idx]);
    }

    write_results_json("managed_gups", um_paradigm.c_str(), log_table_size,
                        table_size, actual_total, wall_clock_total,
                        aggregate_gups, &sampler);

    if (nvml_ok) nvml_shutdown(&nvml);
    cudaFree(d_seeds);
    cudaFree(table);

    return 0;
}
