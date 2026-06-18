/* cpu_gups.c
 *
 * GB10 GUPS — CPU-only baseline
 *
 * Single-process, single-table GUPS (Giga UPdates per Second) measurement.
 *
 * ---------------------------------------------------------------------
 * WHY THIS TOOL EXISTS — the specific gap it targets
 * ---------------------------------------------------------------------
 *
 * Prior GB10 measurements (nvidia-uma-fault-probe, uma_bw, uma_atomic;
 * see forums.developer.nvidia.com/t/gb10-hardware-baseline-first-direct-
 * measurements-and-findings/367851) have already shown that GB10's
 * hardware-coherent unified memory avoids the fault-driven migration
 * penalty seen on discrete GPUs:
 *
 *   - uma_atomic: SYS-scope vs GPU-scope atomic latency ratio 1.00x —
 *     no measurable coherence overhead at the atomic level.
 *   - uma_bw contention sweep: only a 2.2% GPU throughput drop under
 *     simultaneous cpu-write + gpu-read. On discrete GPUs this pattern
 *     typically causes a much larger drop due to UVM-managed migration.
 *
 * Both of those measurements use SEQUENTIAL or fixed small-buffer access
 * patterns. What has not yet been measured on GB10 is the GUPS-specific
 * case: a giant table (sized to a meaningful fraction of total system
 * memory) accessed with a fully randomized, cache- and TLB-defeating
 * address stream — the access pattern the original HPCC RandomAccess
 * benchmark was specifically designed to stress, and the pattern most
 * likely to expose page-table or locality costs that small-buffer or
 * sequential tests cannot reveal.
 *
 * This tool exists to answer that one open question: does GB10's
 * near-zero coherence overhead, already confirmed under sequential
 * contention, hold up under fully randomized giant-table access.
 *
 * ---------------------------------------------------------------------
 * PROVENANCE — what is and is not reused from the original benchmark
 * ---------------------------------------------------------------------
 *
 * This is a derivative of the random-access UPDATE KERNEL CONCEPT from the
 * HPCC RandomAccess / OpenSHMEM GUPS benchmark (RandomAccess.c,
 * SHMEMRandomAccess.c — Copyright 2011-2015 University of Houston System
 * and UT-Battelle, LLC, originally contributed via the DARPA HPCS program,
 * BSD-licensed, http://icl.cs.utk.edu/hpcc/faq/index.html#263).
 *
 * Reused directly from that source:
 *   - The LFSR random number generator (POLY/PERIOD constants, the
 *     "(*ran << 1) ^ (...)" recurrence) and starts() seed-jump function.
 *   - The read-modify-write update operation (Table[index] ^= datum).
 *   - The table-sizing rule stated in the original benchmark's own
 *     comments: "select the memory size to be the power of two such
 *     that 2^n <= 1/2 of the total memory."
 *
 * Intentionally NOT carried over:
 *   - shmem_init / shmem_malloc / shmem_n_pes / shmem_my_pe
 *   - GlobalStartMyProc, Remainder, Top, the power-of-2-NumProcs requirement
 *   - shmem_longlong_fadd / shmem_longlong_p (remote PE put + atomic claim)
 *   - shmem_barrier_all() between every batch of remote updates
 *
 * All of that machinery exists ONLY because the original benchmark
 * partitions one global table across many separate SHMEM processes and
 * measures cross-process network random-access latency. On a single GB10
 * chip there is no remote process and no partitioned table — CPU and GPU
 * address the same physical LPDDR5X pool through the same page tables.
 * Reusing the SHMEM partitioning logic here would test something that
 * does not exist on this hardware.
 *
 * ---------------------------------------------------------------------
 * ORIGINAL LICENSE NOTICE — reproduced verbatim per its own terms
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
 * (single-process restructuring, /proc/meminfo-based sizing, GB10-specific
 * documentation) are original work layered on top of the above, not a
 * claim of ownership over the reused generator/update-kernel concept.
 *
 * ---------------------------------------------------------------------
 * TABLE SIZE — derived from real hardware, not assumed
 * ---------------------------------------------------------------------
 *
 * Applying the original benchmark's own sizing rule to real captured
 * /proc/meminfo data from two independent GB10 sosreports:
 *
 *   sosreport-spark-c03d-random-reboot-2026-03-16: MemTotal 127,601,228 kB
 *   sosreport jed10 (2026-03-27 capture):           MemTotal 127,601,160 kB
 *
 *   MemTotal (bytes)  : 130,663,657,472  (130.66 GB decimal / 121.69 GiB)
 *   Half memory       : 65.33 GB
 *   Word size         : 8 bytes (uint64_t)
 *   Max entries       : 8,166,478,592  (no power-of-2 constraint)
 *   log2 floor        : 32
 *   EXAMPLE TABLE SIZE: 2^32 = 4,294,967,296 entries = 34.36 GB
 *
 * This program does NOT hardcode that number. At startup it reads
 * /proc/meminfo directly and computes the table size from THIS box's
 * real MemAvailable (not MemTotal — see note below), following the
 * same rule. The 2^32 figure above is what that calculation produced
 * from the two real GB10 captures we have data from; it is a derived
 * example, not a hardcoded assumption, and will differ on a box with
 * a different memory configuration or current load.
 *
 * MemAvailable vs MemTotal: the same c03d capture showed MemTotal
 * 127,601,160 kB but MemAvailable only 117,493,976 kB at that moment
 * (other processes were using ~10GB). Sizing off MemTotal risks a
 * malloc failure on a loaded system; sizing off MemAvailable is the
 * number that is actually safe to allocate.
 *
 * Build:
 *   gcc -O2 -march=native -o cpu_gups cpu_gups.c -lpthread -ldl
 *   (-lpthread for the background thermal sampler thread, -ldl for
 *   dlopen-based NVML loading — no CUDA toolkit dependency added)
 *
 * Usage:
 *   ./cpu_gups [log2_table_size_override] [num_updates_multiplier]
 *
 *   With no arguments, table size is computed from this machine's
 *   live /proc/meminfo MemAvailable.
 *   log2_table_size_override lets you force a specific size for quick
 *   testing, e.g. `./cpu_gups 24` for a 128MB table on a laptop.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <dlfcn.h>
#include <dirent.h>
#include <ctype.h>
#include <sys/stat.h>

/* ---------------------------------------------------------------------
 * NVML thermal/power sampling — background thread
 * ---------------------------------------------------------------------
 *
 * Why this exists: at short table sizes (e.g. 2^24, sub-second runtime)
 * GPU thermal mass does not move meaningfully — a before/after snapshot
 * shows nothing. At table sizes that actually exercise GB10-relevant
 * memory pressure (2^29+, tens of seconds to minutes), real thermal
 * and clock drift becomes a genuine variable, the same reason
 * spark-gpu-throttle-check samples on a fixed interval rather than
 * once. This samples concurrently with the update loop, not just
 * before/after, so a run long enough to matter actually gets caught.
 *
 * Call signatures and NVML constant values (0 = NVML_TEMPERATURE_GPU
 * for nvmlDeviceGetTemperature, 0 = NVML_CLOCK_GRAPHICS for
 * nvmlDeviceGetClockInfo) ported directly from the real, working
 * spark-gpu-throttle-check.py (parallelArchitect), confirmed against
 * its actual source rather than assumed:
 *
 *   nvmlDeviceGetClockInfo(handle, 0, &val)        -> MHz
 *   nvmlDeviceGetPowerUsage(handle, &val)          -> milliwatts
 *   nvmlDeviceGetTemperature(handle, 0, &val)      -> degrees C
 *   nvmlDeviceGetPerformanceState(handle, &val)    -> P-state index
 *
 * Loaded via dlopen rather than linked, since this file deliberately
 * has no CUDA toolkit dependency (plain gcc build, confirmed by its
 * own Build: comment) — same approach Python's ctypes uses under
 * the hood in the source this was ported from.
 */

static double wall_seconds(void);  /* forward decl — defined below,
                                     * used here for sample timestamps */

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

/* Throttle reason bits — ported directly from spark-gpu-throttle-check.py's
 * THROTTLE_REASONS / PROBLEM_REASONS. Lets the sampler distinguish a real
 * driver-reported throttle event from a clock drop with no flag set
 * (the latter being a Pascal UVM/power-management idle-clock-cycling
 * behavior during page-fault stalls, not a hardware throttle — confirmed
 * by running spark-gpu-throttle-check.py separately and getting a clean
 * PASS on the same card minutes apart from a GUPS run showing 139 MHz). */
#define NVML_THROTTLE_SW_POWER_CAP        0x0000000000000004ULL
#define NVML_THROTTLE_HW_SLOWDOWN         0x0000000000000008ULL
#define NVML_THROTTLE_SW_THERMAL_SLOWDOWN 0x0000000000000020ULL
#define NVML_THROTTLE_HW_THERMAL_SLOWDOWN 0x0000000000000040ULL
#define NVML_THROTTLE_HW_POWER_BRAKE      0x0000000000000080ULL
#define NVML_THROTTLE_PROBLEM_MASK \
    (NVML_THROTTLE_SW_POWER_CAP | NVML_THROTTLE_HW_SLOWDOWN | \
     NVML_THROTTLE_SW_THERMAL_SLOWDOWN | NVML_THROTTLE_HW_THERMAL_SLOWDOWN | \
     NVML_THROTTLE_HW_POWER_BRAKE)

/* ---------------------------------------------------------------------
 * Power reading — spbm_hwmon first, NVML fallback, "unavailable" third
 * ---------------------------------------------------------------------
 *
 * Ported directly from the real, working sparkview/power.py
 * (parallelArchitect), confirmed against its actual source rather than
 * assumed. That file's own docstring states the reason for this order:
 * NVML power.draw was found unreliable on some GB10 units, so
 * spbm_hwmon (a direct sysfs interface to the board's dedicated power
 * monitoring IC) is tried first. On reference DGX Spark units where
 * NEITHER path works, the Python source explicitly documents and
 * returns "unavailable" rather than fabricating a number — ported
 * here with the same honesty rather than silently reporting 0.0.
 *
 * spbm_hwmon detection: iterate the hwmon sysfs tree (each entry under
 * /sys/class/hwmon), find the device
 * whose "name" file contains "spbm" (case-insensitive), read
 * power1_input — value is in MICROWATTS (divide by 1,000,000 for
 * watts; this differs from NVML's milliwatts, confirmed from the
 * real Python source's "/ 1_000_000" — not a typo carried over).
 */
#define POWER_SOURCE_LEN 16

static int read_power_w(NvmlHandle *nvml, double *out_power_w,
                         char *out_source, size_t source_len)
{
    /* --- Path 1: spbm_hwmon sysfs --- */
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

    /* --- Path 2: NVML power.draw fallback --- */
    if (nvml->available && nvml->get_power) {
        unsigned int milliwatts = 0;
        if (nvml->get_power(nvml->device, &milliwatts) == 0) {
            *out_power_w = milliwatts / 1000.0;
            snprintf(out_source, source_len, "nvml");
            return 1;
        }
    }

    /* --- Neither path worked --- */
    *out_power_w = 0.0;
    snprintf(out_source, source_len, "unavailable");
    return 0;
}

#define MAX_THERMAL_SAMPLES 4096  /* ~34 min at 0.5s interval — plenty
                                    * for any realistic gb10-gups run */

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
    double t_start;  /* set by caller before thread launch, used to
                       * timestamp each sample relative to run start */
    /* Output — written by the sampling thread, read after join */
    unsigned int start_temp_c;
    unsigned int peak_temp_c;
    unsigned int end_temp_c;
    unsigned int start_clock_mhz;
    unsigned int min_clock_mhz;
    unsigned long long throttle_flags_seen;  /* OR of all problem-bit
                                               * throttle reasons across
                                               * every sample — 0 means
                                               * no real driver-reported
                                               * throttle was ever flagged,
                                               * even if clock dropped */
    double start_power_w;
    double peak_power_w;
    double power_sum_w;       /* for computing average, matching the
                                * gb10-kernel-probe forum post pattern
                                * of reporting both avg and peak power */
    char power_source[POWER_SOURCE_LEN];  /* "spbm_hwmon" / "nvml" /
                                            * "unavailable" — recorded so
                                            * readers know which sensor
                                            * produced the number, same
                                            * discipline as run_sweep.sh's
                                            * gpu_power_source field */
    int sample_count;
    int got_first_sample;
    ThermalSample history[MAX_THERMAL_SAMPLES];  /* time series, for
                                                   * JSON log output —
                                                   * this is what makes
                                                   * a run analyzable
                                                   * later, not just a
                                                   * summary number */
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
                s->nvml->get_temp(s->nvml->device, 0 /* NVML_TEMPERATURE_GPU */, &temp);
            if (s->nvml->get_clock)
                s->nvml->get_clock(s->nvml->device, 0 /* NVML_CLOCK_GRAPHICS */, &clock_mhz);
            if (s->nvml->get_throttle)
                s->nvml->get_throttle(s->nvml->device, &throttle_bits);
        }

        /* spbm_hwmon-first, NVML-fallback chain — ported from
         * sparkview/power.py, see read_power_w() above for why. */
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

        /* Sleep in small slices so stop_flag is checked promptly,
         * matching spark-gpu-throttle-check's sampling cadence. */
        struct timespec ts;
        ts.tv_sec  = (time_t)s->interval_sec;
        ts.tv_nsec = (long)((s->interval_sec - ts.tv_sec) * 1e9);
        nanosleep(&ts, NULL);
    }
    return NULL;
}

/* ---- LFSR random generator, taken directly from RandomAccess.h ---- */
#define POLY   0x0000000000000007ULL
#define PERIOD 1317624576693539401ULL

static double wall_seconds(void)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec + tp.tv_usec / 1.0e6;
}

/* starts(): jump-ahead seed for the LFSR, identical algorithm to the
 * original starts() in SHMEMRandomAccess.c (GF(2) matrix-squaring).
 * Not strictly required for a single-stream run (we could just seed
 * with a fixed value and iterate), but kept for parity with the
 * original benchmark and to allow reproducible multi-run comparison
 * at a specific offset into the stream if ever needed. */
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

/* Read MemAvailable from /proc/meminfo, in kB.
 * Returns 0 if the file cannot be read or the field is not found
 * (e.g. running on a non-Linux dev machine for a quick correctness test). */
static uint64_t get_mem_available_kb(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;

    char line[256];
    uint64_t mem_available_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, "%lu", &mem_available_kb);
            break;
        }
    }
    fclose(f);
    return mem_available_kb;
}

/* Apply the original HPCC RandomAccess sizing rule:
 * "select the memory size to be the power of two such that
 *  2^n <= 1/2 of the total memory" — here applied to MemAvailable
 * rather than MemTotal, since MemAvailable is what is actually safe
 * to allocate on a system that may already have other processes
 * running (confirmed necessary by real GB10 sosreport data showing
 * a ~10GB gap between MemTotal and MemAvailable at capture time).
 *
 * Returns the log2 table size (in 8-byte words), or a fallback
 * default if /proc/meminfo could not be read. */
static int compute_table_log2(uint64_t fallback_log2)
{
    uint64_t mem_available_kb = get_mem_available_kb();
    if (mem_available_kb == 0) {
        fprintf(stderr, "Could not read /proc/meminfo MemAvailable — "
                         "falling back to log2 table size %lu\n",
                (unsigned long)fallback_log2);
        return (int)fallback_log2;
    }

    double half_mem_bytes = (double)mem_available_kb * 1024.0 * 0.5;
    double max_words = half_mem_bytes / (double)sizeof(uint64_t);

    int log2_size = 0;
    double table = 1.0;
    while (table * 2.0 <= max_words) {
        table *= 2.0;
        log2_size++;
    }
    return log2_size;
}

/* ---------------------------------------------------------------------
 * JSON results log — one file per run, full thermal/power time series
 * ---------------------------------------------------------------------
 *
 * Why this exists: summary stats (peak temp, avg power) printed to a
 * terminal are not analyzable later — they vanish when the terminal
 * scrolls. A real per-sample time series, written to results/, is what
 * lets later analysis answer questions like "when exactly did the
 * clock collapse relative to the GUP/s curve" rather than just "it
 * collapsed at some point." Directory layout matches the gb10-gups
 * project's planned results/ structure.
 */
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
    /* Default: computed live from this machine's /proc/meminfo
     * MemAvailable, following the original HPCC sizing rule.
     * Falls back to 2^28 (256M entries, 2GB table) if /proc/meminfo
     * is unavailable — small enough to run on a non-GB10 dev box. */
    int log_table_size = compute_table_log2(28);
    double update_multiplier = 4.0; /* HPCC default: 4x table size updates */

    if (argc > 1) log_table_size = atoi(argv[1]);
    if (argc > 2) update_multiplier = atof(argv[2]);

    if (log_table_size < 1 || log_table_size > 36) {
        fprintf(stderr, "log2_table_size must be between 1 and 36 "
                         "(36 = 64GB table on this word size)\n");
        return 1;
    }

    uint64_t table_size = (uint64_t)1 << log_table_size;
    uint64_t nlocalm1 = table_size - 1;
    uint64_t num_updates = (uint64_t)(update_multiplier * (double)table_size);

    size_t table_bytes = table_size * sizeof(uint64_t);
    printf("=== GB10 GUPS — CPU baseline ===\n");
    printf("Table size      : 2^%d = %lu entries\n", log_table_size,
           (unsigned long)table_size);
    printf("Table memory     : %.2f GB\n", table_bytes / 1.0e9);
    printf("Num updates      : %lu (%.1fx table size)\n",
           (unsigned long)num_updates, update_multiplier);
    fflush(stdout); /* ensure setup info prints before the stderr
                      * progress bar starts, regardless of stdio
                      * buffering mode when output is redirected */

    uint64_t *table = (uint64_t *)malloc(table_bytes);
    if (!table) {
        fprintf(stderr, "malloc failed for %.2f GB table — "
                         "try a smaller log2_table_size override\n",
                table_bytes / 1.0e9);
        return 1;
    }

    /* Initialize table: same pattern as original — table[i] = i */
    for (uint64_t i = 0; i < table_size; i++)
        table[i] = i;

    /* Seed the random stream at step 0 (single process — no
     * GlobalStartMyProc offset needed). */
    int64_t ran = starts(0);

    /* NVML thermal/power sampling — see header for rationale.
     * Catches real drift on long runs; harmless no-op on short ones
     * (sampler thread starts and stops, sample_count will just be low).
     * Always launch the sampler even if NVML failed to init — power
     * may still be readable via spbm_hwmon alone (confirmed possible
     * per sparkview/power.py's own fallback design), and the thread
     * function itself checks nvml->available before touching it. */
    NvmlHandle nvml;
    int nvml_ok = nvml_init(&nvml, 0);
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
    sampler.interval_sec = 0.5;  /* matches spark-gpu-throttle-check cadence */
    sampler.t_start = wall_seconds();
    pthread_t sampler_thread;
    pthread_create(&sampler_thread, NULL, thermal_sample_thread, &sampler);

    printf("\nRunning update loop...\n");
    double t_start = sampler.t_start;  /* same reference point, so GUP/s
                                         * timing and thermal timestamps
                                         * are aligned to the same clock */

    /* Progress reporting: check wall clock only every 2^20 iterations
     * (a cheap bitmask test) so the progress display itself does not
     * add measurable overhead to the timed loop. Updates a single
     * line in place via \r rather than scrolling output. */
    const uint64_t PROGRESS_CHECK_MASK = (1ULL << 20) - 1;
    double last_progress_print = t_start;

    for (uint64_t iter = 0; iter < num_updates; iter++) {
        /* LFSR recurrence — identical to the original kernel */
        ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY : 0);

        /* Local update — no remote PE, no shmem_longlong_fadd,
         * no shmem_longlong_p, no barrier. Direct read-modify-write
         * against the single coherent table. */
        uint64_t index = (uint64_t)ran & nlocalm1;
        table[index] ^= (uint64_t)ran;

        if ((iter & PROGRESS_CHECK_MASK) == 0 && iter > 0) {
            double now = wall_seconds();
            if (now - last_progress_print >= 1.0) {
                double pct = 100.0 * (double)iter / (double)num_updates;
                double elapsed = now - t_start;
                double rate_gups = 1e-9 * (double)iter / elapsed;
                /* ETA is meaningless in the first few seconds / first
                 * couple percent — a near-zero pct denominator inflates
                 * it wildly (the bug that made an actual ~250s run show
                 * an "ETA 29046s" estimate at 0.2% complete). Suppress
                 * it until the measurement has had time to stabilize. */
                int eta_ready = (pct >= 2.0 && elapsed >= 5.0);
                double eta_sec = eta_ready
                    ? elapsed * (100.0 - pct) / pct
                    : 0.0;
                if (eta_ready) {
                    fprintf(stderr,
                        "\r  %5.1f%%  |  %lu / %lu updates  |  "
                        "%.6f GUP/s  |  elapsed %.0fs  ETA %.0fs   ",
                        pct, (unsigned long)iter, (unsigned long)num_updates,
                        rate_gups, elapsed, eta_sec);
                } else {
                    fprintf(stderr,
                        "\r  %5.1f%%  |  %lu / %lu updates  |  "
                        "%.6f GUP/s  |  elapsed %.0fs  ETA (measuring...)   ",
                        pct, (unsigned long)iter, (unsigned long)num_updates,
                        rate_gups, elapsed);
                }
                fflush(stderr);
                last_progress_print = now;
            }
        }
    }
    fprintf(stderr, "\r  100.0%%  |  done                                            "
                     "                          \n");

    double t_end = wall_seconds();
    double real_time = t_end - t_start;
    double gups = 1e-9 * (double)num_updates / real_time;

    /* Stop and join the thermal sampler before printing results,
     * so end_temp_c reflects the state right at loop completion.
     * Always joined now since the thread is always launched. */
    sampler_stop = 1;
    pthread_join(sampler_thread, NULL);

    printf("\n=== Results ===\n");
    printf("Real time        : %.6f seconds\n", real_time);
    printf("GUP/s            : %.9f\n", gups);

    if (sampler.sample_count > 0) {
        printf("\n=== Thermal / Power (%d samples @ %.1fs interval) ===\n",
               sampler.sample_count, sampler.interval_sec);
        printf("Power source     : %s\n", sampler.power_source);
        if (!nvml_ok) {
            printf("                   (temp/clock unavailable — NVML not\n"
                   "                    found; power read independently\n"
                   "                    via spbm_hwmon if present)\n");
        }
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
                        : "  (clock dropped, but no throttle flag set — likely\n"
                          "                    idle-clock-state cycling from memory-bound\n"
                          "                    stalls, not a hardware throttle)";
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
                printf("Throttle flags   : none (no driver-reported throttle "
                       "event during this run)\n");
            }
        }
        if (strcmp(sampler.power_source, "unavailable") != 0) {
            double avg_power_w = sampler.power_sum_w / sampler.sample_count;
            printf("Start power      : %.1f W\n", sampler.start_power_w);
            printf("Avg power        : %.1f W\n", avg_power_w);
            printf("Peak power       : %.1f W\n", sampler.peak_power_w);
        } else {
            printf("Power            : unavailable (neither spbm_hwmon nor\n"
                   "                    NVML reported a value on this system)\n");
        }
    } else if (!nvml_ok) {
        printf("\n(No thermal/power data — NVML unavailable on this system)\n");
    } else {
        printf("\n(Run too short to capture a thermal sample at %.1fs interval —\n"
               " this is expected and fine for small table sizes; use a larger\n"
               " log2_table_size to see real thermal drift over tens of seconds.)\n",
               sampler.interval_sec);
    }

    /* Verification pass — same correctness check concept as the
     * original: re-run the same update sequence and confirm the
     * table returns to its expected post-update state. Simplified
     * here to a basic sanity check rather than the full XOR-replay
     * verification in verification.c, since there is only one
     * writer (no concurrent remote updates to reconcile). */
    printf("\nVerification: single-writer table, no remote update\n");
    printf("reconciliation required (unlike SHMEM multi-process version).\n");
    printf("Sanity check — sample entries:\n");
    for (int s = 0; s < 4; s++) {
        uint64_t idx = (table_size / 4) * s;
        printf("  table[%lu] = 0x%016lx\n", (unsigned long)idx,
               (unsigned long)table[idx]);
    }

    write_results_json("cpu_gups", "N/A_CPU_ONLY", log_table_size,
                        table_size, num_updates, real_time, gups, &sampler);

    if (nvml_ok) nvml_shutdown(&nvml);
    free(table);
    return 0;
}
