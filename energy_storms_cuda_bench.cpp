/*
 * energy_storms_cuda_bench.cpp
 *
 * Monte Carlo timing benchmark for the CUDA energy-storms kernel.
 *
 * Runs the core() function N times (default 30, override with -r N)
 * and reports: mean, std-dev, median, min, max, and a 95 % confidence
 * interval on the mean.  The first run is always treated as a warm-up
 * and excluded from the statistics.
 *
 * Usage:
 *   ./energy_storms_cuda_bench [-r RUNS] <size> <storm_file> [...]
 *
 * Example:
 *   ./energy_storms_cuda_bench -r 50 20000 \
 *       test_files/test_02_a30k_p20k_w1 \
 *       test_files/test_02_a30k_p20k_w2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <algorithm>   /* std::sort */
#include <cuda.h>
#include <cuda_runtime.h>
#include "energy_storms.h"

/* Declared in energy_storms_cuda_core.o */
void core(int layer_size, int num_storms, Storm *storms,
          float *maximum, int *positions);

/* ---- wall-clock timer (seconds) ---------------------------------------- */
static double wtime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/* ---- simple statistics -------------------------------------------------- */
static void compute_stats(double *samples, int n,
                          double *out_mean, double *out_std,
                          double *out_median, double *out_min, double *out_max,
                          double *out_ci95) {
    /* sort for median / min / max */
    double *sorted = (double *)malloc(sizeof(double) * n);
    memcpy(sorted, samples, sizeof(double) * n);
    std::sort(sorted, sorted + n);

    *out_min = sorted[0];
    *out_max = sorted[n - 1];
    *out_median = (n % 2 == 0)
                  ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0
                  : sorted[n/2];

    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += samples[i];
    *out_mean = sum / n;

    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = samples[i] - *out_mean;
        var += d * d;
    }
    *out_std = (n > 1) ? sqrt(var / (n - 1)) : 0.0;

    /* 95 % CI using t-distribution approximation:
       t_{0.975, n-1} ≈ 1.96 for n>=30, otherwise use a conservative 2.045
       (exact for df=29). For very small n we fall back to 2.776 (df=4). */
    double t;
    if      (n >= 30) t = 1.960;
    else if (n >= 10) t = 2.045;
    else              t = 2.776;
    *out_ci95 = t * (*out_std) / sqrt((double)n);

    free(sorted);
}

/* ---- helper: (re-)initialise output arrays before each run -------------- */
static void reset_outputs(float *maximum, int *positions, int num_storms) {
    for (int i = 0; i < num_storms; i++) {
        maximum[i]   = 0.0f;
        positions[i] = 0;
    }
}

/* ======================================================================== */
int main(int argc, char *argv[]) {

    /* ---- parse optional -r flag ---------------------------------------- */
    int runs = 30;   /* default number of measured runs (+ 1 warm-up) */
    int arg_offset = 1;

    if (argc > 2 && strcmp(argv[1], "-r") == 0) {
        runs = atoi(argv[2]);
        if (runs < 2) {
            fprintf(stderr, "Error: -r requires at least 2 runs\n");
            return EXIT_FAILURE;
        }
        arg_offset = 3;
    }

    if (argc - arg_offset < 2) {
        fprintf(stderr,
            "Usage: %s [-r RUNS] <size> <storm_file> [...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int layer_size  = atoi(argv[arg_offset]);
    int num_storms  = argc - arg_offset - 1;
    Storm *storms   = (Storm *)malloc(sizeof(Storm) * num_storms);

    for (int i = 0; i < num_storms; i++)
        storms[i] = read_storm_file(argv[arg_offset + 1 + i]);

    float *maximum   = (float *)malloc(sizeof(float) * num_storms);
    int   *positions = (int   *)malloc(sizeof(int)   * num_storms);

    /* ---- warm GPU up (run 0, result discarded) -------------------------- */
    printf("Warming up GPU...\n");
    cudaSetDevice(0);
    cudaDeviceSynchronize();

    reset_outputs(maximum, positions, num_storms);
    cudaDeviceSynchronize();
    double t0 = wtime();
    core(layer_size, num_storms, storms, maximum, positions);
    cudaDeviceSynchronize();
    double warmup_time = wtime() - t0;
    printf("Warm-up run: %.6f s  (excluded from stats)\n\n", warmup_time);

    /* Save the reference result from the warm-up run for consistency check */
    float *ref_max = (float *)malloc(sizeof(float) * num_storms);
    int   *ref_pos = (int   *)malloc(sizeof(int)   * num_storms);
    memcpy(ref_max, maximum,   sizeof(float) * num_storms);
    memcpy(ref_pos, positions, sizeof(int)   * num_storms);

    /* ---- measured runs ------------------------------------------------- */
    double *samples = (double *)malloc(sizeof(double) * runs);
    int     mismatches = 0;

    printf("Running %d measured iterations...\n", runs);
    for (int r = 0; r < runs; r++) {
        reset_outputs(maximum, positions, num_storms);
        cudaDeviceSynchronize();

        double t_start = wtime();
        core(layer_size, num_storms, storms, maximum, positions);
        cudaDeviceSynchronize();
        samples[r] = wtime() - t_start;

        /* Consistency check: results must match the warm-up reference */
        for (int s = 0; s < num_storms; s++) {
            if (positions[s] != ref_pos[s] ||
                fabsf(maximum[s] - ref_max[s]) > 1e-3f * ref_max[s]) {
                mismatches++;
                fprintf(stderr,
                    "  [run %d / storm %d] result mismatch: "
                    "pos %d vs %d, max %.6f vs %.6f\n",
                    r, s, positions[s], ref_pos[s],
                    maximum[s], ref_max[s]);
            }
        }

        printf("  Run %3d: %.6f s\n", r + 1, samples[r]);
    }

    /* ---- statistics ---------------------------------------------------- */
    double mean, std, median, mn, mx, ci95;
    compute_stats(samples, runs, &mean, &std, &median, &mn, &mx, &ci95);

    printf("\n");
    printf("========================================\n");
    printf("  Monte Carlo timing summary (%d runs)\n", runs);
    printf("========================================\n");
    printf("  Mean     : %.6f s\n", mean);
    printf("  Std dev  : %.6f s\n", std);
    printf("  Median   : %.6f s\n", median);
    printf("  Min      : %.6f s\n", mn);
    printf("  Max      : %.6f s\n", mx);
    printf("  95%% CI   : [ %.6f , %.6f ] s\n", mean - ci95, mean + ci95);
    printf("  CV       : %.2f %%\n", 100.0 * std / mean);
    printf("========================================\n");

    if (mismatches > 0)
        printf("\nWARNING: %d result mismatch(es) detected across runs!\n",
               mismatches);
    else
        printf("\nAll runs produced consistent results.\n");

    /* ---- print reference result ---------------------------------------- */
    printf("\nResult (from warm-up run):");
    for (int s = 0; s < num_storms; s++)
        printf(" %d %f", ref_pos[s], ref_max[s]);
    printf("\n");

    /* ---- clean up ------------------------------------------------------- */
    for (int i = 0; i < num_storms; i++) free(storms[i].posval);
    free(storms);
    free(maximum); free(positions);
    free(ref_max); free(ref_pos);
    free(samples);
    return 0;
}
