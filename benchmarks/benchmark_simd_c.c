#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "../waveforms/_cwaveform.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

static double now_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + 1e-9 * (double)value.tv_nsec;
#endif
}

static double best_evaluate(const cwaveform_wave *wave, const double *positions,
                            size_t count, double *output, int repeats) {
    double best = INFINITY;
    int repeat;
    for (repeat = -1; repeat < repeats; ++repeat) {
        double start = now_seconds();
        double elapsed;
        if (cwaveform_wave_evaluate(
                wave, positions, count, 0, 1.0, -INFINITY, INFINITY,
                output) != 0) abort();
        elapsed = now_seconds() - start;
        if (repeat >= 0 && elapsed < best) best = elapsed;
    }
    return best;
}

static double best_sample(const cwaveform_wave *wave, int64_t start_tick,
                          size_t count, int dtype, void *output, int repeats) {
    double best = INFINITY;
    int repeat;
    for (repeat = -1; repeat < repeats; ++repeat) {
        double start = now_seconds();
        double elapsed;
        if (cwaveform_wave_sample(
                wave, start_tick, count, 50, 1, 0, 1.0,
                -INFINITY, INFINITY, dtype, 1.0, output) != 0) abort();
        elapsed = now_seconds() - start;
        if (repeat >= 0 && elapsed < best) best = elapsed;
    }
    return best;
}

static double best_quantize(const double *values, size_t count, int dtype,
                            void *output, int repeats) {
    double best = INFINITY;
    int repeat;
    for (repeat = -1; repeat < repeats; ++repeat) {
        double start = now_seconds();
        double elapsed;
        if (cwaveform_quantize(values, count, dtype, 1.0, output) != 0) abort();
        elapsed = now_seconds() - start;
        if (repeat >= 0 && elapsed < best) best = elapsed;
    }
    return best;
}

int main(int argc, char **argv) {
    const size_t count = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10)
                                  : 1000000u;
    const int repeats = argc > 2 ? atoi(argv[2]) : 15;
    const double rate = 2400000000.0;
    const double duration = (double)count / rate;
    const double start = -duration / 2.0;
    const double pi = 3.14159265358979323846;
    cwaveform_wave *gaussian = cwaveform_wave_gaussian(duration * 1.8);
    cwaveform_wave *carrier = cwaveform_wave_cos(2.0 * pi * 100e6, 0.0);
    cwaveform_wave *pulse = cwaveform_wave_mul_affine(
        gaussian, 0, 0.8, carrier, 0, 1.0);
    cwaveform_wave *powered = cwaveform_wave_power(pulse, 2);
    double *positions = malloc(count * sizeof(*positions));
    double *values = malloc(count * sizeof(*values));
    double *samples = malloc(count * sizeof(*samples));
    int16_t *samples16 = malloc(count * sizeof(*samples16));
    int32_t *samples32 = malloc(count * sizeof(*samples32));
    int16_t *expected16 = malloc(count * sizeof(*expected16));
    int32_t *expected32 = malloc(count * sizeof(*expected32));
    size_t index;
    double evaluate_supported;
    double evaluate_power;
    double sample_float64;
    double sample_int16;
    double sample_int32;
    double quantize_int16;
    double quantize_int32;
    if (pulse == NULL || powered == NULL || positions == NULL || values == NULL
            || samples == NULL || samples16 == NULL || samples32 == NULL
            || expected16 == NULL || expected32 == NULL) abort();
    for (index = 0; index < count; ++index) {
        positions[index] = start + (double)index / rate;
        values[index] = 0.999 * sin(-100.0 + 200.0 * (double)index
                                   / (double)count);
    }
    evaluate_supported = best_evaluate(
        pulse, positions, count, samples, repeats);
    evaluate_power = best_evaluate(powered, positions, count, samples, repeats);
    sample_float64 = best_sample(
        pulse, llround(start * 120000000000.0), count,
        CWAVEFORM_FLOAT64, samples, repeats);
    sample_int16 = best_sample(
        pulse, llround(start * 120000000000.0), count,
        CWAVEFORM_INT16, samples16, repeats);
    sample_int32 = best_sample(
        pulse, llround(start * 120000000000.0), count,
        CWAVEFORM_INT32, samples32, repeats);
    quantize_int16 = best_quantize(
        values, count, CWAVEFORM_INT16, expected16, repeats);
    quantize_int32 = best_quantize(
        values, count, CWAVEFORM_INT32, expected32, repeats);
    if (cwaveform_wave_sample(
            pulse, llround(start * 120000000000.0), count, 50, 1, 0, 1.0,
            -INFINITY, INFINITY, CWAVEFORM_FLOAT64, 1.0, samples) != 0
            || cwaveform_quantize(
                samples, count, CWAVEFORM_INT16, 1.0, expected16) != 0
            || cwaveform_quantize(
                samples, count, CWAVEFORM_INT32, 1.0, expected32) != 0
            || memcmp(samples16, expected16, count * sizeof(*samples16)) != 0
            || memcmp(samples32, expected32, count * sizeof(*samples32)) != 0)
        abort();
    printf("{\n  \"count\": %zu,\n  \"milliseconds\": {\n", count);
    printf("    \"evaluate_supported\": %.6f,\n", 1e3 * evaluate_supported);
    printf("    \"evaluate_power\": %.6f,\n", 1e3 * evaluate_power);
    printf("    \"sample_float64\": %.6f,\n", 1e3 * sample_float64);
    printf("    \"sample_int16\": %.6f,\n", 1e3 * sample_int16);
    printf("    \"sample_int32\": %.6f,\n", 1e3 * sample_int32);
    printf("    \"quantize_int16\": %.6f,\n", 1e3 * quantize_int16);
    printf("    \"quantize_int32\": %.6f\n  }\n}\n", 1e3 * quantize_int32);
    free(positions); free(values); free(samples); free(samples16);
    free(samples32); free(expected16); free(expected32);
    cwaveform_wave_release(powered); cwaveform_wave_release(pulse);
    cwaveform_wave_release(carrier); cwaveform_wave_release(gaussian);
    return 0;
}
