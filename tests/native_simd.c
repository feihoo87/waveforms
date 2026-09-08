/* Force every available native kernel, including non-default AVX2 paths.
 * gcc -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
 *   tests/native_simd.c -lm -o /tmp/wf-simd-test
 * macOS: replace -lm with -framework Accelerate.
 */
#include <assert.h>
#include <stdio.h>
#include "../waveforms/_cwaveform.c"

typedef int (*quantizer)(const double *, size_t, int, double, void *);
typedef int (*mapper)(const cwaveform_nonlinear_map *, const double *,
                      size_t, int, double, void *);

static void check_quantizer(quantizer apply) {
    double padded[516], *values = padded + 1;
    int16_t out16[516];
    int32_t out32[516];
    int bits;
    size_t count, index, shift;
    for (bits = 16; bits <= 32; bits += 16) {
        double limit = bits == 16 ? 32768.0 : 2147483648.0;
        double cases[] = {0., -0., .5, nextafter(.5, 0.), nextafter(.5, 1.),
                         1.5, nextafter(1.5, 0.), 127.5, limit - 1.5,
                         limit - 1., limit, 2 * limit, DBL_MAX};
        size_t case_count = sizeof(cases) / sizeof(cases[0]);
        for (count = 0; count <= 513; ++count) {
            for (shift = 0; shift < 8; ++shift) {
                for (index = 0; index < count; ++index) {
                    values[index] = cases[(index + shift) % case_count] / limit;
                    if ((index / case_count) % 2) values[index] = -values[index];
                }
                out16[0] = out16[count + 1] = 123;
                out32[0] = out32[count + 1] = 123;
                assert(apply(values, count, bits, 1.,
                             bits == 16 ? (void *)(out16 + 1) : (void *)(out32 + 1)) == 0);
                for (index = 0; index < count; ++index) {
                    if (bits == 16)
                        assert(out16[index + 1] == wf_quantize16(values[index], 1.));
                    else
                        assert(out32[index + 1] == wf_quantize32(values[index], 1.));
                }
                assert(out16[0] == 123 && out16[count + 1] == 123);
                assert(out32[0] == 123 && out32[count + 1] == 123);
            }
        }
        for (index = 0; index < 65; ++index) {
            memset(values, 0, 65 * sizeof(double));
            values[index] = index % 3 == 0 ? NAN : index % 3 == 1 ? INFINITY : -INFINITY;
            assert(apply(values, 65, bits, 1.,
                         bits == 16 ? (void *)out16 : (void *)out32) == -3);
        }
    }
}

static void check_mapper(mapper apply) {
    double input[66], output[66], expected[66], in_place[66];
    int16_t output16[66], expected16[66];
    int32_t output32[66], expected32[66];
    double coefficients[1024];
    int method, storage, clip;
    size_t points, index, count;
    for (index = 0; index < 1024; ++index)
        coefficients[index] = .17 * sin((double)index * .731);
    for (method = 1; method <= 2; ++method)
    for (storage = 32; storage <= 64; storage += 32)
    for (clip = 0; clip <= 1; ++clip)
    for (points = 2; points <= 257; points = points == 2 ? 3 : points == 3 ? 257 : 258) {
        cwaveform_nonlinear_map *map = cwaveform_nonlinear_map_create(
            method, storage, clip, -1., 1., .25, .07, coefficients, points);
        assert(map != NULL);
        for (count = 0; count <= 65; ++count) {
            for (index = 0; index < count; ++index) {
                input[index] = sin(index * .713) * (clip ? 1.5 : .99) - .25;
                if (index % 4 == 0) input[index] = -1.25;
                if (index % 4 == 1) input[index] = .75;
                in_place[index] = input[index];
            }
            output[count] = 123.;
            assert(wf_nonlinear_apply_scalar(map, input, count, 0, 1., expected) == 0);
            assert(apply(map, input, count, 0, 1., output) == 0);
            assert(apply(map, in_place, count, 0, 1., in_place) == 0);
            for (index = 0; index < count; ++index) {
                assert(fabs(output[index] - expected[index]) <= 2e-15);
                assert(output[index] == in_place[index]);
            }
            assert(output[count] == 123.);
            assert(wf_nonlinear_apply_scalar(map, input, count, 16, 1., expected16) == 0);
            assert(apply(map, input, count, 16, 1., output16) == 0);
            assert(memcmp(output16, expected16, count * sizeof(int16_t)) == 0);
            assert(wf_nonlinear_apply_scalar(map, input, count, 32, 1., expected32) == 0);
            assert(apply(map, input, count, 32, 1., output32) == 0);
            assert(memcmp(output32, expected32, count * sizeof(int32_t)) == 0);
        }
        for (index = 0; index < 65; ++index) {
            memset(input, 0, sizeof(input));
            input[index] = index % 3 == 0 ? NAN : index % 3 == 1 ? INFINITY : -INFINITY;
            assert(apply(map, input, 65, 0, 1., output) == -3);
            if (!clip) {
                input[index] = 2.;
                assert(apply(map, input, 65, 0, 1., output) == -3);
            }
        }
        cwaveform_nonlinear_map_release(map);
    }
}

int main(void) {
    check_quantizer(cwaveform_quantize);
    check_mapper(cwaveform_nonlinear_map_apply);
#if defined(WF_HAVE_ARM64_NEON)
    check_quantizer(wf_quantize_array_neon);
#if !defined(WF_DISABLE_NONLINEAR_SIMD)
    check_mapper(wf_nonlinear_apply_neon);
#endif
    puts("NEON checks passed");
#endif
#if defined(WF_HAVE_X86_SIMD) && !defined(WF_DISABLE_X86_SIMD)
    if (wf_cpu_supports_avx2()) {
        check_quantizer(wf_quantize_array_avx2);
#if !defined(WF_DISABLE_NONLINEAR_SIMD)
        check_mapper(wf_nonlinear_apply_avx2);
#endif
        puts("AVX2 checks passed");
    }
#if !defined(WF_DISABLE_AVX512) && !defined(_MSC_VER)
    if (wf_cpu_supports_avx512()) {
        check_quantizer(wf_quantize_array_avx512);
#if !defined(WF_DISABLE_NONLINEAR_SIMD)
        check_mapper(wf_nonlinear_apply_avx512);
#endif
        puts("AVX-512 checks passed");
    }
#endif
#endif
    puts("native SIMD checks passed");
    return 0;
}
