/* Private-kernel A/B probe, built separately from the installed extension:
 * gcc -O3 -fPIC -shared -Iwaveforms benchmarks/simd_kernel_probe.c -lm -o probe.so
 * This is diagnostic code, not part of the library's public ABI.
 */
#include "_cwaveform.c"

int wf_probe_quantize(const double *input, size_t count, int bits,
                       double full_scale, void *output, int mode, size_t loops) {
    size_t repeat;
    int status = 0;
    for (repeat = 0; repeat < loops; ++repeat) {
        if (mode == 1) {
            size_t index;
            for (index = 0; index < count; ++index) {
                if (!isfinite(input[index])) return -3;
                if (bits == 16) ((int16_t *)output)[index] = wf_quantize16(input[index], full_scale);
                else ((int32_t *)output)[index] = wf_quantize32(input[index], full_scale);
            }
        }
#if defined(WF_HAVE_X86_SIMD) && !defined(WF_DISABLE_X86_SIMD)
        else if (mode == 2 && wf_cpu_supports_avx2())
            status = wf_quantize_array_avx2(input, count, bits, full_scale, output);
#if !defined(WF_DISABLE_AVX512) && !defined(_MSC_VER)
        else if (mode == 3 && wf_cpu_supports_avx512())
            status = wf_quantize_array_avx512(input, count, bits, full_scale, output);
#endif
#endif
        else if (mode == 0) status = wf_quantize_array(input, count, bits, full_scale, output);
        else return -99;
        if (status != 0) return status;
    }
    return status;
}

int wf_probe_nonlinear(const cwaveform_nonlinear_map *map, const double *input,
                        size_t count, int bits, double full_scale, void *output,
                        int mode, size_t loops) {
    size_t repeat;
    int status = 0;
    for (repeat = 0; repeat < loops; ++repeat) {
        if (mode == 1)
            status = wf_nonlinear_apply_scalar(map, input, count, bits, full_scale, output);
#if defined(WF_HAVE_X86_SIMD) && !defined(WF_DISABLE_X86_SIMD) && !defined(WF_DISABLE_NONLINEAR_SIMD)
        else if (mode == 2 && wf_cpu_supports_avx2())
            status = wf_nonlinear_apply_avx2(map, input, count, bits, full_scale, output);
#if !defined(WF_DISABLE_AVX512) && !defined(_MSC_VER)
        else if (mode == 3 && wf_cpu_supports_avx512())
            status = wf_nonlinear_apply_avx512(map, input, count, bits, full_scale, output);
#endif
#endif
        else if (mode == 0) status = cwaveform_nonlinear_map_apply(map, input, count, bits, full_scale, output);
        else return -99;
        if (status != 0) return status;
    }
    return status;
}
