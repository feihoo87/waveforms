#ifndef WAVEFORMS_WF_NATIVE_H
#define WAVEFORMS_WF_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(WF_NATIVE_BUILD)
#    define WF_API __declspec(dllexport)
#  elif defined(WF_NATIVE_USE_DLL)
#    define WF_API __declspec(dllimport)
#  else
#    define WF_API
#  endif
#else
#  define WF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stable C ABI for the WNF4/WNS4 cross-language waveform format.
 *
 * Serialized integers and floating-point values are little-endian. Runtime
 * handles own an immutable serialized block plus decoded sidecar data; the
 * serialized address remains stable for the lifetime of the handle.
 */

typedef struct wf_native_wave wf_native_wave;
typedef struct wf_native_stack wf_native_stack;

enum wf_native_dtype {
    WF_NATIVE_FLOAT64 = 0,
    WF_NATIVE_INT16 = 16,
    WF_NATIVE_INT32 = 32
};

WF_API uint64_t wf_native_ticks_per_second(void);

WF_API wf_native_wave *wf_native_wave_constant(double value);
WF_API wf_native_wave *wf_native_wave_gaussian(double width_seconds);
WF_API wf_native_wave *wf_native_wave_cos(double angular_frequency,
                                           double phase);
WF_API wf_native_wave *wf_native_wave_sin(double angular_frequency,
                                           double phase);
WF_API wf_native_wave *wf_native_wave_square(double width_seconds);

WF_API wf_native_wave *wf_native_wave_from_bytes(const uint8_t *data,
                                                  size_t size);
WF_API wf_native_wave *wf_native_wave_add_affine(
    const wf_native_wave *left, int64_t left_delay, double left_scale,
    const wf_native_wave *right, int64_t right_delay, double right_scale);
WF_API wf_native_wave *wf_native_wave_mul_affine(
    const wf_native_wave *left, int64_t left_delay, double left_scale,
    const wf_native_wave *right, int64_t right_delay, double right_scale);
WF_API wf_native_wave *wf_native_wave_materialize(
    const wf_native_wave *wave, int64_t delay, double scale);

WF_API void wf_native_wave_retain(wf_native_wave *wave);
WF_API void wf_native_wave_release(wf_native_wave *wave);
WF_API const uint8_t *wf_native_wave_bytes(const wf_native_wave *wave,
                                            size_t *size);
WF_API uint64_t wf_native_wave_hash(const wf_native_wave *wave);
WF_API int wf_native_wave_equal(const wf_native_wave *left,
                                const wf_native_wave *right);
WF_API int64_t wf_native_wave_lower_tick(const wf_native_wave *wave);
WF_API int64_t wf_native_wave_upper_tick(const wf_native_wave *wave);
WF_API uint32_t wf_native_wave_node_count(const wf_native_wave *wave);

WF_API int wf_native_wave_evaluate(
    const wf_native_wave *wave, const double *positions, size_t count,
    int64_t delay_tick, double scale, double lower_clip, double upper_clip,
    double *output);
WF_API int wf_native_wave_sample(
    const wf_native_wave *wave, int64_t start_tick, size_t count,
    int64_t step_numerator, int64_t step_denominator, int64_t delay_tick,
    double scale, double lower_clip, double upper_clip, int dtype,
    double full_scale, void *output);

WF_API wf_native_stack *wf_native_stack_create(
    wf_native_wave *const *templates, const uint32_t *template_ids,
    const int64_t *delay_ticks, const double *scales,
    size_t template_count, size_t event_count);
WF_API wf_native_stack *wf_native_stack_from_bytes(const uint8_t *data,
                                                    size_t size);
WF_API wf_native_stack *wf_native_stack_materialize(
    const wf_native_stack *stack, int64_t global_shift, double offset);
WF_API void wf_native_stack_retain(wf_native_stack *stack);
WF_API void wf_native_stack_release(wf_native_stack *stack);
WF_API const uint8_t *wf_native_stack_bytes(const wf_native_stack *stack,
                                             size_t *size);
WF_API uint64_t wf_native_stack_hash(const wf_native_stack *stack);
WF_API size_t wf_native_stack_event_count(const wf_native_stack *stack);
WF_API size_t wf_native_stack_template_count(const wf_native_stack *stack);
WF_API int wf_native_stack_evaluate(
    const wf_native_stack *stack, const double *positions, size_t count,
    int64_t global_shift, double offset, double *output);
WF_API int wf_native_stack_sample(
    const wf_native_stack *stack, int64_t start_tick, size_t count,
    int64_t step_numerator, int64_t step_denominator, int64_t global_shift,
    double offset, int dtype, double full_scale, void *output);
WF_API wf_native_wave *wf_native_stack_simplify(
    const wf_native_stack *stack, int64_t global_shift, double offset);

WF_API const char *wf_native_format_description(void);

#ifdef __cplusplus
}
#endif

#endif
