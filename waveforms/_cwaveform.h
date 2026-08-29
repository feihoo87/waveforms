#ifndef WAVEFORMS_CWAVEFORM_H
#define WAVEFORMS_CWAVEFORM_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(CWAVEFORM_BUILD)
#    define CWAVEFORM_API __declspec(dllexport)
#  elif defined(CWAVEFORM_USE_DLL)
#    define CWAVEFORM_API __declspec(dllimport)
#  else
#    define CWAVEFORM_API
#  endif
#else
#  define CWAVEFORM_API __attribute__((visibility("default")))
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

typedef struct cwaveform_wave cwaveform_wave;
typedef struct cwaveform_stack cwaveform_stack;
typedef struct cwaveform_sample_plan cwaveform_sample_plan;

/* Stable language-neutral builtin identifiers.  These intentionally match
 * the historic Python/Cython opcode values so other language bindings can
 * construct the same blocks without translating an enum. */
enum cwaveform_builtin {
    CWAVEFORM_LINEAR = 1,
    CWAVEFORM_GAUSSIAN = 2,
    CWAVEFORM_ERF = 3,
    CWAVEFORM_COS = 4,
    CWAVEFORM_SINC = 5,
    CWAVEFORM_EXP = 6,
    CWAVEFORM_INTERP = 7,
    CWAVEFORM_LINEAR_CHIRP = 8,
    CWAVEFORM_EXPONENTIAL_CHIRP = 9,
    CWAVEFORM_HYPERBOLIC_CHIRP = 10,
    CWAVEFORM_COSH = 11,
    CWAVEFORM_SINH = 12,
    CWAVEFORM_DRAG = 13,
    CWAVEFORM_MOLLIFIER = 14,
    CWAVEFORM_D_GAUSSIAN = 15,
    CWAVEFORM_DRAG_SIN = 16,
    CWAVEFORM_DRAG_SINX = 17
};

enum cwaveform_dtype {
    CWAVEFORM_FLOAT64 = 0,
    CWAVEFORM_INT16 = 16,
    CWAVEFORM_INT32 = 32
};

CWAVEFORM_API uint64_t cwaveform_ticks_per_second(void);
CWAVEFORM_API int cwaveform_set_ticks_per_second(uint64_t ticks_per_second);

CWAVEFORM_API cwaveform_wave *cwaveform_wave_constant(double value);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_gaussian(double width_seconds);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_cos(double angular_frequency,
                                           double phase);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_sin(double angular_frequency,
                                           double phase);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_square(double width_seconds);
/* Generic builtin parameters use seconds for time values.  The C core
 * quantizes them to the process clock before serializing. */
CWAVEFORM_API cwaveform_wave *cwaveform_wave_builtin(
    int builtin, const double *parameters, size_t parameter_count,
    int64_t shift_tick);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_window(
    const cwaveform_wave *wave, int64_t lower_tick, int64_t upper_tick);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_power(
    const cwaveform_wave *wave, int power);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_derivative(
    const cwaveform_wave *wave, unsigned order);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_filter(
    const cwaveform_wave *wave, double low, double high, double epsilon);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_simplify(
    const cwaveform_wave *wave, double epsilon);

CWAVEFORM_API cwaveform_wave *cwaveform_wave_from_bytes(const uint8_t *data,
                                                  size_t size);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_add_affine(
    const cwaveform_wave *left, int64_t left_delay, double left_scale,
    const cwaveform_wave *right, int64_t right_delay, double right_scale);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_mul_affine(
    const cwaveform_wave *left, int64_t left_delay, double left_scale,
    const cwaveform_wave *right, int64_t right_delay, double right_scale);
CWAVEFORM_API cwaveform_wave *cwaveform_wave_materialize(
    const cwaveform_wave *wave, int64_t delay, double scale);

CWAVEFORM_API void cwaveform_wave_retain(cwaveform_wave *wave);
CWAVEFORM_API void cwaveform_wave_release(cwaveform_wave *wave);
CWAVEFORM_API const uint8_t *cwaveform_wave_bytes(const cwaveform_wave *wave,
                                            size_t *size);
CWAVEFORM_API uint64_t cwaveform_wave_hash(const cwaveform_wave *wave);
CWAVEFORM_API int cwaveform_wave_equal(const cwaveform_wave *left,
                                const cwaveform_wave *right);
CWAVEFORM_API int64_t cwaveform_wave_lower_tick(const cwaveform_wave *wave);
CWAVEFORM_API int64_t cwaveform_wave_upper_tick(const cwaveform_wave *wave);
CWAVEFORM_API uint32_t cwaveform_wave_node_count(const cwaveform_wave *wave);

CWAVEFORM_API int cwaveform_wave_evaluate(
    const cwaveform_wave *wave, const double *positions, size_t count,
    int64_t delay_tick, double scale, double lower_clip, double upper_clip,
    double *output);
CWAVEFORM_API int cwaveform_wave_sample(
    const cwaveform_wave *wave, int64_t start_tick, size_t count,
    int64_t step_numerator, int64_t step_denominator, int64_t delay_tick,
    double scale, double lower_clip, double upper_clip, int dtype,
    double full_scale, void *output);
CWAVEFORM_API int cwaveform_quantize(
    const double *values, size_t count, int dtype,
    double full_scale, void *output);

CWAVEFORM_API cwaveform_stack *cwaveform_stack_create(
    cwaveform_wave *const *templates, const uint32_t *template_ids,
    const int64_t *delay_ticks, const double *scales,
    size_t template_count, size_t event_count);
CWAVEFORM_API cwaveform_stack *cwaveform_stack_from_bytes(const uint8_t *data,
                                                    size_t size);
CWAVEFORM_API cwaveform_stack *cwaveform_stack_materialize(
    const cwaveform_stack *stack, int64_t global_shift, double offset);
CWAVEFORM_API cwaveform_stack *cwaveform_stack_combine(
    const cwaveform_stack *left, int64_t left_shift,
    const cwaveform_stack *right, int64_t right_shift);
CWAVEFORM_API cwaveform_stack *cwaveform_stack_append(
    const cwaveform_stack *stack, int64_t global_shift,
    const cwaveform_wave *wave, int64_t wave_delay, double wave_scale);
CWAVEFORM_API cwaveform_stack *cwaveform_stack_scale(
    const cwaveform_stack *stack, double scale);
CWAVEFORM_API void cwaveform_stack_retain(cwaveform_stack *stack);
CWAVEFORM_API void cwaveform_stack_release(cwaveform_stack *stack);
CWAVEFORM_API const uint8_t *cwaveform_stack_bytes(const cwaveform_stack *stack,
                                             size_t *size);
CWAVEFORM_API uint64_t cwaveform_stack_hash(const cwaveform_stack *stack);
CWAVEFORM_API size_t cwaveform_stack_event_count(const cwaveform_stack *stack);
CWAVEFORM_API size_t cwaveform_stack_template_count(const cwaveform_stack *stack);
CWAVEFORM_API int cwaveform_stack_evaluate(
    const cwaveform_stack *stack, const double *positions, size_t count,
    int64_t global_shift, double offset, double *output);
CWAVEFORM_API int cwaveform_stack_sample(
    const cwaveform_stack *stack, int64_t start_tick, size_t count,
    int64_t step_numerator, int64_t step_denominator, int64_t global_shift,
    double offset, int dtype, double full_scale, void *output);
CWAVEFORM_API cwaveform_sample_plan *cwaveform_sample_plan_create(
    const cwaveform_stack *stack, int64_t start_tick, size_t count,
    int64_t step_numerator, int64_t step_denominator,
    int64_t global_shift);
CWAVEFORM_API void cwaveform_sample_plan_retain(cwaveform_sample_plan *plan);
CWAVEFORM_API void cwaveform_sample_plan_release(cwaveform_sample_plan *plan);
CWAVEFORM_API size_t cwaveform_sample_plan_count(
    const cwaveform_sample_plan *plan);
CWAVEFORM_API size_t cwaveform_sample_plan_group_count(
    const cwaveform_sample_plan *plan);
CWAVEFORM_API int cwaveform_sample_plan_non_overlapping(
    const cwaveform_sample_plan *plan);
CWAVEFORM_API int cwaveform_sample_plan_sample(
    const cwaveform_sample_plan *plan, double offset, int dtype,
    double full_scale, void *output);
CWAVEFORM_API cwaveform_wave *cwaveform_stack_simplify(
    const cwaveform_stack *stack, int64_t global_shift, double offset);

CWAVEFORM_API const char *cwaveform_format_description(void);

#ifdef __cplusplus
}
#endif

#endif
