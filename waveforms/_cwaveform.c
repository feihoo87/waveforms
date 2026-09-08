#define CWAVEFORM_BUILD 1
#include "_cwaveform.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define WF_HAVE_ARM64_NEON 1
#endif

#if ((defined(__x86_64__) || defined(__i386__)) \
        && (defined(__GNUC__) || defined(__clang__))) \
        || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <immintrin.h>
#define WF_HAVE_X86_SIMD 1
#endif

#if defined(WF_HAVE_X86_SIMD) \
        && (defined(__GNUC__) || defined(__clang__))
#define WF_TARGET_AVX2 __attribute__((target("avx2")))
#define WF_TARGET_AVX512 \
    __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
#else
#define WF_TARGET_AVX2
#define WF_TARGET_AVX512
#endif

#if defined(_MSC_VER)
#define WF_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define WF_NOINLINE __attribute__((noinline))
#else
#define WF_NOINLINE
#endif

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#define WF_DEFAULT_TICKS_PER_SECOND UINT64_C(120000000000)
#define WF_WAVE_HEADER_SIZE 24u
#define WF_NODE_SIZE 28u
#define WF_STACK_HEADER_SIZE 16u
#define WF_NONLINEAR_HEADER_SIZE 48u
#define WF_VERSION 2u

static uint64_t wf_ticks_per_second = WF_DEFAULT_TICKS_PER_SECOND;

enum wf_op {
    WF_OP_CONSTANT = 1,
    WF_OP_GAUSSIAN = 2,
    WF_OP_COS = 3,
    WF_OP_SIN = 4,
    WF_OP_SQUARE = 5,
    WF_OP_LINEAR = 6,
    WF_OP_ERF = 7,
    WF_OP_SINC = 8,
    WF_OP_EXP = 9,
    WF_OP_INTERP = 10,
    WF_OP_LINEAR_CHIRP = 11,
    WF_OP_EXPONENTIAL_CHIRP = 12,
    WF_OP_HYPERBOLIC_CHIRP = 13,
    WF_OP_COSH = 14,
    WF_OP_SINH = 15,
    WF_OP_DRAG = 16,
    WF_OP_MOLLIFIER = 17,
    WF_OP_D_GAUSSIAN = 18,
    WF_OP_DRAG_SIN = 19,
    WF_OP_DRAG_SINX = 20,
    WF_OP_ADD = 32,
    WF_OP_MUL = 33,
    WF_OP_SCALE = 34,
    WF_OP_POWER = 35,
    WF_OP_WINDOW = 36
};

typedef struct wf_node {
    uint8_t op;
    uint8_t flags;
    uint16_t parameter_count;
    uint32_t left;
    uint32_t right;
    int64_t shift;
    double p0;
    uint32_t parameter_offset;
    double p1;
    double p2;
    int64_t lower;
    int64_t upper;
} wf_node;

typedef struct wf_drag_sin_kernel {
    size_t order;
    size_t power;
    size_t basis_count;
    double angular;
    double normalization;
    double *transform;
    double *derivatives;
    double *left_polynomial;
    double *right_polynomial;
} wf_drag_sin_kernel;

struct cwaveform_wave {
    uint32_t references;
    uint32_t node_count;
    uint32_t root;
    uint64_t hash;
    size_t data_size;
    uint8_t *data;
    wf_node *nodes;
    double *parameters;
    size_t parameter_count;
    wf_drag_sin_kernel **drag_sin_kernels;
};

struct cwaveform_stack {
    uint32_t references;
    size_t template_count;
    size_t event_count;
    cwaveform_wave **templates;
    uint32_t *template_ids;
    int64_t *delays;
    double *scales;
    int64_t lower_tick;
    int64_t upper_tick;
    uint64_t hash;
    size_t data_size;
    uint8_t *data;
};

typedef struct wf_plan_scale_group {
    double scale;
    double *samples;
    size_t destination_count;
    size_t destination_capacity;
    int64_t *destinations;
} wf_plan_scale_group;

typedef struct wf_plan_group {
    uint32_t template_id;
    uint64_t phase;
    int64_t first_tick;
    size_t sample_count;
    double *samples;
    size_t placement_count;
    size_t placement_capacity;
    int64_t *destinations;
    double *scales;
    size_t scale_group_count;
    size_t scale_group_capacity;
    wf_plan_scale_group *scale_groups;
    int grouped_scales;
} wf_plan_group;

struct cwaveform_sample_plan {
    uint32_t references;
    size_t count;
    size_t group_count;
    size_t group_capacity;
    wf_plan_group *groups;
    int non_overlapping;
};

struct cwaveform_nonlinear_map {
    uint32_t references;
    uint8_t method;
    uint8_t storage;
    uint8_t extrapolation;
    size_t point_count;
    size_t coefficient_count;
    double x_min;
    double x_max;
    double input_offset;
    double output_offset;
    double coordinate_scale;
    double *coefficients;
    uint64_t hash;
    size_t data_size;
    uint8_t *data;
};

static double wf_node_parameter(const cwaveform_wave *wave,
                                const wf_node *node, size_t index);
static int wf_wave_prepare_drag_sin_kernels(cwaveform_wave *wave);
static void wf_wave_clear_drag_sin_kernels(cwaveform_wave *wave);

static double wf_hypot(double x, double y) {
    double high = fabs(x);
    double low = fabs(y);
    double ratio;
    if (isinf(high) || isinf(low)) return INFINITY;
    if (isnan(high) || isnan(low)) return NAN;
    if (low > high) {
        double temporary = high;
        high = low;
        low = temporary;
    }
    if (high == 0.0) return 0.0;
    ratio = low / high;
    return high * sqrt(1.0 + ratio * ratio);
}

static void wf_put_u16(uint8_t *p, uint16_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void wf_put_u32(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static void wf_put_u64(uint8_t *p, uint64_t value) {
    unsigned index;
    for (index = 0; index < 8; ++index) {
        p[index] = (uint8_t)(value >> (index * 8));
    }
}

static void wf_put_i64(uint8_t *p, int64_t value) {
    wf_put_u64(p, (uint64_t)value);
}

static void wf_put_f64(uint8_t *p, double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    wf_put_u64(p, bits);
}

static void wf_put_f32(uint8_t *p, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    wf_put_u32(p, bits);
}

static uint16_t wf_get_u16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t wf_get_u32(const uint8_t *p) {
    return ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
            | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static uint64_t wf_get_u64(const uint8_t *p) {
    uint64_t value = 0;
    unsigned index;
    for (index = 0; index < 8; ++index) {
        value |= (uint64_t)p[index] << (index * 8);
    }
    return value;
}

static int64_t wf_get_i64(const uint8_t *p) {
    return (int64_t)wf_get_u64(p);
}

static double wf_get_f64(const uint8_t *p) {
    uint64_t bits = wf_get_u64(p);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float wf_get_f32(const uint8_t *p) {
    uint32_t bits = wf_get_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t wf_hash_bytes(const uint8_t *data, size_t size) {
    uint64_t value = UINT64_C(1469598103934665603);
    size_t index;
    for (index = 0; index < size; ++index) {
        value ^= data[index];
        value *= UINT64_C(1099511628211);
    }
    return value;
}

static int64_t wf_seconds_to_tick(double value) {
    long double scaled = (long double)value * (long double)wf_ticks_per_second;
    if (scaled >= (long double)INT64_MAX) {
        return INT64_MAX;
    }
    if (scaled <= (long double)INT64_MIN) {
        return INT64_MIN;
    }
    return (int64_t)llroundl(scaled);
}

static int64_t wf_add_tick(int64_t value, int64_t shift) {
    if (value == INT64_MIN || value == INT64_MAX) {
        return value;
    }
    if (shift > 0 && value > INT64_MAX - shift) {
        return INT64_MAX;
    }
    if (shift < 0 && value < INT64_MIN - shift) {
        return INT64_MIN;
    }
    return value + shift;
}

static int wf_empty_support(int64_t lower, int64_t upper) {
    return lower >= upper;
}

static void wf_union_support(int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                             int64_t *lower, int64_t *upper) {
    if (wf_empty_support(a0, a1)) {
        *lower = b0;
        *upper = b1;
    } else if (wf_empty_support(b0, b1)) {
        *lower = a0;
        *upper = a1;
    } else {
        *lower = a0 < b0 ? a0 : b0;
        *upper = a1 > b1 ? a1 : b1;
    }
}

static void wf_intersect_support(int64_t a0, int64_t a1,
                                 int64_t b0, int64_t b1,
                                 int64_t *lower, int64_t *upper) {
    *lower = a0 > b0 ? a0 : b0;
    *upper = a1 < b1 ? a1 : b1;
    if (*lower >= *upper) {
        *lower = INT64_MAX;
        *upper = INT64_MIN;
    }
}

static void wf_encode_node(uint8_t *data, const wf_node *node) {
    memset(data, 0, WF_NODE_SIZE);
    data[0] = node->op;
    data[1] = node->flags;
    wf_put_u16(data + 2, node->parameter_count);
    wf_put_u32(data + 4, node->parameter_count != 0
               ? node->parameter_offset : node->left);
    wf_put_u32(data + 8, node->right);
    wf_put_i64(data + 12, node->shift);
    wf_put_f64(data + 20, node->p0);
}

static void wf_decode_node(wf_node *node, const uint8_t *data) {
    memset(node, 0, sizeof(*node));
    node->op = data[0];
    node->flags = data[1];
    node->parameter_count = wf_get_u16(data + 2);
    node->left = wf_get_u32(data + 4);
    node->right = wf_get_u32(data + 8);
    node->shift = wf_get_i64(data + 12);
    node->p0 = wf_get_f64(data + 20);
    if (node->parameter_count != 0) {
        node->parameter_offset = node->left;
        node->left = 0;
    }
}

static int wf_prepare_decoded_node(wf_node *node, const wf_node *nodes,
                                   uint32_t index, size_t parameter_count) {
    if ((size_t)node->parameter_offset > parameter_count
            || (size_t)node->parameter_count
               > parameter_count - (size_t)node->parameter_offset) return -1;
    switch (node->op) {
        case WF_OP_CONSTANT:
            if (!isfinite(node->p0)) return -1;
            node->lower = node->p0 == 0.0 ? INT64_MAX : INT64_MIN;
            node->upper = node->p0 == 0.0 ? INT64_MIN : INT64_MAX;
            break;
        case WF_OP_GAUSSIAN:
            if (!isfinite(node->p0) || node->p0 <= 0.0) return -1;
            node->lower = wf_add_tick(
                node->shift, wf_seconds_to_tick(-0.75 * node->p0));
            node->upper = wf_add_tick(
                node->shift, wf_seconds_to_tick(0.75 * node->p0));
            node->p1 = (double)wf_seconds_to_tick(
                node->p0 / 3.3302184446307908)
                / (double)wf_ticks_per_second;
            if (node->p1 <= 0.0) return -1;
            break;
        case WF_OP_COS:
        case WF_OP_SIN:
            if (!isfinite(node->p0) || node->p0 <= 0.0) return -1;
            node->lower = INT64_MIN;
            node->upper = INT64_MAX;
            break;
        case WF_OP_SQUARE:
            if (!isfinite(node->p0) || node->p0 <= 0.0) return -1;
            node->lower = wf_add_tick(
                node->shift, wf_seconds_to_tick(-0.5 * node->p0));
            node->upper = wf_add_tick(
                node->shift, wf_seconds_to_tick(0.5 * node->p0));
            break;
        case WF_OP_LINEAR:
        case WF_OP_ERF:
        case WF_OP_SINC:
        case WF_OP_EXP:
        case WF_OP_INTERP:
        case WF_OP_LINEAR_CHIRP:
        case WF_OP_EXPONENTIAL_CHIRP:
        case WF_OP_HYPERBOLIC_CHIRP:
        case WF_OP_COSH:
        case WF_OP_SINH:
        case WF_OP_DRAG:
        case WF_OP_MOLLIFIER:
        case WF_OP_D_GAUSSIAN:
        case WF_OP_DRAG_SIN:
        case WF_OP_DRAG_SINX:
            if (!isfinite(node->p0)) return -1;
            node->lower = INT64_MIN;
            node->upper = INT64_MAX;
            break;
        case WF_OP_ADD:
            if (node->left >= index || node->right >= index) return -1;
            wf_union_support(nodes[node->left].lower, nodes[node->left].upper,
                             nodes[node->right].lower, nodes[node->right].upper,
                             &node->lower, &node->upper);
            break;
        case WF_OP_MUL:
            if (node->left >= index || node->right >= index) return -1;
            wf_intersect_support(
                nodes[node->left].lower, nodes[node->left].upper,
                nodes[node->right].lower, nodes[node->right].upper,
                &node->lower, &node->upper);
            break;
        case WF_OP_SCALE:
            if (node->left >= index || !isfinite(node->p0)) return -1;
            node->lower = node->p0 == 0.0
                ? INT64_MAX : nodes[node->left].lower;
            node->upper = node->p0 == 0.0
                ? INT64_MIN : nodes[node->left].upper;
            break;
        case WF_OP_POWER:
            if (node->left >= index || !isfinite(node->p0)
                    || node->p0 != floor(node->p0)) return -1;
            if (node->p0 == 0.0) {
                node->lower = INT64_MIN;
                node->upper = INT64_MAX;
            } else {
                node->lower = nodes[node->left].lower;
                node->upper = nodes[node->left].upper;
            }
            break;
        case WF_OP_WINDOW:
            if (node->left >= index || node->parameter_count != 0) return -1;
            node->lower = node->shift;
            memcpy(&node->upper, &node->p0, sizeof(node->upper));
            if (node->lower >= node->upper) {
                node->lower = INT64_MAX;
                node->upper = INT64_MIN;
            } else {
                int64_t child_lower = nodes[node->left].lower;
                int64_t child_upper = nodes[node->left].upper;
                if (child_lower > node->lower) node->lower = child_lower;
                if (child_upper < node->upper) node->upper = child_upper;
            }
            break;
        default:
            return -1;
    }
    return 0;
}

static cwaveform_wave *wf_wave_from_parts(const wf_node *nodes,
                                          uint32_t node_count,
                                          uint32_t root,
                                          const double *parameters,
                                          size_t parameter_count) {
    cwaveform_wave *wave;
    size_t size;
    uint32_t index;
    if (nodes == NULL || node_count == 0 || root >= node_count) {
        return NULL;
    }
    if ((size_t)node_count > (SIZE_MAX - WF_WAVE_HEADER_SIZE) / WF_NODE_SIZE
            || parameter_count > UINT32_MAX
            || parameter_count > (SIZE_MAX - WF_WAVE_HEADER_SIZE
                - (size_t)node_count * WF_NODE_SIZE) / sizeof(double)) {
        return NULL;
    }
    size = WF_WAVE_HEADER_SIZE + (size_t)node_count * WF_NODE_SIZE
        + parameter_count * sizeof(double);
    wave = (cwaveform_wave *)calloc(1, sizeof(*wave));
    if (wave == NULL) {
        return NULL;
    }
    wave->data = (uint8_t *)calloc(1, size);
    wave->nodes = (wf_node *)malloc((size_t)node_count * sizeof(*wave->nodes));
    wave->parameters = parameter_count == 0 ? NULL
        : (double *)malloc(parameter_count * sizeof(double));
    if (wave->data == NULL || wave->nodes == NULL
            || (parameter_count != 0 && wave->parameters == NULL)) {
        cwaveform_wave_release(wave);
        return NULL;
    }
    memcpy(wave->nodes, nodes, (size_t)node_count * sizeof(*nodes));
    if (parameter_count != 0)
        memcpy(wave->parameters, parameters, parameter_count * sizeof(double));
    wave->references = 1;
    wave->node_count = node_count;
    wave->root = root;
    wave->parameter_count = parameter_count;
    wave->data_size = size;
    memcpy(wave->data, "WNF4", 4);
    wf_put_u16(wave->data + 4, WF_VERSION);
    wf_put_u16(wave->data + 6, 0);
    wf_put_u32(wave->data + 8, node_count);
    wf_put_u32(wave->data + 12, root);
    wf_put_u32(wave->data + 16, (uint32_t)parameter_count);
    wf_put_u32(wave->data + 20, 0);
    for (index = 0; index < node_count; ++index) {
        wf_encode_node(wave->data + WF_WAVE_HEADER_SIZE
                       + (size_t)index * WF_NODE_SIZE, nodes + index);
    }
    for (index = 0; index < parameter_count; ++index) {
        wf_put_f64(wave->data + WF_WAVE_HEADER_SIZE
                   + (size_t)node_count * WF_NODE_SIZE
                   + (size_t)index * sizeof(double), parameters[index]);
    }
    wave->hash = wf_hash_bytes(wave->data, size);
    if (wf_wave_prepare_drag_sin_kernels(wave) != 0) {
        cwaveform_wave_release(wave);
        return NULL;
    }
    return wave;
}

static cwaveform_wave *wf_wave_from_nodes(const wf_node *nodes,
                                          uint32_t node_count,
                                          uint32_t root) {
    return wf_wave_from_parts(nodes, node_count, root, NULL, 0);
}

static cwaveform_wave *wf_wave_single(uint8_t op, int64_t lower, int64_t upper,
                                      int64_t shift, double p0, double p1) {
    wf_node node;
    memset(&node, 0, sizeof(node));
    node.op = op;
    node.lower = lower;
    node.upper = upper;
    node.shift = shift;
    node.p0 = p0;
    node.p1 = p1;
    return wf_wave_from_nodes(&node, 1, 0);
}

uint64_t cwaveform_ticks_per_second(void) {
    return wf_ticks_per_second;
}

int cwaveform_set_ticks_per_second(uint64_t ticks_per_second) {
    if (ticks_per_second == 0) return -1;
    wf_ticks_per_second = ticks_per_second;
    return 0;
}

cwaveform_wave *cwaveform_wave_constant(double value) {
    int64_t lower = value == 0.0 ? INT64_MAX : INT64_MIN;
    int64_t upper = value == 0.0 ? INT64_MIN : INT64_MAX;
    if (!isfinite(value)) {
        return NULL;
    }
    return wf_wave_single(WF_OP_CONSTANT, lower, upper, 0, value, 0.0);
}

cwaveform_wave *cwaveform_wave_gaussian(double width_seconds) {
    int64_t lower;
    int64_t upper;
    if (!isfinite(width_seconds) || width_seconds <= 0.0) {
        return cwaveform_wave_constant(0.0);
    }
    lower = wf_seconds_to_tick(-0.75 * width_seconds);
    upper = wf_seconds_to_tick(0.75 * width_seconds);
    return wf_wave_single(WF_OP_GAUSSIAN, lower, upper, 0,
                          width_seconds,
                          (double)wf_seconds_to_tick(
                              width_seconds / 3.3302184446307908)
                          / (double)wf_ticks_per_second);
}

cwaveform_wave *cwaveform_wave_cos(double angular_frequency, double phase) {
    if (!isfinite(angular_frequency) || !isfinite(phase)) {
        return NULL;
    }
    if (angular_frequency == 0.0) {
        return cwaveform_wave_constant(cos(phase));
    }
    if (angular_frequency < 0.0) {
        angular_frequency = -angular_frequency;
        phase = -phase;
    }
    return wf_wave_single(
        WF_OP_COS, INT64_MIN, INT64_MAX,
        wf_seconds_to_tick(-phase / angular_frequency),
        angular_frequency, 0.0
    );
}

cwaveform_wave *cwaveform_wave_sin(double angular_frequency, double phase) {
    if (!isfinite(angular_frequency) || !isfinite(phase)) {
        return NULL;
    }
    if (angular_frequency == 0.0) {
        return cwaveform_wave_constant(sin(phase));
    }
    if (angular_frequency < 0.0) {
        angular_frequency = -angular_frequency;
        phase = -phase + 3.14159265358979323846;
    }
    return wf_wave_single(
        WF_OP_COS, INT64_MIN, INT64_MAX,
        wf_seconds_to_tick((1.57079632679489661923 - phase)
                           / angular_frequency),
        angular_frequency, 0.0
    );
}

cwaveform_wave *cwaveform_wave_square(double width_seconds) {
    int64_t lower;
    int64_t upper;
    if (!isfinite(width_seconds) || width_seconds <= 0.0) {
        return cwaveform_wave_constant(0.0);
    }
    lower = wf_seconds_to_tick(-0.5 * width_seconds);
    upper = wf_seconds_to_tick(0.5 * width_seconds);
    return wf_wave_single(WF_OP_SQUARE, lower, upper, 0,
                          width_seconds, 0.0);
}

static uint8_t wf_builtin_op(int builtin) {
    switch (builtin) {
        case CWAVEFORM_LINEAR: return WF_OP_LINEAR;
        case CWAVEFORM_GAUSSIAN: return WF_OP_GAUSSIAN;
        case CWAVEFORM_ERF: return WF_OP_ERF;
        case CWAVEFORM_COS: return WF_OP_COS;
        case CWAVEFORM_SINC: return WF_OP_SINC;
        case CWAVEFORM_EXP: return WF_OP_EXP;
        case CWAVEFORM_INTERP: return WF_OP_INTERP;
        case CWAVEFORM_LINEAR_CHIRP: return WF_OP_LINEAR_CHIRP;
        case CWAVEFORM_EXPONENTIAL_CHIRP: return WF_OP_EXPONENTIAL_CHIRP;
        case CWAVEFORM_HYPERBOLIC_CHIRP: return WF_OP_HYPERBOLIC_CHIRP;
        case CWAVEFORM_COSH: return WF_OP_COSH;
        case CWAVEFORM_SINH: return WF_OP_SINH;
        case CWAVEFORM_DRAG: return WF_OP_DRAG;
        case CWAVEFORM_MOLLIFIER: return WF_OP_MOLLIFIER;
        case CWAVEFORM_D_GAUSSIAN: return WF_OP_D_GAUSSIAN;
        case CWAVEFORM_DRAG_SIN: return WF_OP_DRAG_SIN;
        case CWAVEFORM_DRAG_SINX: return WF_OP_DRAG_SINX;
        default: return 0;
    }
}

static int wf_builtin_parameter_count_valid(int builtin, size_t count) {
    switch (builtin) {
        case CWAVEFORM_LINEAR: return count == 0;
        case CWAVEFORM_GAUSSIAN:
        case CWAVEFORM_ERF:
        case CWAVEFORM_COS:
        case CWAVEFORM_SINC:
        case CWAVEFORM_EXP:
        case CWAVEFORM_COSH:
        case CWAVEFORM_SINH: return count == 1;
        case CWAVEFORM_INTERP: return count >= 3;
        case CWAVEFORM_LINEAR_CHIRP: return count == 4;
        case CWAVEFORM_EXPONENTIAL_CHIRP:
        case CWAVEFORM_HYPERBOLIC_CHIRP: return count == 3;
        case CWAVEFORM_DRAG: return count == 6;
        case CWAVEFORM_MOLLIFIER:
        case CWAVEFORM_D_GAUSSIAN: return count == 2;
        case CWAVEFORM_DRAG_SIN:
        case CWAVEFORM_DRAG_SINX: return count >= 7;
        default: return 0;
    }
}

cwaveform_wave *cwaveform_wave_builtin(
    int builtin, const double *parameters, size_t parameter_count,
    int64_t shift_tick) {
    wf_node node;
    uint8_t op = wf_builtin_op(builtin);
    size_t index;
    if (op == 0 || !wf_builtin_parameter_count_valid(builtin, parameter_count)
            || parameter_count > UINT16_MAX + (size_t)1
            || (parameter_count != 0 && parameters == NULL)) return NULL;
    for (index = 0; index < parameter_count; ++index) {
        if (!isfinite(parameters[index])) {
            if (!(builtin == CWAVEFORM_DRAG && index == 4
                  && isnan(parameters[index]))
                    && !((builtin == CWAVEFORM_DRAG_SIN
                          || builtin == CWAVEFORM_DRAG_SINX)
                         && index == 6 && isnan(parameters[index]))) return NULL;
        }
    }
    memset(&node, 0, sizeof(node));
    node.op = op;
    node.flags = builtin == CWAVEFORM_GAUSSIAN ? 1 : 0;
    node.shift = shift_tick;
    node.p0 = parameter_count == 0 ? 0.0 : parameters[0];
    node.parameter_count = parameter_count == 0
        ? 0 : (uint16_t)(parameter_count - 1);
    node.parameter_offset = 0;
    node.lower = INT64_MIN;
    node.upper = INT64_MAX;
    return wf_wave_from_parts(&node, 1, 0,
                              parameter_count <= 1 ? NULL : parameters + 1,
                              parameter_count <= 1 ? 0 : parameter_count - 1);
}

cwaveform_wave *cwaveform_wave_window(const cwaveform_wave *wave,
                                      int64_t lower_tick,
                                      int64_t upper_tick) {
    wf_node *nodes;
    wf_node node;
    cwaveform_wave *result;
    double *parameters;
    if (wave == NULL) return NULL;
    if (lower_tick >= upper_tick) return cwaveform_wave_constant(0.0);
    nodes = (wf_node *)calloc((size_t)wave->node_count + 1, sizeof(*nodes));
    parameters = wave->parameter_count == 0 ? NULL
        : (double *)malloc(wave->parameter_count * sizeof(double));
    if (nodes == NULL || (wave->parameter_count != 0 && parameters == NULL)) {
        free(nodes);
        free(parameters);
        return NULL;
    }
    memcpy(nodes, wave->nodes, (size_t)wave->node_count * sizeof(*nodes));
    if (wave->parameter_count != 0)
        memcpy(parameters, wave->parameters,
               wave->parameter_count * sizeof(double));
    memset(&node, 0, sizeof(node));
    node.op = WF_OP_WINDOW;
    node.left = wave->root;
    node.shift = lower_tick;
    memcpy(&node.p0, &upper_tick, sizeof(upper_tick));
    node.lower = lower_tick > wave->nodes[wave->root].lower
        ? lower_tick : wave->nodes[wave->root].lower;
    node.upper = upper_tick < wave->nodes[wave->root].upper
        ? upper_tick : wave->nodes[wave->root].upper;
    nodes[wave->node_count] = node;
    result = wf_wave_from_parts(nodes, wave->node_count + 1,
                                wave->node_count, parameters,
                                wave->parameter_count);
    free(nodes);
    free(parameters);
    return result;
}

cwaveform_wave *cwaveform_wave_power(const cwaveform_wave *wave, int power) {
    wf_node *nodes;
    wf_node node;
    cwaveform_wave *result;
    double *parameters;
    if (wave == NULL) return NULL;
    if (power == 0) return cwaveform_wave_constant(1.0);
    if (power == 1) {
        cwaveform_wave_retain((cwaveform_wave *)wave);
        return (cwaveform_wave *)wave;
    }
    nodes = (wf_node *)calloc((size_t)wave->node_count + 1, sizeof(*nodes));
    parameters = wave->parameter_count == 0 ? NULL
        : (double *)malloc(wave->parameter_count * sizeof(double));
    if (nodes == NULL || (wave->parameter_count != 0 && parameters == NULL)) {
        free(nodes);
        free(parameters);
        return NULL;
    }
    memcpy(nodes, wave->nodes, (size_t)wave->node_count * sizeof(*nodes));
    if (wave->parameter_count != 0)
        memcpy(parameters, wave->parameters,
               wave->parameter_count * sizeof(double));
    memset(&node, 0, sizeof(node));
    node.op = WF_OP_POWER;
    node.left = wave->root;
    node.p0 = (double)power;
    node.lower = wave->nodes[wave->root].lower;
    node.upper = wave->nodes[wave->root].upper;
    nodes[wave->node_count] = node;
    result = wf_wave_from_parts(nodes, wave->node_count + 1,
                                wave->node_count, parameters,
                                wave->parameter_count);
    free(nodes);
    free(parameters);
    return result;
}

static cwaveform_wave *wf_subwave(const cwaveform_wave *wave, uint32_t root) {
    if (wave == NULL || root >= wave->node_count) return NULL;
    return wf_wave_from_parts(wave->nodes, root + 1, root,
                              wave->parameters, wave->parameter_count);
}

static cwaveform_wave *wf_scaled(cwaveform_wave *wave, double scale) {
    cwaveform_wave *result;
    if (wave == NULL) return NULL;
    result = cwaveform_wave_materialize(wave, 0, scale);
    cwaveform_wave_release(wave);
    return result;
}

static cwaveform_wave *wf_add_owned(cwaveform_wave *left,
                                    cwaveform_wave *right) {
    cwaveform_wave *result;
    if (left == NULL || right == NULL) {
        cwaveform_wave_release(left);
        cwaveform_wave_release(right);
        return NULL;
    }
    result = cwaveform_wave_add_affine(left, 0, 1.0, right, 0, 1.0);
    cwaveform_wave_release(left);
    cwaveform_wave_release(right);
    return result;
}

static cwaveform_wave *wf_mul_owned(cwaveform_wave *left,
                                    cwaveform_wave *right) {
    cwaveform_wave *result;
    if (left == NULL || right == NULL) {
        cwaveform_wave_release(left);
        cwaveform_wave_release(right);
        return NULL;
    }
    result = cwaveform_wave_mul_affine(left, 0, 1.0, right, 0, 1.0);
    cwaveform_wave_release(left);
    cwaveform_wave_release(right);
    return result;
}

static cwaveform_wave *wf_derivative_node(const cwaveform_wave *wave,
                                          uint32_t root) {
    const wf_node *node = wave->nodes + root;
    double parameters[8];
    cwaveform_wave *left;
    cwaveform_wave *right;
    cwaveform_wave *result;
    switch (node->op) {
        case WF_OP_CONSTANT:
        case WF_OP_SQUARE:
            return cwaveform_wave_constant(0.0);
        case WF_OP_LINEAR:
            return cwaveform_wave_constant(1.0);
        case WF_OP_GAUSSIAN: {
            double std = node->flags ? node->p0 : node->p1;
            parameters[0] = std;
            parameters[1] = 1.0;
            result = cwaveform_wave_builtin(
                CWAVEFORM_D_GAUSSIAN, parameters, 2, node->shift);
            if (!node->flags) {
                cwaveform_wave *windowed = cwaveform_wave_window(
                    result, node->lower, node->upper);
                cwaveform_wave_release(result);
                result = windowed;
            }
            return result;
        }
        case WF_OP_D_GAUSSIAN:
            parameters[0] = node->p0;
            parameters[1] = wf_node_parameter(wave, node, 1) + 1.0;
            return cwaveform_wave_builtin(
                CWAVEFORM_D_GAUSSIAN, parameters, 2, node->shift);
        case WF_OP_ERF:
            parameters[0] = node->p0;
            result = cwaveform_wave_builtin(
                CWAVEFORM_GAUSSIAN, parameters, 1, node->shift);
            return wf_scaled(result,
                2.0 / (node->p0 * sqrt(3.14159265358979323846)));
        case WF_OP_COS:
            parameters[0] = node->p0;
            result = cwaveform_wave_builtin(
                CWAVEFORM_COS, parameters, 1,
                wf_add_tick(node->shift,
                    wf_seconds_to_tick(-3.14159265358979323846
                                       / (2.0 * node->p0))));
            return wf_scaled(result, node->p0);
        case WF_OP_SIN:
            parameters[0] = node->p0;
            result = cwaveform_wave_builtin(
                CWAVEFORM_COS, parameters, 1, node->shift);
            return wf_scaled(result, node->p0);
        case WF_OP_SINC: {
            double frequency = 3.14159265358979323846 * node->p0;
            cwaveform_wave *variable;
            cwaveform_wave *carrier;
            cwaveform_wave *first;
            cwaveform_wave *second;
            parameters[0] = frequency;
            variable = cwaveform_wave_builtin(
                CWAVEFORM_LINEAR, NULL, 0, node->shift);
            carrier = cwaveform_wave_builtin(
                CWAVEFORM_COS, parameters, 1, node->shift);
            first = wf_mul_owned(cwaveform_wave_power(variable, -1), carrier);
            cwaveform_wave_release(variable);
            variable = cwaveform_wave_builtin(
                CWAVEFORM_LINEAR, NULL, 0, node->shift);
            carrier = cwaveform_wave_builtin(
                CWAVEFORM_COS, parameters, 1,
                wf_add_tick(node->shift,
                    wf_seconds_to_tick(3.14159265358979323846
                                       / (2.0 * frequency))));
            second = wf_scaled(wf_mul_owned(
                cwaveform_wave_power(variable, -2), carrier), -1.0 / frequency);
            cwaveform_wave_release(variable);
            return wf_add_owned(first, second);
        }
        case WF_OP_EXP:
            result = wf_subwave(wave, root);
            return wf_scaled(result, node->p0);
        case WF_OP_INTERP: {
            size_t total = (size_t)node->parameter_count + 1;
            size_t point_count = total - 2;
            double *gradient = (double *)malloc(total * sizeof(double));
            size_t index;
            if (gradient == NULL) return NULL;
            gradient[0] = node->p0;
            gradient[1] = wf_node_parameter(wave, node, 1);
            for (index = 0; index < point_count; ++index) {
                double value;
                if (index == 0)
                    value = wf_node_parameter(wave, node, 3)
                        - wf_node_parameter(wave, node, 2);
                else if (index + 1 == point_count)
                    value = wf_node_parameter(wave, node, point_count + 1)
                        - wf_node_parameter(wave, node, point_count);
                else
                    value = (wf_node_parameter(wave, node, index + 3)
                             - wf_node_parameter(wave, node, index + 1)) / 2.0;
                gradient[index + 2] = value;
            }
            result = cwaveform_wave_builtin(
                CWAVEFORM_INTERP, gradient, total, node->shift);
            free(gradient);
            return wf_scaled(result, (double)(point_count - 1)
                / (wf_node_parameter(wave, node, 1) - node->p0));
        }
        case WF_OP_COSH:
            parameters[0] = node->p0;
            result = cwaveform_wave_builtin(
                CWAVEFORM_SINH, parameters, 1, node->shift);
            return wf_scaled(result, node->p0);
        case WF_OP_SINH:
            parameters[0] = node->p0;
            result = cwaveform_wave_builtin(
                CWAVEFORM_COSH, parameters, 1, node->shift);
            return wf_scaled(result, node->p0);
        case WF_OP_LINEAR_CHIRP: {
            double f0 = node->p0;
            double f1 = wf_node_parameter(wave, node, 1);
            double duration = wf_node_parameter(wave, node, 2);
            parameters[0] = f0;
            parameters[1] = f1;
            parameters[2] = duration;
            parameters[3] = wf_node_parameter(wave, node, 3)
                + 3.14159265358979323846 / 2.0;
            left = wf_scaled(cwaveform_wave_builtin(
                CWAVEFORM_LINEAR_CHIRP, parameters, 4, node->shift),
                2.0 * 3.14159265358979323846 * f0);
            right = wf_mul_owned(
                cwaveform_wave_builtin(CWAVEFORM_LINEAR, NULL, 0, node->shift),
                cwaveform_wave_builtin(
                    CWAVEFORM_LINEAR_CHIRP, parameters, 4, node->shift));
            right = wf_scaled(right, 2.0 * 3.14159265358979323846
                              * (f1 - f0) / duration);
            return wf_add_owned(left, right);
        }
        case WF_OP_EXPONENTIAL_CHIRP:
            parameters[0] = node->p0;
            parameters[1] = wf_node_parameter(wave, node, 1);
            parameters[2] = wf_node_parameter(wave, node, 2)
                + 3.14159265358979323846 / 2.0;
            left = cwaveform_wave_builtin(
                CWAVEFORM_EXP, parameters + 1, 1, node->shift);
            right = cwaveform_wave_builtin(
                CWAVEFORM_EXPONENTIAL_CHIRP, parameters, 3, node->shift);
            return wf_scaled(wf_mul_owned(left, right),
                2.0 * 3.14159265358979323846 * node->p0);
        case WF_OP_HYPERBOLIC_CHIRP: {
            double k = wf_node_parameter(wave, node, 1);
            parameters[0] = node->p0;
            parameters[1] = k;
            parameters[2] = wf_node_parameter(wave, node, 2)
                + 3.14159265358979323846 / 2.0;
            left = cwaveform_wave_builtin(
                CWAVEFORM_LINEAR, NULL, 0,
                wf_add_tick(node->shift, wf_seconds_to_tick(-1.0 / k)));
            left = cwaveform_wave_power(left, -1);
            right = cwaveform_wave_builtin(
                CWAVEFORM_HYPERBOLIC_CHIRP, parameters, 3, node->shift);
            return wf_scaled(wf_mul_owned(left, right),
                2.0 * 3.14159265358979323846 * node->p0);
        }
        case WF_OP_MOLLIFIER:
            parameters[0] = node->p0;
            parameters[1] = wf_node_parameter(wave, node, 1) + 1.0;
            return cwaveform_wave_builtin(
                CWAVEFORM_MOLLIFIER, parameters, 2, node->shift);
        case WF_OP_ADD:
            return wf_add_owned(wf_derivative_node(wave, node->left),
                                wf_derivative_node(wave, node->right));
        case WF_OP_MUL:
            left = wf_mul_owned(wf_derivative_node(wave, node->left),
                                wf_subwave(wave, node->right));
            right = wf_mul_owned(wf_subwave(wave, node->left),
                                 wf_derivative_node(wave, node->right));
            return wf_add_owned(left, right);
        case WF_OP_SCALE:
            return wf_scaled(wf_derivative_node(wave, node->left), node->p0);
        case WF_OP_POWER:
            if (node->p0 == 0.0) return cwaveform_wave_constant(0.0);
            left = wf_subwave(wave, node->left);
            right = cwaveform_wave_power(left, (int)node->p0 - 1);
            cwaveform_wave_release(left);
            return wf_scaled(wf_mul_owned(
                right, wf_derivative_node(wave, node->left)), node->p0);
        case WF_OP_WINDOW: {
            int64_t upper;
            memcpy(&upper, &node->p0, sizeof(upper));
            left = wf_derivative_node(wave, node->left);
            result = cwaveform_wave_window(left, node->shift, upper);
            cwaveform_wave_release(left);
            return result;
        }
        default:
            return NULL;
    }
}

cwaveform_wave *cwaveform_wave_derivative(const cwaveform_wave *wave,
                                          unsigned order) {
    cwaveform_wave *result;
    unsigned index;
    if (wave == NULL) return NULL;
    if (order == 0) {
        cwaveform_wave_retain((cwaveform_wave *)wave);
        return (cwaveform_wave *)wave;
    }
    result = wf_derivative_node(wave, wave->root);
    for (index = 1; index < order && result != NULL; ++index) {
        cwaveform_wave *next = wf_derivative_node(result, result->root);
        cwaveform_wave_release(result);
        result = next;
    }
    return result;
}

static uint32_t wf_clone_affine(const cwaveform_wave *source, wf_node *target,
                                uint32_t offset, uint32_t parameter_offset,
                                int64_t delay, double scale, uint32_t *next);

typedef struct wf_spectral_term {
    cwaveform_wave *envelope;
    double frequency;
    double real;
    double imag;
} wf_spectral_term;

typedef struct wf_spectrum {
    wf_spectral_term *terms;
    size_t count;
    size_t capacity;
} wf_spectrum;

static void wf_make_zero_node(wf_node *node) {
    memset(node, 0, sizeof(*node));
    node->op = WF_OP_CONSTANT;
    node->lower = INT64_MAX;
    node->upper = INT64_MIN;
}

static void wf_spectrum_clear(wf_spectrum *spectrum) {
    size_t index;
    if (spectrum == NULL) return;
    for (index = 0; index < spectrum->count; ++index)
        cwaveform_wave_release(spectrum->terms[index].envelope);
    free(spectrum->terms);
    memset(spectrum, 0, sizeof(*spectrum));
}

/* Append one signed complex-exponential component.  Equal envelope/frequency
 * pairs are combined as they are produced so powers of one carrier grow
 * linearly rather than exponentially.  The envelope reference is consumed. */
static int wf_spectrum_append(wf_spectrum *spectrum,
                              cwaveform_wave *envelope,
                              double frequency, double real, double imag) {
    size_t index;
    wf_spectral_term *terms;
    size_t capacity;
    if (spectrum == NULL || envelope == NULL || !isfinite(frequency)
            || !isfinite(real) || !isfinite(imag)) {
        cwaveform_wave_release(envelope);
        return -1;
    }
    if (frequency == 0.0) frequency = 0.0;
    if (real == 0.0 && imag == 0.0) {
        cwaveform_wave_release(envelope);
        return 0;
    }
    for (index = 0; index < spectrum->count; ++index) {
        wf_spectral_term *term = spectrum->terms + index;
        if (term->frequency == frequency
                && cwaveform_wave_equal(term->envelope, envelope)) {
            term->real += real;
            term->imag += imag;
            cwaveform_wave_release(envelope);
            if (!isfinite(term->real) || !isfinite(term->imag)) return -1;
            if (term->real == 0.0 && term->imag == 0.0) {
                cwaveform_wave_release(term->envelope);
                spectrum->terms[index] = spectrum->terms[spectrum->count - 1];
                --spectrum->count;
            }
            return 0;
        }
    }
    if (spectrum->count == spectrum->capacity) {
        capacity = spectrum->capacity == 0 ? 8 : spectrum->capacity * 2;
        if (capacity < spectrum->capacity
                || capacity > SIZE_MAX / sizeof(*terms)) {
            cwaveform_wave_release(envelope);
            return -1;
        }
        terms = (wf_spectral_term *)realloc(
            spectrum->terms, capacity * sizeof(*terms));
        if (terms == NULL) {
            cwaveform_wave_release(envelope);
            return -1;
        }
        spectrum->terms = terms;
        spectrum->capacity = capacity;
    }
    spectrum->terms[spectrum->count].envelope = envelope;
    spectrum->terms[spectrum->count].frequency = frequency;
    spectrum->terms[spectrum->count].real = real;
    spectrum->terms[spectrum->count].imag = imag;
    ++spectrum->count;
    return 0;
}

static int wf_spectrum_move(wf_spectrum *target, wf_spectrum *source) {
    size_t index;
    for (index = 0; index < source->count; ++index) {
        wf_spectral_term *term = source->terms + index;
        cwaveform_wave *envelope = term->envelope;
        term->envelope = NULL;
        if (wf_spectrum_append(target, envelope, term->frequency,
                               term->real, term->imag) != 0) return -1;
    }
    return 0;
}

static int wf_spectrum_product(const wf_spectrum *left,
                               const wf_spectrum *right,
                               wf_spectrum *result) {
    size_t i;
    size_t j;
    for (i = 0; i < left->count; ++i) {
        for (j = 0; j < right->count; ++j) {
            const wf_spectral_term *a = left->terms + i;
            const wf_spectral_term *b = right->terms + j;
            cwaveform_wave *left_envelope = a->envelope;
            cwaveform_wave *right_envelope = b->envelope;
            cwaveform_wave *envelope;
            double real = a->real * b->real - a->imag * b->imag;
            double imag = a->real * b->imag + a->imag * b->real;
            cwaveform_wave_retain(left_envelope);
            cwaveform_wave_retain(right_envelope);
            envelope = wf_mul_owned(left_envelope, right_envelope);
            if (wf_spectrum_append(result, envelope,
                                   a->frequency + b->frequency,
                                   real, imag) != 0) return -1;
        }
    }
    return 0;
}

static int wf_spectrum_node(const cwaveform_wave *wave, uint32_t index,
                            wf_spectrum *result);

static int wf_spectrum_power(const cwaveform_wave *wave, uint32_t index,
                             unsigned exponent, wf_spectrum *result) {
    wf_spectrum base = {0};
    wf_spectrum power = {0};
    cwaveform_wave *one = cwaveform_wave_constant(1.0);
    int status = -1;
    if (one == NULL || wf_spectrum_append(&power, one, 0.0, 1.0, 0.0) != 0)
        goto done;
    if (wf_spectrum_node(wave, index, &base) != 0) goto done;
    while (exponent != 0) {
        if (exponent & 1u) {
            wf_spectrum next = {0};
            if (wf_spectrum_product(&power, &base, &next) != 0) {
                wf_spectrum_clear(&next);
                goto done;
            }
            wf_spectrum_clear(&power);
            power = next;
        }
        exponent >>= 1;
        if (exponent != 0) {
            wf_spectrum next = {0};
            if (wf_spectrum_product(&base, &base, &next) != 0) {
                wf_spectrum_clear(&next);
                goto done;
            }
            wf_spectrum_clear(&base);
            base = next;
        }
    }
    if (wf_spectrum_move(result, &power) != 0) goto done;
    status = 0;
done:
    wf_spectrum_clear(&base);
    wf_spectrum_clear(&power);
    return status;
}

/* Convert the real expression tree into signed complex-exponential terms.
 * Non-trigonometric nodes are retained as envelopes, matching the historic
 * symbolic filter which treats their spectrum as baseband. */
static int wf_spectrum_node(const cwaveform_wave *wave, uint32_t index,
                            wf_spectrum *result) {
    const wf_node *node = wave->nodes + index;
    wf_spectrum left = {0};
    wf_spectrum right = {0};
    wf_spectrum product = {0};
    cwaveform_wave *envelope;
    size_t term_index;
    int status = -1;
    switch (node->op) {
        case WF_OP_CONSTANT:
            envelope = cwaveform_wave_constant(1.0);
            return wf_spectrum_append(result, envelope, 0.0, node->p0, 0.0);
        case WF_OP_COS:
        case WF_OP_SIN: {
            const double half = 0.5;
            const double two_pi = 6.28318530717958647692;
            double phase = remainder(
                -node->p0 * (double)node->shift
                    / (double)wf_ticks_per_second,
                two_pi);
            double cosine = cos(phase);
            double sine = sin(phase);
            envelope = cwaveform_wave_constant(1.0);
            if (node->op == WF_OP_COS) {
                if (wf_spectrum_append(result, envelope, node->p0,
                                       half * cosine, half * sine) != 0)
                    return -1;
                envelope = cwaveform_wave_constant(1.0);
                return wf_spectrum_append(result, envelope, -node->p0,
                                          half * cosine, -half * sine);
            }
            if (wf_spectrum_append(result, envelope, node->p0,
                                   half * sine, -half * cosine) != 0)
                return -1;
            envelope = cwaveform_wave_constant(1.0);
            return wf_spectrum_append(result, envelope, -node->p0,
                                      half * sine, half * cosine);
        }
        case WF_OP_ADD:
            if (wf_spectrum_node(wave, node->left, &left) != 0
                    || wf_spectrum_node(wave, node->right, &right) != 0
                    || wf_spectrum_move(result, &left) != 0
                    || wf_spectrum_move(result, &right) != 0) goto done;
            status = 0;
            break;
        case WF_OP_MUL:
            if (wf_spectrum_node(wave, node->left, &left) != 0
                    || wf_spectrum_node(wave, node->right, &right) != 0
                    || wf_spectrum_product(&left, &right, &product) != 0
                    || wf_spectrum_move(result, &product) != 0) goto done;
            status = 0;
            break;
        case WF_OP_SCALE:
            if (wf_spectrum_node(wave, node->left, &left) != 0) goto done;
            for (term_index = 0; term_index < left.count; ++term_index) {
                left.terms[term_index].real *= node->p0;
                left.terms[term_index].imag *= node->p0;
            }
            if (wf_spectrum_move(result, &left) != 0) goto done;
            status = 0;
            break;
        case WF_OP_POWER:
            if (node->p0 >= 0.0 && node->p0 <= (double)UINT_MAX)
                return wf_spectrum_power(
                    wave, node->left, (unsigned)node->p0, result);
            envelope = wf_subwave(wave, index);
            return wf_spectrum_append(result, envelope, 0.0, 1.0, 0.0);
        case WF_OP_WINDOW: {
            int64_t upper;
            if (wf_spectrum_node(wave, node->left, &left) != 0) goto done;
            memcpy(&upper, &node->p0, sizeof(upper));
            for (term_index = 0; term_index < left.count; ++term_index) {
                cwaveform_wave *windowed = cwaveform_wave_window(
                    left.terms[term_index].envelope, node->shift, upper);
                if (windowed == NULL) goto done;
                cwaveform_wave_release(left.terms[term_index].envelope);
                left.terms[term_index].envelope = windowed;
            }
            if (wf_spectrum_move(result, &left) != 0) goto done;
            status = 0;
            break;
        }
        default:
            envelope = wf_subwave(wave, index);
            return wf_spectrum_append(result, envelope, 0.0, 1.0, 0.0);
    }
done:
    wf_spectrum_clear(&left);
    wf_spectrum_clear(&right);
    wf_spectrum_clear(&product);
    return status;
}

static cwaveform_wave *wf_spectral_term_wave(const wf_spectral_term *term) {
    cwaveform_wave *envelope = term->envelope;
    cwaveform_wave *carrier;
    double amplitude;
    if (term->frequency == 0.0) {
        if (term->real == 0.0)
            return cwaveform_wave_constant(0.0);
        cwaveform_wave_retain(envelope);
        return wf_scaled(envelope, term->real);
    }
    /* Only the positive half of a real signal's conjugate spectrum is
     * reconstructed; double it here to recover the real carrier amplitude. */
    amplitude = 2.0 * wf_hypot(term->real, term->imag);
    if (amplitude == 0.0) return cwaveform_wave_constant(0.0);
    carrier = cwaveform_wave_cos(
        term->frequency, atan2(term->imag, term->real));
    carrier = wf_scaled(carrier, amplitude);
    cwaveform_wave_retain(envelope);
    return wf_mul_owned(envelope, carrier);
}

cwaveform_wave *cwaveform_wave_filter(const cwaveform_wave *wave,
                                      double low, double high,
                                      double epsilon) {
    wf_spectrum spectrum = {0};
    cwaveform_wave *result = NULL;
    size_t index;
    if (wave == NULL || !isfinite(low) || isnan(high) || low < 0.0
            || high < low || !isfinite(epsilon) || epsilon < 0.0) return NULL;
    if (wf_spectrum_node(wave, wave->root, &spectrum) != 0) goto done;
    result = cwaveform_wave_constant(0.0);
    if (result == NULL) goto done;
    for (index = 0; index < spectrum.count; ++index) {
        const wf_spectral_term *term = spectrum.terms + index;
        double frequency = fabs(term->frequency);
        if (term->frequency < 0.0) continue;
        if (low <= frequency && frequency < high) {
            cwaveform_wave *component = wf_spectral_term_wave(term);
            result = wf_add_owned(result, component);
            if (result == NULL) goto done;
        }
    }
    if (result != NULL) {
        cwaveform_wave *compact = cwaveform_wave_simplify(result, epsilon);
        cwaveform_wave_release(result);
        result = compact;
    }
done:
    wf_spectrum_clear(&spectrum);
    return result;
}

typedef struct wf_canonical_term {
    cwaveform_wave *wave;
    double scale;
} wf_canonical_term;

static int wf_wave_byte_compare(const cwaveform_wave *left,
                                const cwaveform_wave *right) {
    size_t common = left->data_size < right->data_size
        ? left->data_size : right->data_size;
    int result = memcmp(left->data, right->data, common);
    if (result != 0) return result;
    if (left->data_size < right->data_size) return -1;
    if (left->data_size > right->data_size) return 1;
    return 0;
}

static int wf_term_compare(const void *a, const void *b) {
    const wf_canonical_term *left = (const wf_canonical_term *)a;
    const wf_canonical_term *right = (const wf_canonical_term *)b;
    return wf_wave_byte_compare(left->wave, right->wave);
}

static int wf_wave_pointer_compare(const void *a, const void *b) {
    const cwaveform_wave *left = *(cwaveform_wave *const *)a;
    const cwaveform_wave *right = *(cwaveform_wave *const *)b;
    return wf_wave_byte_compare(left, right);
}

static cwaveform_wave *wf_simplify_node(const cwaveform_wave *wave,
                                        uint32_t root, double epsilon);

/* Flatten one associative operator without constructing intermediate waves. */
static int wf_collect_roots(const cwaveform_wave *wave, uint32_t root,
                            uint8_t operation, uint32_t *roots,
                            size_t *root_count) {
    uint32_t *pending;
    size_t pending_count = 0;
    pending = (uint32_t *)malloc((size_t)wave->node_count * sizeof(*pending));
    if (pending == NULL) return -1;
    pending[pending_count++] = root;
    while (pending_count != 0) {
        uint32_t index = pending[--pending_count];
        const wf_node *node = wave->nodes + index;
        if (node->op == operation) {
            pending[pending_count++] = node->right;
            pending[pending_count++] = node->left;
        } else {
            roots[(*root_count)++] = index;
        }
    }
    free(pending);
    return 0;
}

/* Consume sorted compact operands and concatenate their blocks in one pass. */
static cwaveform_wave *wf_concat_owned(cwaveform_wave **waves, size_t count,
                                       uint8_t operation) {
    size_t index;
    size_t capacity = count == 0 ? 1 : count - 1;
    size_t parameter_count = 0;
    size_t parameter_cursor = 0;
    uint32_t next = 0;
    uint32_t root = UINT32_MAX;
    wf_node *nodes;
    double *parameters;
    cwaveform_wave *result = NULL;
    for (index = 0; index < count; ++index) {
        if ((size_t)waves[index]->node_count > SIZE_MAX - capacity
                || waves[index]->parameter_count
                   > SIZE_MAX - parameter_count) goto done;
        capacity += waves[index]->node_count;
        parameter_count += waves[index]->parameter_count;
    }
    if (capacity > UINT32_MAX || parameter_count > UINT32_MAX) goto done;
    nodes = (wf_node *)calloc(capacity, sizeof(*nodes));
    parameters = parameter_count == 0 ? NULL
        : (double *)malloc(parameter_count * sizeof(*parameters));
    if (nodes == NULL || (parameter_count != 0 && parameters == NULL)) {
        free(nodes);
        free(parameters);
        goto done;
    }
    for (index = 0; index < count; ++index) {
        cwaveform_wave *item = waves[index];
        uint32_t item_root;
        if (item->parameter_count != 0)
            memcpy(parameters + parameter_cursor, item->parameters,
                   item->parameter_count * sizeof(*parameters));
        item_root = wf_clone_affine(item, nodes, next,
                                    (uint32_t)parameter_cursor,
                                    0, 1.0, &next);
        parameter_cursor += item->parameter_count;
        if (root == UINT32_MAX) {
            root = item_root;
        } else {
            wf_node node;
            memset(&node, 0, sizeof(node));
            node.op = operation;
            node.left = root;
            node.right = item_root;
            if (operation == WF_OP_ADD)
                wf_union_support(nodes[root].lower, nodes[root].upper,
                                 nodes[item_root].lower,
                                 nodes[item_root].upper,
                                 &node.lower, &node.upper);
            else
                wf_intersect_support(nodes[root].lower, nodes[root].upper,
                                     nodes[item_root].lower,
                                     nodes[item_root].upper,
                                     &node.lower, &node.upper);
            nodes[next] = node;
            root = next++;
        }
    }
    if (root == UINT32_MAX) {
        wf_make_zero_node(nodes);
        root = 0;
        next = 1;
    }
    result = wf_wave_from_parts(nodes, next, root,
                                parameters, parameter_count);
    free(nodes);
    free(parameters);
done:
    for (index = 0; index < count; ++index)
        cwaveform_wave_release(waves[index]);
    return result;
}

static cwaveform_wave *wf_scale_owned(cwaveform_wave *wave, double scale,
                                      double epsilon) {
    cwaveform_wave *result;
    if (wave == NULL || !isfinite(scale)) {
        cwaveform_wave_release(wave);
        return NULL;
    }
    if (fabs(scale) <= epsilon) {
        cwaveform_wave_release(wave);
        return cwaveform_wave_constant(0.0);
    }
    if (scale == 1.0) return wave;
    if (wave->nodes[wave->root].op == WF_OP_CONSTANT) {
        double value = wave->nodes[wave->root].p0 * scale;
        cwaveform_wave_release(wave);
        return cwaveform_wave_constant(fabs(value) <= epsilon ? 0.0 : value);
    }
    if (wave->nodes[wave->root].op == WF_OP_SCALE) {
        const wf_node *node = wave->nodes + wave->root;
        cwaveform_wave *child = wf_subwave(wave, node->left);
        scale *= node->p0;
        cwaveform_wave_release(wave);
        return wf_scale_owned(child, scale, epsilon);
    }
    result = cwaveform_wave_materialize(wave, 0, scale);
    cwaveform_wave_release(wave);
    return result;
}

static cwaveform_wave *wf_simplify_associative(
    const cwaveform_wave *wave, uint32_t root, double epsilon,
    uint8_t operation) {
    uint32_t *roots;
    wf_canonical_term *terms;
    cwaveform_wave **items;
    size_t root_count = 0;
    size_t term_count = 0;
    size_t item_count = 0;
    size_t index;
    double scalar = operation == WF_OP_ADD ? 0.0 : 1.0;
    cwaveform_wave *result = NULL;
    roots = (uint32_t *)malloc((size_t)wave->node_count * sizeof(*roots));
    terms = (wf_canonical_term *)calloc(
        wave->node_count, sizeof(*terms));
    items = (cwaveform_wave **)calloc(
        (size_t)wave->node_count + 1, sizeof(*items));
    if (roots == NULL || terms == NULL || items == NULL
            || wf_collect_roots(wave, root, operation,
                                roots, &root_count) != 0) goto done;
    for (index = 0; index < root_count; ++index) {
        cwaveform_wave *item = wf_simplify_node(wave, roots[index], epsilon);
        double scale = 1.0;
        if (item == NULL) goto done;
        if (item->nodes[item->root].op == WF_OP_SCALE) {
            const wf_node *node = item->nodes + item->root;
            cwaveform_wave *child = wf_subwave(item, node->left);
            scale = node->p0;
            cwaveform_wave_release(item);
            item = child;
            if (item == NULL) goto done;
        }
        if (item->nodes[item->root].op == WF_OP_CONSTANT) {
            double value = scale * item->nodes[item->root].p0;
            cwaveform_wave_release(item);
            if (operation == WF_OP_ADD) scalar += value;
            else scalar *= value;
            if (operation == WF_OP_MUL && fabs(scalar) <= epsilon) {
                result = cwaveform_wave_constant(0.0);
                goto done;
            }
        } else {
            terms[term_count].wave = item;
            terms[term_count].scale = scale;
            ++term_count;
        }
    }
    qsort(terms, term_count, sizeof(*terms), wf_term_compare);
    for (index = 0; index < term_count;) {
        size_t stop = index + 1;
        if (operation == WF_OP_ADD) {
            double scale = terms[index].scale;
            while (stop < term_count
                    && wf_wave_byte_compare(terms[index].wave,
                                            terms[stop].wave) == 0) {
                scale += terms[stop].scale;
                cwaveform_wave_release(terms[stop].wave);
                terms[stop].wave = NULL;
                ++stop;
            }
            terms[index].wave = wf_scale_owned(
                terms[index].wave, scale, epsilon);
            if (terms[index].wave == NULL) goto done;
            if (terms[index].wave->nodes[terms[index].wave->root].op
                    == WF_OP_CONSTANT
                    && terms[index].wave->nodes[terms[index].wave->root].p0
                       == 0.0) {
                cwaveform_wave_release(terms[index].wave);
            } else {
                items[item_count++] = terms[index].wave;
            }
            terms[index].wave = NULL;
        } else {
            unsigned power = 1;
            scalar *= terms[index].scale;
            while (stop < term_count
                    && wf_wave_byte_compare(terms[index].wave,
                                            terms[stop].wave) == 0) {
                scalar *= terms[stop].scale;
                ++power;
                cwaveform_wave_release(terms[stop].wave);
                terms[stop].wave = NULL;
                ++stop;
            }
            if (power == 1) {
                items[item_count++] = terms[index].wave;
            } else {
                items[item_count] = cwaveform_wave_power(
                    terms[index].wave, (int)power);
                cwaveform_wave_release(terms[index].wave);
                if (items[item_count] == NULL) goto done;
                ++item_count;
            }
            terms[index].wave = NULL;
        }
        index = stop;
    }
    if (operation == WF_OP_ADD && fabs(scalar) > epsilon)
        items[item_count++] = cwaveform_wave_constant(scalar);
    if (item_count == 0) {
        result = cwaveform_wave_constant(
            operation == WF_OP_ADD ? 0.0 : scalar);
        goto done;
    }
    qsort(items, item_count, sizeof(*items), wf_wave_pointer_compare);
    result = wf_concat_owned(items, item_count, operation);
    for (index = 0; index < item_count; ++index) items[index] = NULL;
    if (operation == WF_OP_MUL)
        result = wf_scale_owned(result, scalar, epsilon);
done:
    if (terms != NULL) {
        for (index = 0; index < term_count; ++index)
            cwaveform_wave_release(terms[index].wave);
    }
    if (items != NULL) {
        for (index = 0; index < item_count; ++index)
            cwaveform_wave_release(items[index]);
    }
    free(roots);
    free(terms);
    free(items);
    return result;
}

static cwaveform_wave *wf_simplify_node(const cwaveform_wave *wave,
                                        uint32_t root, double epsilon) {
    const wf_node *node = wave->nodes + root;
    cwaveform_wave *child;
    cwaveform_wave *result;
    wf_node leaf;
    switch (node->op) {
        case WF_OP_ADD:
        case WF_OP_MUL:
            return wf_simplify_associative(
                wave, root, epsilon, node->op);
        case WF_OP_SCALE:
            child = wf_simplify_node(wave, node->left, epsilon);
            return wf_scale_owned(child, node->p0, epsilon);
        case WF_OP_POWER:
            child = wf_simplify_node(wave, node->left, epsilon);
            if (child == NULL) return NULL;
            if (node->p0 == 1.0) return child;
            if (child->nodes[child->root].op == WF_OP_CONSTANT) {
                double value = pow(child->nodes[child->root].p0, node->p0);
                cwaveform_wave_release(child);
                return cwaveform_wave_constant(
                    fabs(value) <= epsilon ? 0.0 : value);
            }
            result = cwaveform_wave_power(child, (int)node->p0);
            cwaveform_wave_release(child);
            return result;
        case WF_OP_WINDOW: {
            int64_t upper;
            child = wf_simplify_node(wave, node->left, epsilon);
            if (child == NULL) return NULL;
            memcpy(&upper, &node->p0, sizeof(upper));
            result = cwaveform_wave_window(child, node->shift, upper);
            cwaveform_wave_release(child);
            return result;
        }
        default:
            leaf = *node;
            leaf.left = 0;
            leaf.right = 0;
            leaf.parameter_offset = 0;
            if (leaf.op == WF_OP_CONSTANT && fabs(leaf.p0) <= epsilon)
                leaf.p0 = 0.0;
            return wf_wave_from_parts(
                &leaf, 1, 0,
                node->parameter_count == 0 ? NULL
                    : wave->parameters + node->parameter_offset,
                node->parameter_count);
    }
}

cwaveform_wave *cwaveform_wave_simplify(const cwaveform_wave *wave,
                                        double epsilon) {
    if (wave == NULL || !isfinite(epsilon) || epsilon < 0.0) return NULL;
    return wf_simplify_node(wave, wave->root, epsilon);
}

cwaveform_wave *cwaveform_wave_from_bytes(const uint8_t *data, size_t size) {
    cwaveform_wave *wave;
    uint32_t node_count;
    uint32_t root;
    uint32_t parameter_count;
    uint32_t index;
    if (data == NULL || size < WF_WAVE_HEADER_SIZE
            || memcmp(data, "WNF4", 4) != 0
            || wf_get_u16(data + 4) != WF_VERSION
            || wf_get_u16(data + 6) != 0
            || wf_get_u32(data + 20) != 0) {
        return NULL;
    }
    node_count = wf_get_u32(data + 8);
    root = wf_get_u32(data + 12);
    parameter_count = wf_get_u32(data + 16);
    if (node_count == 0 || root >= node_count
            || (size_t)node_count > (SIZE_MAX - WF_WAVE_HEADER_SIZE) / WF_NODE_SIZE
            || (size_t)parameter_count > (SIZE_MAX - WF_WAVE_HEADER_SIZE
                - (size_t)node_count * WF_NODE_SIZE) / sizeof(double)
            || size != WF_WAVE_HEADER_SIZE + (size_t)node_count * WF_NODE_SIZE
                       + (size_t)parameter_count * sizeof(double)) {
        return NULL;
    }
    wave = (cwaveform_wave *)calloc(1, sizeof(*wave));
    if (wave == NULL) {
        return NULL;
    }
    wave->data = (uint8_t *)malloc(size);
    wave->nodes = (wf_node *)malloc((size_t)node_count * sizeof(*wave->nodes));
    wave->parameters = parameter_count == 0 ? NULL
        : (double *)malloc((size_t)parameter_count * sizeof(double));
    if (wave->data == NULL || wave->nodes == NULL
            || (parameter_count != 0 && wave->parameters == NULL)) {
        cwaveform_wave_release(wave);
        return NULL;
    }
    memcpy(wave->data, data, size);
    for (index = 0; index < parameter_count; ++index) {
        wave->parameters[index] = wf_get_f64(
            data + WF_WAVE_HEADER_SIZE + (size_t)node_count * WF_NODE_SIZE
            + (size_t)index * sizeof(double));
        if (!isfinite(wave->parameters[index])) {
            /* DRAG uses one NaN sentinel for a missing block frequency. */
            if (!isnan(wave->parameters[index])) {
                cwaveform_wave_release(wave);
                return NULL;
            }
        }
    }
    for (index = 0; index < node_count; ++index) {
        wf_decode_node(wave->nodes + index,
                       data + WF_WAVE_HEADER_SIZE + (size_t)index * WF_NODE_SIZE);
        if (wf_prepare_decoded_node(wave->nodes + index, wave->nodes,
                                    index, parameter_count) != 0) {
            cwaveform_wave_release(wave);
            return NULL;
        }
    }
    wave->references = 1;
    wave->node_count = node_count;
    wave->root = root;
    wave->parameter_count = parameter_count;
    wave->data_size = size;
    wave->hash = wf_hash_bytes(data, size);
    if (wf_wave_prepare_drag_sin_kernels(wave) != 0) {
        cwaveform_wave_release(wave);
        return NULL;
    }
    return wave;
}

static uint32_t wf_clone_affine(const cwaveform_wave *source, wf_node *target,
                                uint32_t offset, uint32_t parameter_offset,
                                int64_t delay, double scale, uint32_t *next) {
    uint32_t index;
    uint32_t root;
    for (index = 0; index < source->node_count; ++index) {
        wf_node node = source->nodes[index];
        if (node.op == WF_OP_ADD || node.op == WF_OP_MUL) {
            node.left += offset;
            node.right += offset;
        } else if (node.op == WF_OP_SCALE || node.op == WF_OP_POWER
                   || node.op == WF_OP_WINDOW) {
            node.left += offset;
        }
        if (node.parameter_count != 0)
            node.parameter_offset += parameter_offset;
        if (node.op == WF_OP_WINDOW) {
            int64_t upper;
            memcpy(&upper, &node.p0, sizeof(upper));
            upper = wf_add_tick(upper, delay);
            memcpy(&node.p0, &upper, sizeof(upper));
            node.shift = wf_add_tick(node.shift, delay);
        } else if (node.op != WF_OP_CONSTANT
                   && node.op != WF_OP_ADD && node.op != WF_OP_MUL
                   && node.op != WF_OP_SCALE && node.op != WF_OP_POWER) {
            node.shift = wf_add_tick(node.shift, delay);
        }
        node.lower = wf_add_tick(node.lower, delay);
        node.upper = wf_add_tick(node.upper, delay);
        target[offset + index] = node;
    }
    *next = offset + source->node_count;
    root = offset + source->root;
    if (scale != 1.0) {
        wf_node node;
        memset(&node, 0, sizeof(node));
        node.op = WF_OP_SCALE;
        node.left = root;
        node.p0 = scale;
        node.lower = scale == 0.0 ? INT64_MAX : target[root].lower;
        node.upper = scale == 0.0 ? INT64_MIN : target[root].upper;
        target[*next] = node;
        root = *next;
        ++*next;
    }
    return root;
}

static cwaveform_wave *wf_combine_affine(
    const cwaveform_wave *left, int64_t left_delay, double left_scale,
    const cwaveform_wave *right, int64_t right_delay, double right_scale,
    uint8_t operation) {
    uint32_t capacity;
    uint32_t next = 0;
    uint32_t left_root;
    uint32_t right_root;
    wf_node *nodes;
    double *parameters;
    size_t parameter_count;
    wf_node node;
    cwaveform_wave *result;
    if (left == NULL || right == NULL || !isfinite(left_scale)
            || !isfinite(right_scale)) {
        return NULL;
    }
    /* Canonical operand order makes commutative expressions byte-identical
     * without requiring a second normalization pass. */
    {
        int swap = 0;
        if (left->hash > right->hash) swap = 1;
        else if (left->hash == right->hash && left_delay > right_delay) swap = 1;
        else if (left->hash == right->hash && left_delay == right_delay
                 && left_scale > right_scale) swap = 1;
        if (swap) {
            const cwaveform_wave *temporary_wave = left;
            int64_t temporary_delay = left_delay;
            double temporary_scale = left_scale;
            left = right;
            left_delay = right_delay;
            left_scale = right_scale;
            right = temporary_wave;
            right_delay = temporary_delay;
            right_scale = temporary_scale;
        }
    }
    if (left->node_count > UINT32_MAX - right->node_count - 3) {
        return NULL;
    }
    capacity = left->node_count + right->node_count + 3;
    if (left->parameter_count > SIZE_MAX - right->parameter_count)
        return NULL;
    parameter_count = left->parameter_count + right->parameter_count;
    nodes = (wf_node *)calloc(capacity, sizeof(*nodes));
    parameters = parameter_count == 0 ? NULL
        : (double *)malloc(parameter_count * sizeof(double));
    if (nodes == NULL || (parameter_count != 0 && parameters == NULL)) {
        free(nodes);
        free(parameters);
        return NULL;
    }
    if (left->parameter_count != 0)
        memcpy(parameters, left->parameters,
               left->parameter_count * sizeof(double));
    if (right->parameter_count != 0)
        memcpy(parameters + left->parameter_count, right->parameters,
               right->parameter_count * sizeof(double));
    left_root = wf_clone_affine(left, nodes, 0, 0, left_delay, left_scale,
                                &next);
    right_root = wf_clone_affine(
        right, nodes, next, (uint32_t)left->parameter_count,
        right_delay, right_scale, &next);
    memset(&node, 0, sizeof(node));
    node.op = operation;
    node.left = left_root;
    node.right = right_root;
    if (operation == WF_OP_ADD) {
        wf_union_support(nodes[left_root].lower, nodes[left_root].upper,
                         nodes[right_root].lower, nodes[right_root].upper,
                         &node.lower, &node.upper);
    } else {
        wf_intersect_support(nodes[left_root].lower, nodes[left_root].upper,
                             nodes[right_root].lower, nodes[right_root].upper,
                             &node.lower, &node.upper);
    }
    nodes[next] = node;
    result = wf_wave_from_parts(nodes, next + 1, next,
                                parameters, parameter_count);
    free(nodes);
    free(parameters);
    return result;
}

cwaveform_wave *cwaveform_wave_add_affine(
    const cwaveform_wave *left, int64_t left_delay, double left_scale,
    const cwaveform_wave *right, int64_t right_delay, double right_scale) {
    if (left == NULL || right == NULL) return NULL;
    if (left_scale == 0.0
            || (left->nodes[left->root].op == WF_OP_CONSTANT
                && left->nodes[left->root].p0 == 0.0))
        return cwaveform_wave_materialize(right, right_delay, right_scale);
    if (right_scale == 0.0
            || (right->nodes[right->root].op == WF_OP_CONSTANT
                && right->nodes[right->root].p0 == 0.0))
        return cwaveform_wave_materialize(left, left_delay, left_scale);
    if (cwaveform_wave_equal(left, right) && left_delay == right_delay)
        return cwaveform_wave_materialize(
            left, left_delay, left_scale + right_scale);
    if (left->nodes[left->root].op == WF_OP_CONSTANT
            && right->nodes[right->root].op == WF_OP_CONSTANT)
        return cwaveform_wave_constant(left_scale * left->nodes[left->root].p0
                                       + right_scale * right->nodes[right->root].p0);
    return wf_combine_affine(left, left_delay, left_scale, right, right_delay,
                             right_scale, WF_OP_ADD);
}

cwaveform_wave *cwaveform_wave_mul_affine(
    const cwaveform_wave *left, int64_t left_delay, double left_scale,
    const cwaveform_wave *right, int64_t right_delay, double right_scale) {
    if (left == NULL || right == NULL) return NULL;
    if (left_scale == 0.0 || right_scale == 0.0
            || (left->nodes[left->root].op == WF_OP_CONSTANT
                && left->nodes[left->root].p0 == 0.0)
            || (right->nodes[right->root].op == WF_OP_CONSTANT
                && right->nodes[right->root].p0 == 0.0))
        return cwaveform_wave_constant(0.0);
    if (left->nodes[left->root].op == WF_OP_CONSTANT) {
        return cwaveform_wave_materialize(
            right, right_delay,
            left_scale * right_scale * left->nodes[left->root].p0);
    }
    if (right->nodes[right->root].op == WF_OP_CONSTANT) {
        return cwaveform_wave_materialize(
            left, left_delay,
            left_scale * right_scale * right->nodes[right->root].p0);
    }
    if (cwaveform_wave_equal(left, right) && left_delay == right_delay) {
        cwaveform_wave *materialized = cwaveform_wave_materialize(
            left, left_delay, 1.0);
        cwaveform_wave *powered;
        if (materialized == NULL) return NULL;
        powered = cwaveform_wave_power(materialized, 2);
        cwaveform_wave_release(materialized);
        if (powered == NULL) return NULL;
        materialized = cwaveform_wave_materialize(
            powered, 0, left_scale * right_scale);
        cwaveform_wave_release(powered);
        return materialized;
    }
    return wf_combine_affine(left, left_delay, left_scale, right, right_delay,
                             right_scale, WF_OP_MUL);
}

cwaveform_wave *cwaveform_wave_materialize(const cwaveform_wave *wave,
                                            int64_t delay, double scale) {
    wf_node *nodes;
    double *parameters;
    uint32_t next = 0;
    uint32_t root;
    cwaveform_wave *result;
    if (wave == NULL || !isfinite(scale)) {
        return NULL;
    }
    nodes = (wf_node *)calloc((size_t)wave->node_count + 1, sizeof(*nodes));
    parameters = wave->parameter_count == 0 ? NULL
        : (double *)malloc(wave->parameter_count * sizeof(double));
    if (nodes == NULL || (wave->parameter_count != 0 && parameters == NULL)) {
        free(nodes);
        free(parameters);
        return NULL;
    }
    if (wave->parameter_count != 0)
        memcpy(parameters, wave->parameters,
               wave->parameter_count * sizeof(double));
    root = wf_clone_affine(wave, nodes, 0, 0, delay, scale, &next);
    result = wf_wave_from_parts(nodes, next, root, parameters,
                                wave->parameter_count);
    free(nodes);
    free(parameters);
    return result;
}

void cwaveform_wave_retain(cwaveform_wave *wave) {
    if (wave != NULL) {
        ++wave->references;
    }
}

void cwaveform_wave_release(cwaveform_wave *wave) {
    if (wave == NULL) {
        return;
    }
    if (wave->references > 1) {
        --wave->references;
        return;
    }
    free(wave->data);
    free(wave->nodes);
    free(wave->parameters);
    wf_wave_clear_drag_sin_kernels(wave);
    free(wave);
}

const uint8_t *cwaveform_wave_bytes(const cwaveform_wave *wave, size_t *size) {
    if (wave == NULL) {
        return NULL;
    }
    if (size != NULL) {
        *size = wave->data_size;
    }
    return wave->data;
}

uint64_t cwaveform_wave_hash(const cwaveform_wave *wave) {
    return wave == NULL ? 0 : wave->hash;
}

int cwaveform_wave_equal(const cwaveform_wave *left,
                         const cwaveform_wave *right) {
    return left != NULL && right != NULL && left->hash == right->hash
        && left->data_size == right->data_size
        && memcmp(left->data, right->data, left->data_size) == 0;
}

int64_t cwaveform_wave_lower_tick(const cwaveform_wave *wave) {
    return wave == NULL ? INT64_MAX : wave->nodes[wave->root].lower;
}

int64_t cwaveform_wave_upper_tick(const cwaveform_wave *wave) {
    return wave == NULL ? INT64_MIN : wave->nodes[wave->root].upper;
}

uint32_t cwaveform_wave_node_count(const cwaveform_wave *wave) {
    return wave == NULL ? 0 : wave->node_count;
}

static double wf_node_parameter(const cwaveform_wave *wave,
                                const wf_node *node, size_t index) {
    if (index == 0) return node->p0;
    return wave->parameters[node->parameter_offset + index - 1];
}

static double wf_hermite(unsigned order, double x) {
    unsigned index;
    double previous = 1.0;
    double current;
    if (order == 0) return 1.0;
    current = 2.0 * x;
    for (index = 1; index < order; ++index) {
        double next = 2.0 * x * current - 2.0 * (double)index * previous;
        previous = current;
        current = next;
    }
    return current;
}

static double wf_mollifier_value(double local, double radius,
                                  unsigned derivative) {
    double x;
    double xx_1;
    double envelope;
    unsigned n;
    size_t degree;
    double *coefficients;
    if (!(radius > 0.0) || derivative > 64) return NAN;
    x = local / radius;
    xx_1 = x * x - 1.0;
    if (xx_1 >= 0.0) return 0.0;
    envelope = exp(1.0 / xx_1 + 1.0);
    if (derivative == 0) return envelope;
    /* Ascending polynomial coefficients.  P1(x)=-2x and
     * P(n+1)=(x^2-1)^2 Pn' + (-4n x^3 +(4n-2)x)Pn. */
    coefficients = (double *)calloc((size_t)3 * derivative + 2,
                                     sizeof(double));
    if (coefficients == NULL) return NAN;
    coefficients[1] = -2.0;
    degree = 1;
    for (n = 1; n < derivative; ++n) {
        size_t i;
        size_t next_degree = degree + 3;
        double *next = (double *)calloc((size_t)3 * derivative + 2,
                                        sizeof(double));
        if (next == NULL) {
            free(coefficients);
            return NAN;
        }
        for (i = 1; i <= degree; ++i) {
            double value = (double)i * coefficients[i];
            next[i - 1] += value;
            next[i + 1] -= 2.0 * value;
            next[i + 3] += value;
        }
        for (i = 0; i <= degree; ++i) {
            next[i + 1] += (4.0 * n - 2.0) * coefficients[i];
            next[i + 3] -= 4.0 * n * coefficients[i];
        }
        free(coefficients);
        coefficients = next;
        degree = next_degree;
    }
    {
        double polynomial = coefficients[degree];
        size_t i = degree;
        while (i-- != 0) polynomial = polynomial * x + coefficients[i];
        free(coefficients);
        return envelope * polynomial
            / (pow(-xx_1, 2.0 * derivative) * pow(radius, derivative));
    }
}

static int wf_solve_linear(double *matrix, double *values, size_t size) {
    size_t column;
    for (column = 0; column < size; ++column) {
        size_t pivot = column;
        size_t row;
        double maximum = fabs(matrix[column * size + column]);
        for (row = column + 1; row < size; ++row) {
            double candidate = fabs(matrix[row * size + column]);
            if (candidate > maximum) {
                maximum = candidate;
                pivot = row;
            }
        }
        if (!(maximum > 0.0)) return -1;
        if (pivot != column) {
            size_t item;
            for (item = column; item < size; ++item) {
                double temporary = matrix[column * size + item];
                matrix[column * size + item] = matrix[pivot * size + item];
                matrix[pivot * size + item] = temporary;
            }
            {
                double temporary = values[column];
                values[column] = values[pivot];
                values[pivot] = temporary;
            }
        }
        for (row = column + 1; row < size; ++row) {
            double factor = matrix[row * size + column]
                / matrix[column * size + column];
            size_t item;
            for (item = column; item < size; ++item)
                matrix[row * size + item]
                    -= factor * matrix[column * size + item];
            values[row] -= factor * values[column];
        }
    }
    while (column-- != 0) {
        size_t item;
        for (item = column + 1; item < size; ++item)
            values[column] -= matrix[column * size + item] * values[item];
        values[column] /= matrix[column * size + column];
    }
    return 0;
}

static double wf_factorial_ratio(size_t high, size_t low) {
    double value = 1.0;
    size_t item;
    for (item = low + 1; item <= high; ++item) value *= (double)item;
    return value;
}

static int wf_edge_polynomial(const double *derivatives, size_t size,
                              double x, double *ascending) {
    double *matrix;
    double *values;
    size_t derivative;
    size_t coefficient;
    matrix = (double *)malloc(size * size * sizeof(double));
    values = (double *)malloc(size * sizeof(double));
    if (matrix == NULL || values == NULL) {
        free(matrix);
        free(values);
        return -1;
    }
    memcpy(values, derivatives, size * sizeof(double));
    values[0] -= 1.0;
    for (derivative = 0; derivative < size; ++derivative) {
        for (coefficient = 0; coefficient < size; ++coefficient) {
            size_t degree = size + coefficient;
            matrix[derivative * size + coefficient] =
                pow(x, (double)(degree - derivative))
                * wf_factorial_ratio(degree, degree - derivative);
        }
    }
    if (wf_solve_linear(matrix, values, size) != 0) {
        free(matrix);
        free(values);
        return -1;
    }
    memset(ascending, 0, (2 * size) * sizeof(double));
    ascending[0] = 1.0;
    for (coefficient = 0; coefficient < size; ++coefficient)
        ascending[size + coefficient] = values[coefficient];
    free(matrix);
    free(values);
    return 0;
}

static double wf_polynomial_derivative_value(const double *coefficients,
                                             size_t count,
                                             size_t derivative, double x) {
    size_t degree;
    double value = 0.0;
    if (derivative >= count) return 0.0;
    degree = count;
    while (degree-- > derivative) {
        value = value * x
            + coefficients[degree] * wf_factorial_ratio(
                degree, degree - derivative);
    }
    return value;
}

static void wf_drag_sin_kernel_free(wf_drag_sin_kernel *kernel) {
    if (kernel == NULL) return;
    free(kernel->transform);
    free(kernel->derivatives);
    free(kernel->left_polynomial);
    free(kernel->right_polynomial);
    free(kernel);
}

static wf_drag_sin_kernel *wf_drag_sin_kernel_create(
    const cwaveform_wave *wave, const wf_node *node, int sinx) {
    size_t total = (size_t)node->parameter_count + 1;
    size_t block_count = total - 7;
    size_t order = block_count + 1;
    size_t power = ((block_count + 2) >> 1) << 1;
    size_t basis_count;
    double width = wf_node_parameter(wave, node, 2);
    double delta = wf_node_parameter(wave, node, 3);
    double tab = wf_node_parameter(wave, node, 6);
    wf_drag_sin_kernel *kernel;
    double basis[67];
    double values[65];
    size_t index;
    if (!(width > 0.0) || block_count > 64) return NULL;
    if (power < 2) power = 2;
    basis_count = power + 1;
    kernel = (wf_drag_sin_kernel *)calloc(1, sizeof(*kernel));
    if (kernel == NULL) return NULL;
    kernel->order = order;
    kernel->power = power;
    kernel->basis_count = basis_count;
    kernel->angular = 3.14159265358979323846 / width;
    kernel->normalization = 1.0;
    kernel->transform = (double *)calloc(order * 4, sizeof(double));
    kernel->derivatives = (double *)calloc(
        order * basis_count, sizeof(double));
    if (sinx) {
        kernel->left_polynomial = (double *)calloc(
            2 * order, sizeof(double));
        kernel->right_polynomial = (double *)calloc(
            2 * order, sizeof(double));
    }
    if (kernel->transform == NULL || kernel->derivatives == NULL
            || (sinx && (kernel->left_polynomial == NULL
                         || kernel->right_polynomial == NULL))) {
        wf_drag_sin_kernel_free(kernel);
        return NULL;
    }
    kernel->transform[0] = 1.0;
    kernel->transform[3] = 1.0;
    for (index = 0; index < block_count; ++index) {
        double block_frequency = wf_node_parameter(wave, node, 7 + index);
        double coefficient = 1.0 / (2.0 * 3.14159265358979323846
                                    * (block_frequency - delta));
        size_t row = index + 1;
        while (row-- != 0) {
            double *target = kernel->transform + (row + 1) * 4;
            const double *source = kernel->transform + row * 4;
            target[0] += -coefficient * source[1];
            target[1] += coefficient * source[0];
            target[2] += -coefficient * source[3];
            target[3] += coefficient * source[2];
        }
    }
    kernel->derivatives[power] = 1.0;
    for (index = 1; index < order; ++index) {
        size_t exponent;
        if (index & 1) {
            for (exponent = 0; exponent < power; ++exponent)
                kernel->derivatives[index * basis_count + exponent] =
                    kernel->derivatives[
                        (index - 1) * basis_count + exponent + 1]
                    * (double)(exponent + 1) * kernel->angular;
        } else {
            for (exponent = 0; exponent + 2 <= power; ++exponent)
                kernel->derivatives[index * basis_count + exponent] =
                    kernel->derivatives[
                        (index - 2) * basis_count + exponent + 2]
                    * (double)(exponent + 1) * (double)(exponent + 2);
            for (exponent = 0; exponent <= power; ++exponent)
                kernel->derivatives[index * basis_count + exponent] -=
                    kernel->derivatives[
                        (index - 2) * basis_count + exponent]
                    * (double)(exponent * exponent);
            for (exponent = 0; exponent <= power; ++exponent)
                kernel->derivatives[index * basis_count + exponent]
                    *= kernel->angular * kernel->angular;
        }
    }

    if (sinx) {
        int side;
        for (side = 0; side < 2; ++side) {
            double boundary_angle = kernel->angular
                * ((side == 0 ? (1.0 - tab) : (1.0 + tab))
                   * width / 2.0);
            double boundary_sine = sin(boundary_angle);
            double boundary_cosine = cos(boundary_angle);
            double edge_x = (side == 0 ? -tab : tab) * width / 2.0;
            double *polynomial = side == 0
                ? kernel->left_polynomial : kernel->right_polynomial;
            double sine_power = 1.0;
            size_t exponent;
            memset(values, 0, order * sizeof(double));
            for (exponent = 0; exponent <= power; ++exponent) {
                basis[exponent] = sine_power;
                if (exponent & 1) basis[exponent] *= boundary_cosine;
                sine_power *= boundary_sine;
            }
            for (index = 0; index < order; ++index)
                for (exponent = 0; exponent <= power; ++exponent)
                    values[index] += kernel->derivatives[
                        index * basis_count + exponent] * basis[exponent];
            if (wf_edge_polynomial(
                    values, order, edge_x, polynomial) != 0) {
                wf_drag_sin_kernel_free(kernel);
                return NULL;
            }
        }
    } else {
        double components[2] = {0.0, 0.0};
        size_t exponent;
        memset(basis, 0, basis_count * sizeof(double));
        memset(values, 0, order * sizeof(double));
        for (exponent = 0; exponent <= power; exponent += 2)
            basis[exponent] = 1.0;
        for (index = 0; index < order; ++index)
            for (exponent = 0; exponent <= power; ++exponent)
                values[index] += kernel->derivatives[
                    index * basis_count + exponent] * basis[exponent];
        for (index = 0; index < order; ++index) {
            components[0] += kernel->transform[index * 4] * values[index];
            components[1] += kernel->transform[index * 4 + 2]
                * values[index];
        }
        kernel->normalization = wf_hypot(components[0], components[1]);
    }
    return kernel;
}

static int wf_wave_prepare_drag_sin_kernels(cwaveform_wave *wave) {
    uint32_t index;
    int needed = 0;
    for (index = 0; index < wave->node_count; ++index) {
        uint8_t op = wave->nodes[index].op;
        if (op == WF_OP_DRAG_SIN || op == WF_OP_DRAG_SINX) {
            needed = 1;
            break;
        }
    }
    if (!needed) return 0;
    wave->drag_sin_kernels = (wf_drag_sin_kernel **)calloc(
        wave->node_count, sizeof(*wave->drag_sin_kernels));
    if (wave->drag_sin_kernels == NULL) return -1;
    for (index = 0; index < wave->node_count; ++index) {
        const wf_node *node = wave->nodes + index;
        if (node->op != WF_OP_DRAG_SIN && node->op != WF_OP_DRAG_SINX)
            continue;
        wave->drag_sin_kernels[index] = wf_drag_sin_kernel_create(
            wave, node, node->op == WF_OP_DRAG_SINX);
        if (wave->drag_sin_kernels[index] == NULL) {
            wf_wave_clear_drag_sin_kernels(wave);
            return -1;
        }
    }
    return 0;
}

static void wf_wave_clear_drag_sin_kernels(cwaveform_wave *wave) {
    uint32_t index;
    if (wave->drag_sin_kernels == NULL) return;
    for (index = 0; index < wave->node_count; ++index)
        wf_drag_sin_kernel_free(wave->drag_sin_kernels[index]);
    free(wave->drag_sin_kernels);
    wave->drag_sin_kernels = NULL;
}

static double wf_drag_sin_value(const cwaveform_wave *wave,
                                uint32_t node_index,
                                const wf_node *node, double local,
                                int sinx) {
    const wf_drag_sin_kernel *kernel = wave->drag_sin_kernels == NULL
        ? NULL : wave->drag_sin_kernels[node_index];
    double t0 = wf_node_parameter(wave, node, 0);
    double frequency = wf_node_parameter(wave, node, 1);
    double width = wf_node_parameter(wave, node, 2);
    double delta = wf_node_parameter(wave, node, 3);
    double phase = wf_node_parameter(wave, node, 4);
    double plateau = wf_node_parameter(wave, node, 5);
    double tab = wf_node_parameter(wave, node, 6);
    double basis[67];
    double values[65];
    double components[2] = {0.0, 0.0};
    double midpoint;
    double plateau_stop;
    size_t index;
    if (kernel == NULL) return NAN;
    midpoint = t0 + width / 2.0;
    plateau_stop = midpoint + plateau;
    memset(values, 0, kernel->order * sizeof(double));
    if (sinx) {
        int left = local >= midpoint - tab * width / 2.0
            && local <= midpoint;
        int right = local >= plateau_stop
            && local <= plateau_stop + tab * width / 2.0;
        if (left || right) {
            double x = left ? local - midpoint : local - plateau_stop;
            const double *polynomial = left
                ? kernel->left_polynomial : kernel->right_polynomial;
            for (index = 0; index < kernel->order; ++index)
                values[index] = wf_polynomial_derivative_value(
                    polynomial, 2 * kernel->order, index, x);
        } else {
            double adjusted = local >= plateau_stop
                ? local - plateau : local;
            double angle = kernel->angular * (adjusted - t0);
            double sine = sin(angle);
            double cosine = cos(angle);
            double sine_power = 1.0;
            int in_plateau = local > midpoint && local < plateau_stop;
            size_t exponent;
            for (exponent = 0; exponent <= kernel->power; ++exponent) {
                basis[exponent] = sine_power;
                if (exponent & 1) basis[exponent] *= cosine;
                if (in_plateau) basis[exponent] = 0.0;
                sine_power *= sine;
            }
            for (index = 0; index < kernel->order; ++index)
                for (exponent = 0; exponent <= kernel->power; ++exponent)
                    values[index] += kernel->derivatives[
                        index * kernel->basis_count + exponent]
                        * basis[exponent];
            if (in_plateau) values[0] = 1.0;
        }
    } else {
        double adjusted = local >= plateau_stop ? local - plateau : local;
        double angle = kernel->angular * (adjusted - t0);
        double sine = sin(angle);
        double cosine = cos(angle);
        double sine_power = 1.0;
        int in_plateau = local > midpoint && local < plateau_stop;
        size_t exponent;
        for (exponent = 0; exponent <= kernel->power; ++exponent) {
            basis[exponent] = sine_power;
            if (exponent & 1) basis[exponent] *= cosine;
            if (in_plateau) basis[exponent] = 0.0;
            sine_power *= sine;
        }
        for (index = 0; index < kernel->order; ++index)
            for (exponent = 0; exponent <= kernel->power; ++exponent)
                values[index] += kernel->derivatives[
                    index * kernel->basis_count + exponent]
                    * basis[exponent];
        if (in_plateau) values[0] = 1.0;
    }
    for (index = 0; index < kernel->order; ++index) {
        components[0] += kernel->transform[index * 4] * values[index];
        components[1] += kernel->transform[index * 4 + 2] * values[index];
    }
    components[0] /= kernel->normalization;
    components[1] /= kernel->normalization;
    {
        double carrier = 2.0 * 3.14159265358979323846
            * (frequency + delta) * local
            - (2.0 * 3.14159265358979323846 * delta * t0 + phase);
        return components[0] * cos(carrier) + components[1] * sin(carrier);
    }
}

static double wf_evaluate_one(const cwaveform_wave *wave, double position,
                              double *values) {
    uint32_t index;
    const double ticks = (double)wf_ticks_per_second;
    for (index = 0; index < wave->node_count; ++index) {
        const wf_node *node = wave->nodes + index;
        double lower = node->lower == INT64_MIN ? -DBL_MAX
            : (double)node->lower / ticks;
        double upper = node->upper == INT64_MAX ? DBL_MAX
            : (double)node->upper / ticks;
        double local;
        if (node->lower >= node->upper || position < lower || position >= upper) {
            values[index] = 0.0;
            continue;
        }
        switch (node->op) {
            case WF_OP_CONSTANT:
                values[index] = node->p0;
                break;
            case WF_OP_GAUSSIAN:
                local = position - (double)node->shift / ticks;
                local /= node->flags ? node->p0 : node->p1;
                values[index] = exp(-(local * local));
                break;
            case WF_OP_COS:
                local = position - (double)node->shift / ticks;
                values[index] = cos(node->p0 * local + node->p1);
                break;
            case WF_OP_SIN:
                local = position - (double)node->shift / ticks;
                values[index] = sin(node->p0 * local + node->p1);
                break;
            case WF_OP_SQUARE:
                values[index] = 1.0;
                break;
            case WF_OP_LINEAR:
                values[index] = position - (double)node->shift / ticks;
                break;
            case WF_OP_ERF:
                local = position - (double)node->shift / ticks;
                values[index] = erf(local / node->p0);
                break;
            case WF_OP_SINC:
                local = 3.14159265358979323846 * node->p0
                    * (position - (double)node->shift / ticks);
                values[index] = local == 0.0 ? 1.0 : sin(local) / local;
                break;
            case WF_OP_EXP:
                local = position - (double)node->shift / ticks;
                values[index] = exp(node->p0 * local);
                break;
            case WF_OP_INTERP: {
                size_t total = (size_t)node->parameter_count + 1;
                size_t point_count = total - 2;
                double start = node->p0;
                double stop = wf_node_parameter(wave, node, 1);
                double coordinate = (position - (double)node->shift / ticks
                                     - start) * (double)(point_count - 1)
                    / (stop - start);
                if (coordinate <= 0.0) {
                    values[index] = wf_node_parameter(wave, node, 2);
                } else if (coordinate >= (double)(point_count - 1)) {
                    values[index] = wf_node_parameter(
                        wave, node, point_count + 1);
                } else {
                    size_t left = (size_t)floor(coordinate);
                    double fraction = coordinate - (double)left;
                    double a = wf_node_parameter(wave, node, left + 2);
                    double b = wf_node_parameter(wave, node, left + 3);
                    values[index] = a + fraction * (b - a);
                }
                break;
            }
            case WF_OP_LINEAR_CHIRP: {
                double f0 = node->p0;
                double f1 = wf_node_parameter(wave, node, 1);
                double duration = wf_node_parameter(wave, node, 2);
                double phase = wf_node_parameter(wave, node, 3);
                local = position - (double)node->shift / ticks;
                values[index] = sin(phase + 2.0 * 3.14159265358979323846
                    * ((f1 - f0) / (2.0 * duration) * local * local
                       + f0 * local));
                break;
            }
            case WF_OP_EXPONENTIAL_CHIRP: {
                double f0 = node->p0;
                double alpha = wf_node_parameter(wave, node, 1);
                double phase = wf_node_parameter(wave, node, 2);
                local = position - (double)node->shift / ticks;
                values[index] = sin(phase + 2.0 * 3.14159265358979323846
                    * f0 * (exp(alpha * local) - 1.0) / alpha);
                break;
            }
            case WF_OP_HYPERBOLIC_CHIRP: {
                double f0 = node->p0;
                double k = wf_node_parameter(wave, node, 1);
                double phase = wf_node_parameter(wave, node, 2);
                local = position - (double)node->shift / ticks;
                values[index] = sin(phase + 2.0 * 3.14159265358979323846
                    * f0 / k * log(1.0 + k * local));
                break;
            }
            case WF_OP_COSH:
                local = position - (double)node->shift / ticks;
                values[index] = cosh(node->p0 * local);
                break;
            case WF_OP_SINH:
                local = position - (double)node->shift / ticks;
                values[index] = sinh(node->p0 * local);
                break;
            case WF_OP_DRAG: {
                double t0 = node->p0;
                double frequency = wf_node_parameter(wave, node, 1);
                double width = wf_node_parameter(wave, node, 2);
                double delta = wf_node_parameter(wave, node, 3);
                double block = wf_node_parameter(wave, node, 4);
                double phase = wf_node_parameter(wave, node, 5);
                double t = position - (double)node->shift / ticks;
                double omega = 3.14159265358979323846 / width;
                double omega_x = sin(omega * (t - t0));
                double carrier = 2.0 * 3.14159265358979323846
                    * (frequency + delta) * t
                    - (2.0 * 3.14159265358979323846 * delta * t0 + phase);
                omega_x *= omega_x;
                if (isnan(block) || block - delta == 0.0) {
                    values[index] = omega_x * cos(carrier);
                } else {
                    double b = 1.0 / (2.0 * 3.14159265358979323846
                                      * (block - delta));
                    double omega_y = -b * omega
                        * sin(2.0 * omega * (t - t0));
                    values[index] = omega_x * cos(carrier)
                        + omega_y * sin(carrier);
                }
                break;
            }
            case WF_OP_MOLLIFIER:
                local = position - (double)node->shift / ticks;
                values[index] = wf_mollifier_value(
                    local, node->p0,
                    (unsigned)llround(wf_node_parameter(wave, node, 1)));
                break;
            case WF_OP_D_GAUSSIAN: {
                double std = node->p0;
                unsigned order = (unsigned)llround(
                    wf_node_parameter(wave, node, 1));
                double x = (position - (double)node->shift / ticks) / std;
                values[index] = ((order & 1) ? -1.0 : 1.0)
                    * wf_hermite(order, x) * exp(-(x * x))
                    / pow(std, (double)order);
                break;
            }
            case WF_OP_DRAG_SIN:
                local = position - (double)node->shift / ticks;
                values[index] = wf_drag_sin_value(
                    wave, index, node, local, 0);
                break;
            case WF_OP_DRAG_SINX:
                local = position - (double)node->shift / ticks;
                values[index] = wf_drag_sin_value(
                    wave, index, node, local, 1);
                break;
            case WF_OP_ADD:
                values[index] = values[node->left] + values[node->right];
                break;
            case WF_OP_MUL:
                values[index] = values[node->left] * values[node->right];
                break;
            case WF_OP_SCALE:
                values[index] = node->p0 * values[node->left];
                break;
            case WF_OP_POWER:
                values[index] = pow(values[node->left], node->p0);
                break;
            case WF_OP_WINDOW:
                values[index] = values[node->left];
                break;
            default:
                values[index] = NAN;
                break;
        }
    }
    return values[wave->root];
}

#if (defined(__APPLE__) || defined(WF_HAVE_X86_SIMD) || defined(_WIN32)) \
        && !defined(WF_DISABLE_BATCH_EVALUATOR)
#if defined(__APPLE__)
static void wf_vector_exp(double *output, const double *input, size_t count) {
    while (count != 0) {
        int batch = count > (size_t)INT_MAX ? INT_MAX : (int)count;
        vvexp(output, input, &batch);
        output += batch;
        input += batch;
        count -= (size_t)batch;
    }
}

static void wf_vector_cos(double *output, const double *input, size_t count) {
    while (count != 0) {
        int batch = count > (size_t)INT_MAX ? INT_MAX : (int)count;
        vvcos(output, input, &batch);
        output += batch;
        input += batch;
        count -= (size_t)batch;
    }
}

static void wf_vector_sin(double *output, const double *input, size_t count) {
    while (count != 0) {
        int batch = count > (size_t)INT_MAX ? INT_MAX : (int)count;
        vvsin(output, input, &batch);
        output += batch;
        input += batch;
        count -= (size_t)batch;
    }
}

static void wf_vector_cosh(double *output, const double *input, size_t count) {
    while (count != 0) {
        int batch = count > (size_t)INT_MAX ? INT_MAX : (int)count;
        vvcosh(output, input, &batch);
        output += batch;
        input += batch;
        count -= (size_t)batch;
    }
}

static void wf_vector_sinh(double *output, const double *input, size_t count) {
    while (count != 0) {
        int batch = count > (size_t)INT_MAX ? INT_MAX : (int)count;
        vvsinh(output, input, &batch);
        output += batch;
        input += batch;
        count -= (size_t)batch;
    }
}

static void wf_vector_power_scalar(double *output, const double *input,
                                   double exponent, size_t count) {
    if (exponent == 2.0) {
        vDSP_vsqD(input, 1, output, 1, (vDSP_Length)count);
        return;
    }
    while (count != 0) {
        int batch = count > (size_t)INT_MAX ? INT_MAX : (int)count;
        vvpows(output, &exponent, input, &batch);
        output += batch;
        input += batch;
        count -= (size_t)batch;
    }
}
#else
#if defined(__GLIBC__) && defined(__x86_64__) \
        && defined(WF_HAVE_X86_SIMD) \
        && !defined(WF_DISABLE_X86_SIMD)
typedef double wf_v4df __attribute__((vector_size(32)));
typedef double wf_v8df __attribute__((vector_size(64)));
#if defined(WF_DISABLE_AVX512)
#define WF_AVX512_ENABLED 0
#else
#define WF_AVX512_ENABLED 1
#endif

#define WF_DEFINE_VECTOR_MATH(name) \
    extern wf_v4df _ZGVdN4v_##name(wf_v4df); \
    extern wf_v8df _ZGVeN8v_##name(wf_v8df); \
    __attribute__((target("avx2"))) \
    static void wf_vector_##name##_avx2( \
            double *output, const double *input, size_t count) { \
        size_t index = 0; \
        for (; index + 4 <= count; index += 4) { \
            wf_v4df value = (wf_v4df)_mm256_loadu_pd(input + index); \
            _mm256_storeu_pd(output + index, \
                (__m256d)_ZGVdN4v_##name(value)); \
        } \
        for (; index < count; ++index) output[index] = name(input[index]); \
    } \
    __attribute__((target("avx512f"))) \
    static void wf_vector_##name##_avx512( \
            double *output, const double *input, size_t count) { \
        size_t index = 0; \
        for (; index + 8 <= count; index += 8) { \
            wf_v8df value = (wf_v8df)_mm512_loadu_pd(input + index); \
            _mm512_storeu_pd(output + index, \
                (__m512d)_ZGVeN8v_##name(value)); \
        } \
        for (; index < count; ++index) output[index] = name(input[index]); \
    } \
    static void wf_vector_##name( \
            double *output, const double *input, size_t count) { \
        if (count >= 8) { \
            if (WF_AVX512_ENABLED \
                    && __builtin_cpu_supports("avx512f")) { \
                wf_vector_##name##_avx512(output, input, count); \
                return; \
            } \
            if (__builtin_cpu_supports("avx2")) { \
                wf_vector_##name##_avx2(output, input, count); \
                return; \
            } \
        } \
        { \
            size_t index; \
            for (index = 0; index < count; ++index) \
                output[index] = name(input[index]); \
        } \
    }

WF_DEFINE_VECTOR_MATH(exp)
WF_DEFINE_VECTOR_MATH(cos)
WF_DEFINE_VECTOR_MATH(sin)
#undef WF_DEFINE_VECTOR_MATH
#undef WF_AVX512_ENABLED

/* glibc only added vector cosh/sinh in 2.35, while Linux wheels target
 * manylinux_2_34. Keep these uncommon nodes scalar to preserve that ABI. */
static void wf_vector_cosh(double *output, const double *input, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) output[index] = cosh(input[index]);
}

static void wf_vector_sinh(double *output, const double *input, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) output[index] = sinh(input[index]);
}
#else
#define WF_DEFINE_SCALAR_MATH(name) \
    static void wf_vector_##name( \
            double *output, const double *input, size_t count) { \
        size_t index; \
        for (index = 0; index < count; ++index) \
            output[index] = name(input[index]); \
    }
WF_DEFINE_SCALAR_MATH(exp)
WF_DEFINE_SCALAR_MATH(cos)
WF_DEFINE_SCALAR_MATH(sin)
WF_DEFINE_SCALAR_MATH(cosh)
WF_DEFINE_SCALAR_MATH(sinh)
#undef WF_DEFINE_SCALAR_MATH
#endif

static void wf_vector_power_scalar(double *output, const double *input,
                                   double exponent, size_t count) {
    size_t index;
    if (exponent == 2.0) {
        for (index = 0; index < count; ++index)
            output[index] = input[index] * input[index];
    } else {
        for (index = 0; index < count; ++index)
            output[index] = pow(input[index], exponent);
    }
}
#endif

/* Evaluate one node at a time so platform vector-math libraries can process
 * transcendental functions in wide batches. */
static void wf_vector_drag(const cwaveform_wave *wave, const wf_node *node,
                            const double *positions, size_t count,
                            double delay, double *output) {
    const double t0 = node->p0;
    const double frequency = wf_node_parameter(wave, node, 1);
    const double width = wf_node_parameter(wave, node, 2);
    const double delta = wf_node_parameter(wave, node, 3);
    const double block = wf_node_parameter(wave, node, 4);
    const double phase = wf_node_parameter(wave, node, 5);
    const double omega = 3.14159265358979323846 / width;
    const double carrier_frequency = 2.0 * 3.14159265358979323846
        * (frequency + delta);
    const double carrier_phase = 2.0 * 3.14159265358979323846 * delta * t0 + phase;
    const double shift = (double)node->shift / (double)wf_ticks_per_second;
    const double lower = node->lower == INT64_MIN ? -DBL_MAX
        : (double)node->lower / (double)wf_ticks_per_second;
    const double upper = node->upper == INT64_MAX ? DBL_MAX
        : (double)node->upper / (double)wf_ticks_per_second;
    const int corrected = !isnan(block) && block - delta != 0.0;
    const double quadrature_scale = corrected
        ? -1.0 / (2.0 * 3.14159265358979323846 * (block - delta)) * omega
        : 0.0;
    double envelope[256], carrier[256], scratch[256];
    size_t cursor = 0;
    while (cursor < count) {
        size_t batch = count - cursor;
        size_t index;
        if (batch > 256) batch = 256;
        for (index = 0; index < batch; ++index) {
            double t = positions[cursor + index] - delay - shift;
            scratch[index] = omega * (t - t0);
            carrier[index] = carrier_frequency * t - carrier_phase;
        }
        wf_vector_sin(envelope, scratch, batch);
        wf_vector_cos(scratch, carrier, batch);
        for (index = 0; index < batch; ++index)
            envelope[index] = envelope[index] * envelope[index] * scratch[index];
        if (corrected) {
            wf_vector_sin(scratch, carrier, batch);
            for (index = 0; index < batch; ++index) {
                double t = positions[cursor + index] - delay - shift;
                carrier[index] = 2.0 * omega * (t - t0);
            }
            wf_vector_sin(carrier, carrier, batch);
            for (index = 0; index < batch; ++index)
                envelope[index] += (quadrature_scale * carrier[index]) * scratch[index];
        }
        for (index = 0; index < batch; ++index) {
            double position = positions[cursor + index] - delay;
            output[cursor + index] = position >= lower && position < upper
                ? envelope[index] : 0.0;
        }
        cursor += batch;
    }
}

static int wf_evaluate_chunk_vector(
    const cwaveform_wave *wave, const double *positions, size_t count,
    double delay, double scale, double lower_clip, double upper_clip,
    double *output) {
    double *matrix;
    double *scratch;
    uint32_t node_index;
    size_t index;
    if (count > SIZE_MAX / sizeof(double)
            || (count != 0 && (size_t)wave->node_count
                > (SIZE_MAX / sizeof(double)) / count)) return -2;
    matrix = (double *)malloc((size_t)wave->node_count * count * sizeof(double));
    scratch = (double *)malloc(count * sizeof(double));
    if (matrix == NULL || scratch == NULL) {
        free(matrix);
        free(scratch);
        return -2;
    }
    for (node_index = 0; node_index < wave->node_count; ++node_index) {
        const wf_node *node = wave->nodes + node_index;
        double *row = matrix + (size_t)node_index * count;
        double lower = node->lower == INT64_MIN ? -DBL_MAX
            : (double)node->lower / (double)wf_ticks_per_second;
        double upper = node->upper == INT64_MAX ? DBL_MAX
            : (double)node->upper / (double)wf_ticks_per_second;
        double shift = (double)node->shift / (double)wf_ticks_per_second;
        switch (node->op) {
            case WF_OP_CONSTANT:
                if (lower == -DBL_MAX && upper == DBL_MAX) {
                    for (index = 0; index < count; ++index)
                        row[index] = node->p0;
                } else {
                    for (index = 0; index < count; ++index) {
                        double position = positions[index] - delay;
                        row[index] = position >= lower && position < upper
                            ? node->p0 : 0.0;
                    }
                }
                break;
            case WF_OP_GAUSSIAN:
                for (index = 0; index < count; ++index) {
                    double position = positions[index] - delay;
                    double std = node->flags ? node->p0 : node->p1;
                    double local = (position - shift) / std;
                    scratch[index] = position >= lower && position < upper
                        ? -(local * local) : -INFINITY;
                }
                wf_vector_exp(row, scratch, count);
                break;
            case WF_OP_COS:
            case WF_OP_SIN:
                for (index = 0; index < count; ++index) {
                    double position = positions[index] - delay;
                    scratch[index] = node->p0 * (position - shift) + node->p1;
                }
                if (node->op == WF_OP_COS) {
                    wf_vector_cos(row, scratch, count);
                } else {
                    wf_vector_sin(row, scratch, count);
                }
                if (lower != -DBL_MAX || upper != DBL_MAX) {
                    for (index = 0; index < count; ++index) {
                        double position = positions[index] - delay;
                        if (position < lower || position >= upper) row[index] = 0.0;
                    }
                }
                break;
            case WF_OP_LINEAR:
                for (index = 0; index < count; ++index) {
                    double position = positions[index] - delay;
                    row[index] = position >= lower && position < upper
                        ? position - shift : 0.0;
                }
                break;
            case WF_OP_ERF:
                for (index = 0; index < count; ++index)
                    scratch[index] = (positions[index] - delay - shift) / node->p0;
                for (index = 0; index < count; ++index)
                    row[index] = erf(scratch[index]);
                break;
            case WF_OP_SINC:
                for (index = 0; index < count; ++index)
                    scratch[index] = 3.14159265358979323846 * node->p0
                        * (positions[index] - delay - shift);
                wf_vector_sin(row, scratch, count);
                for (index = 0; index < count; ++index)
                    row[index] = scratch[index] == 0.0
                        ? 1.0 : row[index] / scratch[index];
                break;
            case WF_OP_EXP:
                for (index = 0; index < count; ++index)
                    scratch[index] = node->p0
                        * (positions[index] - delay - shift);
                wf_vector_exp(row, scratch, count);
                break;
            case WF_OP_COSH:
            case WF_OP_SINH:
                for (index = 0; index < count; ++index)
                    scratch[index] = node->p0
                        * (positions[index] - delay - shift);
                if (node->op == WF_OP_COSH)
                    wf_vector_cosh(row, scratch, count);
                else
                    wf_vector_sinh(row, scratch, count);
                break;
            case WF_OP_DRAG:
                wf_vector_drag(wave, node, positions, count, delay, row);
                break;
            case WF_OP_SQUARE:
                for (index = 0; index < count; ++index) {
                    double position = positions[index] - delay;
                    row[index] = position >= lower && position < upper
                        ? 1.0 : 0.0;
                }
                break;
            case WF_OP_ADD: {
                const double *left = matrix + (size_t)node->left * count;
                const double *right = matrix + (size_t)node->right * count;
                for (index = 0; index < count; ++index)
                    row[index] = left[index] + right[index];
                break;
            }
            case WF_OP_MUL: {
                const double *left = matrix + (size_t)node->left * count;
                const double *right = matrix + (size_t)node->right * count;
                for (index = 0; index < count; ++index)
                    row[index] = left[index] * right[index];
                break;
            }
            case WF_OP_SCALE: {
                const double *source = matrix + (size_t)node->left * count;
                for (index = 0; index < count; ++index)
                    row[index] = node->p0 * source[index];
                break;
            }
            case WF_OP_POWER: {
                const double *source = matrix + (size_t)node->left * count;
                wf_vector_power_scalar(row, source, node->p0, count);
                if (lower != -DBL_MAX || upper != DBL_MAX) {
                    for (index = 0; index < count; ++index) {
                        double position = positions[index] - delay;
                        if (position < lower || position >= upper) row[index] = 0.0;
                    }
                }
                break;
            }
            case WF_OP_WINDOW: {
                const double *source = matrix + (size_t)node->left * count;
                for (index = 0; index < count; ++index) {
                    double position = positions[index] - delay;
                    row[index] = position >= lower && position < upper
                        ? source[index] : 0.0;
                }
                break;
            }
            default:
                free(matrix);
                free(scratch);
                return -3;
        }
    }
    {
        const double *root = matrix + (size_t)wave->root * count;
        /* Keep scaling and clipping fused in one compiler-vectorized pass.
         * Two separate vDSP calls cost an extra full memory traversal. */
        for (index = 0; index < count; ++index) {
            double value = scale * root[index];
            if (value < lower_clip) value = lower_clip;
            if (value > upper_clip) value = upper_clip;
            output[index] = value;
        }
    }
    free(matrix);
    free(scratch);
    return 0;
}

static size_t wf_vector_chunk_count(const cwaveform_wave *wave, size_t count) {
#if defined(__APPLE__)
    (void)wave;
    return count;
#else
    size_t chunk = (64u * 1024u)
        / ((size_t)wave->node_count * sizeof(double));
    if (chunk < 256) chunk = 256;
    if (chunk > 8192) chunk = 8192;
    return chunk < count ? chunk : count;
#endif
}

static int wf_evaluate_many_vector(
    const cwaveform_wave *wave, const double *positions, size_t count,
    double delay, double scale, double lower_clip, double upper_clip,
    double *output) {
    size_t cursor = 0;
    size_t chunk = wf_vector_chunk_count(wave, count);
    while (cursor < count) {
        size_t batch = count - cursor < chunk ? count - cursor : chunk;
        int status = wf_evaluate_chunk_vector(
            wave, positions + cursor, batch, delay, scale,
            lower_clip, upper_clip, output + cursor);
        if (status != 0) return status;
        cursor += batch;
    }
    return 0;
}
#endif

int cwaveform_wave_evaluate(
    const cwaveform_wave *wave, const double *positions, size_t count,
    int64_t delay_tick, double scale, double lower_clip, double upper_clip,
    double *output) {
    double *values;
    double delay;
    size_t index;
    if (wave == NULL || positions == NULL || output == NULL || !isfinite(scale)) {
        return -1;
    }
#if (defined(__APPLE__) || defined(WF_HAVE_X86_SIMD) || defined(_WIN32)) \
        && !defined(WF_DISABLE_BATCH_EVALUATOR)
    if (count >= 256) {
        int status = wf_evaluate_many_vector(
            wave, positions, count,
            (double)delay_tick / (double)wf_ticks_per_second,
            scale, lower_clip, upper_clip, output);
        if (status == 0) return 0;
    }
#endif
    values = (double *)malloc((size_t)wave->node_count * sizeof(*values));
    if (values == NULL) {
        return -2;
    }
    delay = (double)delay_tick / (double)wf_ticks_per_second;
    for (index = 0; index < count; ++index) {
        double value = scale * wf_evaluate_one(wave, positions[index] - delay,
                                                values);
        if (value < lower_clip) value = lower_clip;
        if (value > upper_clip) value = upper_clip;
        output[index] = value;
    }
    free(values);
    return 0;
}

static int16_t wf_quantize16(double value, double full_scale) {
    double scaled;
    if (value <= -full_scale) return INT16_MIN;
    if (value >= full_scale) return INT16_MAX;
    scaled = round(value * (32768.0 / full_scale));
    if (scaled <= -32768.0) return INT16_MIN;
    if (scaled >= 32767.0) return INT16_MAX;
    return (int16_t)scaled;
}

static int32_t wf_quantize32(double value, double full_scale) {
    double scaled;
    if (value <= -full_scale) return INT32_MIN;
    if (value >= full_scale) return INT32_MAX;
    scaled = round(value * (2147483648.0 / full_scale));
    if (scaled <= -2147483648.0) return INT32_MIN;
    if (scaled >= 2147483647.0) return INT32_MAX;
    return (int32_t)scaled;
}

static int wf_values_are_finite(const double *values, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index)
        if (!isfinite(values[index])) return 0;
    return 1;
}

#if defined(WF_HAVE_ARM64_NEON)
static int wf_quantize_array_neon(const double *values, size_t count, int dtype,
                                  double full_scale, void *output) {
    const float64x2_t finite_limit = vdupq_n_f64(DBL_MAX);
    const float64x2_t multiplier = vdupq_n_f64(
        dtype == CWAVEFORM_INT16
            ? 32768.0 / full_scale : 2147483648.0 / full_scale);
    const float64x2_t minimum = vdupq_n_f64(
        dtype == CWAVEFORM_INT16 ? -32768.0 : -2147483648.0);
    const float64x2_t maximum = vdupq_n_f64(
        dtype == CWAVEFORM_INT16 ? 32767.0 : 2147483647.0);
    uint64x2_t finite = vdupq_n_u64(UINT64_MAX);
    size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        float64x2_t first = vld1q_f64(values + index);
        float64x2_t second = vld1q_f64(values + index + 2);
        int64x2_t first_integer;
        int64x2_t second_integer;
        finite = vandq_u64(finite,
            vcleq_f64(vabsq_f64(first), finite_limit));
        finite = vandq_u64(finite,
            vcleq_f64(vabsq_f64(second), finite_limit));
        first = vmulq_f64(first, multiplier);
        second = vmulq_f64(second, multiplier);
        first = vminq_f64(vmaxq_f64(first, minimum), maximum);
        second = vminq_f64(vmaxq_f64(second, minimum), maximum);
        first_integer = vcvtaq_s64_f64(first);
        second_integer = vcvtaq_s64_f64(second);
        if (dtype == CWAVEFORM_INT16) {
            int32x4_t packed32 = vcombine_s32(
                vmovn_s64(first_integer), vmovn_s64(second_integer));
            vst1_s16((int16_t *)output + index, vmovn_s32(packed32));
        } else {
            vst1q_s32((int32_t *)output + index, vcombine_s32(
                vmovn_s64(first_integer), vmovn_s64(second_integer)));
        }
    }
    for (; index < count; ++index) {
        if (!isfinite(values[index])) return -3;
        if (dtype == CWAVEFORM_INT16)
            ((int16_t *)output)[index] = wf_quantize16(values[index], full_scale);
        else
            ((int32_t *)output)[index] = wf_quantize32(values[index], full_scale);
    }
    return (vgetq_lane_u64(finite, 0) == UINT64_MAX
            && vgetq_lane_u64(finite, 1) == UINT64_MAX) ? 0 : -3;
}
#endif

#if defined(WF_HAVE_X86_SIMD) && !defined(WF_DISABLE_X86_SIMD)
static int wf_cpu_supports_avx2(void) {
#if defined(_MSC_VER)
    int registers[4];
    unsigned __int64 xcr0;
    __cpuid(registers, 0);
    if (registers[0] < 7) return 0;
    __cpuidex(registers, 1, 0);
    if ((registers[2] & ((1 << 27) | (1 << 28)))
            != ((1 << 27) | (1 << 28))) return 0;
    xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) return 0;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2");
#endif
}

#if !defined(WF_DISABLE_AVX512) && !defined(_MSC_VER)
static int wf_cpu_supports_avx512(void) {
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512dq")
        && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vl");
}
#endif

WF_TARGET_AVX2
static int wf_quantize_array_avx2(const double *values, size_t count, int dtype,
                                  double full_scale, void *output) {
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    /* The predecessor of 0.5 avoids rounding nextafter(0.5, 0) up to 1
     * during the addition itself, while exact half-LSB ties still round away
     * from zero. Values have been bounded to the int32 range below. */
    const __m256d half = _mm256_set1_pd(0.49999999999999994);
    const __m256d finite_limit = _mm256_set1_pd(DBL_MAX);
    const __m256d multiplier = _mm256_set1_pd(
        dtype == CWAVEFORM_INT16
            ? 32768.0 / full_scale : 2147483648.0 / full_scale);
    const __m256d minimum = _mm256_set1_pd(
        dtype == CWAVEFORM_INT16 ? -32768.0 : -2147483648.0);
    const __m256d maximum = _mm256_set1_pd(
        dtype == CWAVEFORM_INT16 ? 32767.0 : 2147483647.0);
    int finite = 1;
    size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        __m256d value = _mm256_loadu_pd(values + index);
        __m256d absolute = _mm256_andnot_pd(sign_mask, value);
        __m256d adjustment = _mm256_or_pd(
            half, _mm256_and_pd(sign_mask, value));
        __m128i integer;
        finite &= _mm256_movemask_pd(_mm256_cmp_pd(
            absolute, finite_limit, _CMP_LE_OQ)) == 0xf;
        value = _mm256_mul_pd(value, multiplier);
        value = _mm256_min_pd(_mm256_max_pd(value, minimum), maximum);
        integer = _mm256_cvttpd_epi32(_mm256_add_pd(value, adjustment));
        if (dtype == CWAVEFORM_INT16) {
            __m128i packed = _mm_packs_epi32(integer, _mm_setzero_si128());
            _mm_storel_epi64((__m128i *)((int16_t *)output + index), packed);
        } else {
            _mm_storeu_si128((__m128i *)((int32_t *)output + index), integer);
        }
    }
    for (; index < count; ++index) {
        if (!isfinite(values[index])) return -3;
        if (dtype == CWAVEFORM_INT16)
            ((int16_t *)output)[index] = wf_quantize16(values[index], full_scale);
        else
            ((int32_t *)output)[index] = wf_quantize32(values[index], full_scale);
    }
    _mm256_zeroupper();
    return finite ? 0 : -3;
}

#if !defined(WF_DISABLE_AVX512) && !defined(_MSC_VER)
WF_TARGET_AVX512
static int wf_quantize_array_avx512(
    const double *values, size_t count, int dtype,
    double full_scale, void *output) {
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    const __m512d half = _mm512_set1_pd(0.49999999999999994);
    const __m512d finite_limit = _mm512_set1_pd(DBL_MAX);
    const __m512d multiplier = _mm512_set1_pd(
        dtype == CWAVEFORM_INT16
            ? 32768.0 / full_scale : 2147483648.0 / full_scale);
    const __m512d minimum = _mm512_set1_pd(
        dtype == CWAVEFORM_INT16 ? -32768.0 : -2147483648.0);
    const __m512d maximum = _mm512_set1_pd(
        dtype == CWAVEFORM_INT16 ? 32767.0 : 2147483647.0);
    size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        __m512d value = _mm512_loadu_pd(values + index);
        __m512d absolute = _mm512_andnot_pd(sign_mask, value);
        __m512d adjustment = _mm512_or_pd(
            half, _mm512_and_pd(sign_mask, value));
        __m256i packed32;
        /* Do not carry a mask reduction across iterations: older GCC emits
         * a serial mask -> GPR -> mask chain, expensive on Zen 4. */
        if (_mm512_cmp_pd_mask(absolute, finite_limit, _CMP_LE_OQ) != 0xff)
            return -3;
        value = _mm512_mul_pd(value, multiplier);
        value = _mm512_min_pd(_mm512_max_pd(value, minimum), maximum);
        /* Clipping already bounds the truncated result to int32. Avoid the
         * much more expensive double -> int64 -> int32 conversion chain. */
        packed32 = _mm512_cvttpd_epi32(_mm512_add_pd(value, adjustment));
        if (dtype == CWAVEFORM_INT16) {
            __m128i packed16 = _mm256_cvtepi32_epi16(packed32);
            _mm_storeu_si128(
                (__m128i *)((int16_t *)output + index), packed16);
        } else {
            _mm256_storeu_si256(
                (__m256i *)((int32_t *)output + index), packed32);
        }
    }
    for (; index < count; ++index) {
        if (!isfinite(values[index])) return -3;
        if (dtype == CWAVEFORM_INT16)
            ((int16_t *)output)[index] = wf_quantize16(values[index], full_scale);
        else
            ((int32_t *)output)[index] = wf_quantize32(values[index], full_scale);
    }
    return 0;
}
#endif
#endif

static int wf_quantize_array(const double *values, size_t count, int dtype,
                             double full_scale, void *output) {
    size_t index;
#if defined(WF_HAVE_ARM64_NEON)
    if (count >= 16)
        return wf_quantize_array_neon(
            values, count, dtype, full_scale, output);
#endif
#if defined(WF_HAVE_X86_SIMD) && !defined(WF_DISABLE_X86_SIMD)
    if (count >= 16) {
#if !defined(WF_DISABLE_AVX512) && !defined(_MSC_VER)
        if (wf_cpu_supports_avx512())
            return wf_quantize_array_avx512(
                values, count, dtype, full_scale, output);
#endif
        if (wf_cpu_supports_avx2())
            return wf_quantize_array_avx2(
                values, count, dtype, full_scale, output);
    }
#endif
    if (!wf_values_are_finite(values, count)) return -3;
#if defined(__APPLE__)
    if (count >= 256) {
        double *scaled = (double *)malloc(count * sizeof(*scaled));
        double scale;
        double minimum;
        double maximum;
        if (scaled == NULL) return -2;
        if (dtype == CWAVEFORM_INT16) {
            scale = 32768.0 / full_scale;
            minimum = -32768.0;
            maximum = 32767.0;
        } else if (dtype == CWAVEFORM_INT32) {
            scale = 2147483648.0 / full_scale;
            minimum = -2147483648.0;
            maximum = 2147483647.0;
        } else {
            free(scaled);
            return -1;
        }
        vDSP_vsmulD(values, 1, &scale, scaled, 1, (vDSP_Length)count);
        vDSP_vclipD(scaled, 1, &minimum, &maximum, scaled, 1,
                    (vDSP_Length)count);
        if (dtype == CWAVEFORM_INT16)
            vDSP_vfixr16D(scaled, 1, (int16_t *)output, 1,
                          (vDSP_Length)count);
        else
            vDSP_vfixr32D(scaled, 1, (int32_t *)output, 1,
                          (vDSP_Length)count);
        free(scaled);
        return 0;
    }
#endif
    if (dtype == CWAVEFORM_INT16) {
        for (index = 0; index < count; ++index)
            ((int16_t *)output)[index] = wf_quantize16(values[index], full_scale);
    } else if (dtype == CWAVEFORM_INT32) {
        for (index = 0; index < count; ++index)
            ((int32_t *)output)[index] = wf_quantize32(values[index], full_scale);
    } else {
        return -1;
    }
    return 0;
}

int cwaveform_quantize(const double *values, size_t count, int dtype,
                       double full_scale, void *output) {
    if ((count != 0 && (values == NULL || output == NULL))
            || (dtype != CWAVEFORM_INT16 && dtype != CWAVEFORM_INT32)
            || !isfinite(full_scale) || full_scale <= 0.0) return -1;
    if (count == 0) return 0;
    return wf_quantize_array(values, count, dtype, full_scale, output);
}

static int wf_nonlinear_layout(int method, int storage, size_t point_count,
                               size_t *coefficient_count,
                               size_t *item_size, size_t *data_size) {
    size_t coefficients;
    size_t item;
    if (point_count < 2 || point_count > UINT32_MAX) return -1;
    if (method == CWAVEFORM_NONLINEAR_LINEAR) {
        coefficients = point_count;
    } else if (method == CWAVEFORM_NONLINEAR_CUBIC) {
        if (point_count - 1 > SIZE_MAX / 4) return -1;
        coefficients = 4 * (point_count - 1);
    } else {
        return -1;
    }
    if (storage == CWAVEFORM_NONLINEAR_FLOAT32) {
        item = sizeof(float);
    } else if (storage == CWAVEFORM_NONLINEAR_FLOAT64) {
        item = sizeof(double);
    } else {
        return -1;
    }
    if (coefficients > (SIZE_MAX - WF_NONLINEAR_HEADER_SIZE) / item) {
        return -1;
    }
    *coefficient_count = coefficients;
    *item_size = item;
    *data_size = WF_NONLINEAR_HEADER_SIZE + coefficients * item;
    return 0;
}

static cwaveform_nonlinear_map *wf_nonlinear_allocate(
        int method, int storage, int extrapolation,
        double x_min, double x_max, double input_offset,
        double output_offset, size_t point_count) {
    cwaveform_nonlinear_map *map;
    size_t coefficient_count;
    size_t item_size;
    size_t data_size;
    if ((extrapolation != CWAVEFORM_NONLINEAR_ERROR
            && extrapolation != CWAVEFORM_NONLINEAR_CLIP)
            || !isfinite(x_min) || !isfinite(x_max) || x_min >= x_max
            || !isfinite(input_offset) || !isfinite(output_offset)
            || wf_nonlinear_layout(method, storage, point_count,
                                   &coefficient_count, &item_size,
                                   &data_size) != 0) {
        return NULL;
    }
    map = (cwaveform_nonlinear_map *)calloc(1, sizeof(*map));
    if (map == NULL) return NULL;
    map->coefficients = (double *)malloc(
        coefficient_count * sizeof(*map->coefficients));
    map->data = (uint8_t *)calloc(1, data_size);
    if (map->coefficients == NULL || map->data == NULL) {
        cwaveform_nonlinear_map_release(map);
        return NULL;
    }
    map->references = 1;
    map->method = (uint8_t)method;
    map->storage = (uint8_t)storage;
    map->extrapolation = (uint8_t)extrapolation;
    map->point_count = point_count;
    map->coefficient_count = coefficient_count;
    map->x_min = x_min;
    map->x_max = x_max;
    map->input_offset = input_offset;
    map->output_offset = output_offset;
    map->coordinate_scale = (double)(point_count - 1) / (x_max - x_min);
    map->data_size = data_size;
    if (!isfinite(map->coordinate_scale) || map->coordinate_scale <= 0.0) {
        cwaveform_nonlinear_map_release(map);
        return NULL;
    }
    (void)item_size;
    return map;
}

cwaveform_nonlinear_map *cwaveform_nonlinear_map_create(
        int method, int storage, int extrapolation,
        double x_min, double x_max, double input_offset, double output_offset,
        const double *coefficients, size_t point_count) {
    cwaveform_nonlinear_map *map = wf_nonlinear_allocate(
        method, storage, extrapolation, x_min, x_max,
        input_offset, output_offset, point_count);
    size_t index;
    size_t item_size;
    size_t ignored_size;
    size_t ignored_count;
    if (map == NULL || coefficients == NULL) {
        cwaveform_nonlinear_map_release(map);
        return NULL;
    }
    wf_nonlinear_layout(method, storage, point_count, &ignored_count,
                        &item_size, &ignored_size);
    memcpy(map->data, "NLM1", 4);
    wf_put_u16(map->data + 4, 1);
    map->data[6] = (uint8_t)method;
    map->data[7] = (uint8_t)storage;
    map->data[8] = (uint8_t)extrapolation;
    wf_put_u32(map->data + 12, (uint32_t)point_count);
    wf_put_f64(map->data + 16, x_min);
    wf_put_f64(map->data + 24, x_max);
    wf_put_f64(map->data + 32, input_offset);
    wf_put_f64(map->data + 40, output_offset);
    for (index = 0; index < map->coefficient_count; ++index) {
        double value = coefficients[index];
        uint8_t *target = map->data + WF_NONLINEAR_HEADER_SIZE
            + index * item_size;
        if (!isfinite(value)) {
            cwaveform_nonlinear_map_release(map);
            return NULL;
        }
        if (storage == CWAVEFORM_NONLINEAR_FLOAT32) {
            float compact = (float)value;
            if (!isfinite(compact)) {
                cwaveform_nonlinear_map_release(map);
                return NULL;
            }
            wf_put_f32(target, compact);
            map->coefficients[index] = (double)compact;
        } else {
            wf_put_f64(target, value);
            map->coefficients[index] = value;
        }
    }
    map->hash = wf_hash_bytes(map->data, map->data_size);
    return map;
}

cwaveform_nonlinear_map *cwaveform_nonlinear_map_from_bytes(
        const uint8_t *data, size_t size) {
    cwaveform_nonlinear_map *map;
    int method;
    int storage;
    int extrapolation;
    size_t point_count;
    size_t coefficient_count;
    size_t item_size;
    size_t expected_size;
    size_t index;
    double x_min;
    double x_max;
    double input_offset;
    double output_offset;
    if (data == NULL || size < WF_NONLINEAR_HEADER_SIZE
            || memcmp(data, "NLM1", 4) != 0
            || wf_get_u16(data + 4) != 1
            || data[9] != 0 || data[10] != 0 || data[11] != 0) {
        return NULL;
    }
    method = data[6];
    storage = data[7];
    extrapolation = data[8];
    point_count = wf_get_u32(data + 12);
    x_min = wf_get_f64(data + 16);
    x_max = wf_get_f64(data + 24);
    input_offset = wf_get_f64(data + 32);
    output_offset = wf_get_f64(data + 40);
    if (wf_nonlinear_layout(method, storage, point_count,
                            &coefficient_count, &item_size,
                            &expected_size) != 0
            || size != expected_size) {
        return NULL;
    }
    map = wf_nonlinear_allocate(
        method, storage, extrapolation, x_min, x_max,
        input_offset, output_offset, point_count);
    if (map == NULL) return NULL;
    memcpy(map->data, data, size);
    for (index = 0; index < coefficient_count; ++index) {
        const uint8_t *source = data + WF_NONLINEAR_HEADER_SIZE
            + index * item_size;
        double value = storage == CWAVEFORM_NONLINEAR_FLOAT32
            ? (double)wf_get_f32(source) : wf_get_f64(source);
        if (!isfinite(value)) {
            cwaveform_nonlinear_map_release(map);
            return NULL;
        }
        map->coefficients[index] = value;
    }
    map->hash = wf_hash_bytes(data, size);
    return map;
}

void cwaveform_nonlinear_map_retain(cwaveform_nonlinear_map *map) {
    if (map != NULL) ++map->references;
}

void cwaveform_nonlinear_map_release(cwaveform_nonlinear_map *map) {
    if (map == NULL) return;
    if (map->references > 1) {
        --map->references;
        return;
    }
    free(map->coefficients);
    free(map->data);
    free(map);
}

const uint8_t *cwaveform_nonlinear_map_bytes(
        const cwaveform_nonlinear_map *map, size_t *size) {
    if (map == NULL) return NULL;
    if (size != NULL) *size = map->data_size;
    return map->data;
}

uint64_t cwaveform_nonlinear_map_hash(
        const cwaveform_nonlinear_map *map) {
    return map == NULL ? 0 : map->hash;
}

int cwaveform_nonlinear_map_equal(const cwaveform_nonlinear_map *left,
                                  const cwaveform_nonlinear_map *right) {
    return left != NULL && right != NULL && left->hash == right->hash
        && left->data_size == right->data_size
        && memcmp(left->data, right->data, left->data_size) == 0;
}

int cwaveform_nonlinear_map_method(const cwaveform_nonlinear_map *map) {
    return map == NULL ? 0 : map->method;
}

int cwaveform_nonlinear_map_storage(const cwaveform_nonlinear_map *map) {
    return map == NULL ? 0 : map->storage;
}

int cwaveform_nonlinear_map_extrapolation(
        const cwaveform_nonlinear_map *map) {
    return map == NULL ? -1 : map->extrapolation;
}

size_t cwaveform_nonlinear_map_point_count(
        const cwaveform_nonlinear_map *map) {
    return map == NULL ? 0 : map->point_count;
}

double cwaveform_nonlinear_map_x_min(const cwaveform_nonlinear_map *map) {
    return map == NULL ? NAN : map->x_min;
}

double cwaveform_nonlinear_map_x_max(const cwaveform_nonlinear_map *map) {
    return map == NULL ? NAN : map->x_max;
}

double cwaveform_nonlinear_map_input_offset(
        const cwaveform_nonlinear_map *map) {
    return map == NULL ? NAN : map->input_offset;
}

double cwaveform_nonlinear_map_output_offset(
        const cwaveform_nonlinear_map *map) {
    return map == NULL ? NAN : map->output_offset;
}

static int wf_nonlinear_value(const cwaveform_nonlinear_map *map,
                              double input, double *output) {
    double absolute;
    double coordinate;
    double fraction;
    double value;
    size_t interval_count = map->point_count - 1;
    size_t interval;
    if (!isfinite(input)) return -3;
    absolute = input + map->input_offset;
    if (!isfinite(absolute)) return -3;
    if (absolute < map->x_min) {
        if (map->extrapolation == CWAVEFORM_NONLINEAR_ERROR) return -3;
        coordinate = 0.0;
    } else if (absolute > map->x_max) {
        if (map->extrapolation == CWAVEFORM_NONLINEAR_ERROR) return -3;
        coordinate = (double)interval_count;
    } else if (absolute == map->x_max) {
        coordinate = (double)interval_count;
    } else {
        coordinate = (absolute - map->x_min) * map->coordinate_scale;
        if (coordinate < 0.0) coordinate = 0.0;
        if (coordinate > (double)interval_count)
            coordinate = (double)interval_count;
    }
    interval = (size_t)coordinate;
    if (interval >= interval_count) {
        interval = interval_count - 1;
        fraction = 1.0;
    } else {
        fraction = coordinate - (double)interval;
    }
    if (map->method == CWAVEFORM_NONLINEAR_LINEAR) {
        double left = map->coefficients[interval];
        value = left + fraction
            * (map->coefficients[interval + 1] - left);
    } else {
        const double *coefficient = map->coefficients + 4 * interval;
        value = coefficient[0] + fraction * (coefficient[1]
            + fraction * (coefficient[2] + fraction * coefficient[3]));
    }
    value -= map->output_offset;
    if (!isfinite(value)) return -3;
    *output = value;
    return 0;
}

static int wf_nonlinear_apply_scalar(
        const cwaveform_nonlinear_map *map, const double *input, size_t count,
        int dtype, double full_scale, void *output) {
    size_t index;
    for (index = 0; index < count; ++index) {
        double value;
        int status = wf_nonlinear_value(map, input[index], &value);
        if (status != 0) return status;
        if (dtype == CWAVEFORM_FLOAT64)
            ((double *)output)[index] = value;
        else if (dtype == CWAVEFORM_INT16)
            ((int16_t *)output)[index] = wf_quantize16(value, full_scale);
        else
            ((int32_t *)output)[index] = wf_quantize32(value, full_scale);
    }
    return 0;
}

#if defined(WF_HAVE_ARM64_NEON) && !defined(WF_DISABLE_NONLINEAR_SIMD)
static int wf_nonlinear_apply_neon(
        const cwaveform_nonlinear_map *map, const double *input, size_t count,
        int dtype, double full_scale, void *output) {
    const size_t interval_count = map->point_count - 1;
    const float64x2_t input_offset = vdupq_n_f64(map->input_offset);
    const float64x2_t output_offset = vdupq_n_f64(map->output_offset);
    const float64x2_t x_min = vdupq_n_f64(map->x_min);
    const float64x2_t x_max = vdupq_n_f64(map->x_max);
    const float64x2_t coordinate_scale =
        vdupq_n_f64(map->coordinate_scale);
    const float64x2_t zero = vdupq_n_f64(0.0);
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t last_coordinate =
        vdupq_n_f64((double)interval_count);
    const float64x2_t finite_limit = vdupq_n_f64(DBL_MAX);
    size_t index = 0;
    for (; index + 2 <= count; index += 2) {
        float64x2_t absolute = vaddq_f64(
            vld1q_f64(input + index), input_offset);
        uint64x2_t finite = vcleq_f64(vabsq_f64(absolute), finite_limit);
        uint64x2_t endpoint;
        uint64x2_t raw_indices;
        uint64x2_t indices;
        float64x2_t coordinate;
        float64x2_t fraction;
        float64x2_t value;
        uint64_t first;
        uint64_t second;
        double coefficient0[2];
        double coefficient1[2];
        if (vgetq_lane_u64(finite, 0) == 0
                || vgetq_lane_u64(finite, 1) == 0) return -3;
        if (map->extrapolation == CWAVEFORM_NONLINEAR_ERROR) {
            uint64x2_t outside = vorrq_u64(
                vcltq_f64(absolute, x_min), vcgtq_f64(absolute, x_max));
            if (vgetq_lane_u64(outside, 0) != 0
                    || vgetq_lane_u64(outside, 1) != 0) return -3;
        } else {
            absolute = vmaxq_f64(x_min, vminq_f64(absolute, x_max));
        }
        coordinate = vmulq_f64(
            vsubq_f64(absolute, x_min), coordinate_scale);
        coordinate = vmaxq_f64(
            zero, vminq_f64(coordinate, last_coordinate));
        endpoint = vorrq_u64(
            vcgeq_f64(coordinate, last_coordinate),
            vceqq_f64(absolute, x_max));
        raw_indices = vcvtq_u64_f64(coordinate);
        indices = vbslq_u64(
            endpoint, vdupq_n_u64((uint64_t)(interval_count - 1)),
            raw_indices);
        fraction = vsubq_f64(coordinate, vcvtq_f64_u64(raw_indices));
        fraction = vbslq_f64(endpoint, one, fraction);
        first = vgetq_lane_u64(indices, 0);
        second = vgetq_lane_u64(indices, 1);
        if (map->method == CWAVEFORM_NONLINEAR_LINEAR) {
            float64x2_t left;
            float64x2_t right;
            coefficient0[0] = map->coefficients[first];
            coefficient0[1] = map->coefficients[second];
            coefficient1[0] = map->coefficients[first + 1];
            coefficient1[1] = map->coefficients[second + 1];
            left = vld1q_f64(coefficient0);
            right = vld1q_f64(coefficient1);
            value = vfmaq_f64(left, fraction, vsubq_f64(right, left));
        } else {
            double coefficient2[2];
            double coefficient3[2];
            float64x2_t inner;
            first *= 4;
            second *= 4;
            coefficient0[0] = map->coefficients[first];
            coefficient0[1] = map->coefficients[second];
            coefficient1[0] = map->coefficients[first + 1];
            coefficient1[1] = map->coefficients[second + 1];
            coefficient2[0] = map->coefficients[first + 2];
            coefficient2[1] = map->coefficients[second + 2];
            coefficient3[0] = map->coefficients[first + 3];
            coefficient3[1] = map->coefficients[second + 3];
            inner = vfmaq_f64(
                vld1q_f64(coefficient2), fraction,
                vld1q_f64(coefficient3));
            inner = vfmaq_f64(
                vld1q_f64(coefficient1), fraction, inner);
            value = vfmaq_f64(
                vld1q_f64(coefficient0), fraction, inner);
        }
        value = vsubq_f64(value, output_offset);
        finite = vcleq_f64(vabsq_f64(value), finite_limit);
        if (vgetq_lane_u64(finite, 0) == 0
                || vgetq_lane_u64(finite, 1) == 0) return -3;
        if (dtype == CWAVEFORM_FLOAT64) {
            vst1q_f64((double *)output + index, value);
        } else {
            double values[2];
            vst1q_f64(values, value);
            if (dtype == CWAVEFORM_INT16) {
                ((int16_t *)output)[index] =
                    wf_quantize16(values[0], full_scale);
                ((int16_t *)output)[index + 1] =
                    wf_quantize16(values[1], full_scale);
            } else {
                ((int32_t *)output)[index] =
                    wf_quantize32(values[0], full_scale);
                ((int32_t *)output)[index + 1] =
                    wf_quantize32(values[1], full_scale);
            }
        }
    }
    if (index != count) {
        size_t item_size = dtype == CWAVEFORM_FLOAT64 ? sizeof(double)
            : dtype == CWAVEFORM_INT16 ? sizeof(int16_t) : sizeof(int32_t);
        return wf_nonlinear_apply_scalar(
            map, input + index, count - index, dtype, full_scale,
            (uint8_t *)output + index * item_size);
    }
    return 0;
}
#endif

#if defined(WF_HAVE_X86_SIMD) && !defined(WF_DISABLE_X86_SIMD) \
        && !defined(WF_DISABLE_NONLINEAR_SIMD)
WF_TARGET_AVX2
static int wf_nonlinear_apply_avx2(
        const cwaveform_nonlinear_map *map, const double *input, size_t count,
        int dtype, double full_scale, void *output) {
    const size_t interval_count = map->point_count - 1;
    const __m256d input_offset = _mm256_set1_pd(map->input_offset);
    const __m256d output_offset = _mm256_set1_pd(map->output_offset);
    const __m256d x_min = _mm256_set1_pd(map->x_min);
    const __m256d x_max = _mm256_set1_pd(map->x_max);
    const __m256d coordinate_scale = _mm256_set1_pd(map->coordinate_scale);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d last_coordinate =
        _mm256_set1_pd((double)interval_count);
    const __m256d finite_limit = _mm256_set1_pd(DBL_MAX);
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        __m256d absolute = _mm256_add_pd(
            _mm256_loadu_pd(input + index), input_offset);
        __m256d absolute_magnitude =
            _mm256_andnot_pd(sign_mask, absolute);
        __m256d coordinate;
        __m256d endpoint;
        __m128i raw_indices;
        __m128i indices;
        __m256d fraction;
        __m256d value;
        int32_t index_values[4];
        if (_mm256_movemask_pd(_mm256_cmp_pd(
                absolute_magnitude, finite_limit, _CMP_LE_OQ)) != 0x0f) {
            _mm256_zeroupper();
            return -3;
        }
        if (map->extrapolation == CWAVEFORM_NONLINEAR_ERROR) {
            __m256d outside = _mm256_or_pd(
                _mm256_cmp_pd(absolute, x_min, _CMP_LT_OQ),
                _mm256_cmp_pd(absolute, x_max, _CMP_GT_OQ));
            if (_mm256_movemask_pd(outside) != 0) {
                _mm256_zeroupper();
                return -3;
            }
        } else {
            absolute = _mm256_max_pd(x_min, _mm256_min_pd(absolute, x_max));
        }
        coordinate = _mm256_mul_pd(
            _mm256_sub_pd(absolute, x_min), coordinate_scale);
        coordinate = _mm256_max_pd(
            zero, _mm256_min_pd(coordinate, last_coordinate));
        endpoint = _mm256_or_pd(
            _mm256_cmp_pd(coordinate, last_coordinate, _CMP_GE_OQ),
            _mm256_cmp_pd(absolute, x_max, _CMP_EQ_OQ));
        raw_indices = _mm256_cvttpd_epi32(coordinate);
        _mm_storeu_si128((__m128i *)index_values, raw_indices);
        if (_mm256_movemask_pd(endpoint) != 0) {
            int lane;
            int endpoint_mask = _mm256_movemask_pd(endpoint);
            for (lane = 0; lane < 4; ++lane)
                if ((endpoint_mask & (1 << lane)) != 0)
                    index_values[lane] = (int32_t)(interval_count - 1);
        }
        indices = _mm_loadu_si128((const __m128i *)index_values);
        fraction = _mm256_sub_pd(
            coordinate, _mm256_cvtepi32_pd(raw_indices));
        fraction = _mm256_blendv_pd(fraction, one, endpoint);
        if (map->method == CWAVEFORM_NONLINEAR_LINEAR) {
            __m256d left = _mm256_i32gather_pd(
                map->coefficients, indices, 8);
            __m256d right = _mm256_i32gather_pd(
                map->coefficients + 1, indices, 8);
            value = _mm256_add_pd(
                left, _mm256_mul_pd(fraction, _mm256_sub_pd(right, left)));
        } else {
            /* Cubic coefficients are four adjacent doubles per interval.
             * Load complete rows and transpose instead of gathering the same
             * cache lines four times. Keep Horner's arithmetic unchanged. */
            __m256d row0 = _mm256_loadu_pd(map->coefficients + 4 * (size_t)index_values[0]);
            __m256d row1 = _mm256_loadu_pd(map->coefficients + 4 * (size_t)index_values[1]);
            __m256d row2 = _mm256_loadu_pd(map->coefficients + 4 * (size_t)index_values[2]);
            __m256d row3 = _mm256_loadu_pd(map->coefficients + 4 * (size_t)index_values[3]);
            __m256d lo01 = _mm256_unpacklo_pd(row0, row1);
            __m256d hi01 = _mm256_unpackhi_pd(row0, row1);
            __m256d lo23 = _mm256_unpacklo_pd(row2, row3);
            __m256d hi23 = _mm256_unpackhi_pd(row2, row3);
            __m256d coefficient0 = _mm256_permute2f128_pd(lo01, lo23, 0x20);
            __m256d coefficient1 = _mm256_permute2f128_pd(hi01, hi23, 0x20);
            __m256d coefficient2 = _mm256_permute2f128_pd(lo01, lo23, 0x31);
            __m256d coefficient3 = _mm256_permute2f128_pd(hi01, hi23, 0x31);
            value = _mm256_add_pd(
                coefficient2, _mm256_mul_pd(fraction, coefficient3));
            value = _mm256_add_pd(
                coefficient1, _mm256_mul_pd(fraction, value));
            value = _mm256_add_pd(
                coefficient0, _mm256_mul_pd(fraction, value));
        }
        value = _mm256_sub_pd(value, output_offset);
        absolute_magnitude = _mm256_andnot_pd(sign_mask, value);
        if (_mm256_movemask_pd(_mm256_cmp_pd(
                absolute_magnitude, finite_limit, _CMP_LE_OQ)) != 0x0f) {
            _mm256_zeroupper();
            return -3;
        }
        if (dtype == CWAVEFORM_FLOAT64) {
            _mm256_storeu_pd((double *)output + index, value);
        } else {
            double values[4];
            int lane;
            _mm256_storeu_pd(values, value);
            if (dtype == CWAVEFORM_INT16)
                for (lane = 0; lane < 4; ++lane)
                    ((int16_t *)output)[index + (size_t)lane] =
                        wf_quantize16(values[lane], full_scale);
            else
                for (lane = 0; lane < 4; ++lane)
                    ((int32_t *)output)[index + (size_t)lane] =
                        wf_quantize32(values[lane], full_scale);
        }
    }
    _mm256_zeroupper();
    if (index != count) {
        size_t item_size = dtype == CWAVEFORM_FLOAT64 ? sizeof(double)
            : dtype == CWAVEFORM_INT16 ? sizeof(int16_t) : sizeof(int32_t);
        return wf_nonlinear_apply_scalar(
            map, input + index, count - index, dtype, full_scale,
            (uint8_t *)output + index * item_size);
    }
    return 0;
}

#if !defined(WF_DISABLE_AVX512) && !defined(_MSC_VER)
WF_TARGET_AVX512
static int wf_nonlinear_apply_avx512(
        const cwaveform_nonlinear_map *map, const double *input, size_t count,
        int dtype, double full_scale, void *output) {
    const size_t interval_count = map->point_count - 1;
    const __m512d input_offset = _mm512_set1_pd(map->input_offset);
    const __m512d output_offset = _mm512_set1_pd(map->output_offset);
    const __m512d x_min = _mm512_set1_pd(map->x_min);
    const __m512d x_max = _mm512_set1_pd(map->x_max);
    const __m512d coordinate_scale = _mm512_set1_pd(map->coordinate_scale);
    const __m512d zero = _mm512_setzero_pd();
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d last_coordinate =
        _mm512_set1_pd((double)interval_count);
    const __m512d finite_limit = _mm512_set1_pd(DBL_MAX);
    const __m512i absolute_mask =
        _mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        __m512d absolute = _mm512_add_pd(
            _mm512_loadu_pd(input + index), input_offset);
        __m512d absolute_magnitude = _mm512_castsi512_pd(
            _mm512_and_si512(_mm512_castpd_si512(absolute), absolute_mask));
        __m512d coordinate;
        __mmask8 endpoint;
        __m512i raw_indices;
        __m512i indices;
        __m512d fraction;
        __m512d value;
        if (_mm512_cmp_pd_mask(
                absolute_magnitude, finite_limit, _CMP_LE_OQ) != 0xff) {
            _mm256_zeroupper();
            return -3;
        }
        if (map->extrapolation == CWAVEFORM_NONLINEAR_ERROR) {
            __mmask8 outside = _mm512_cmp_pd_mask(
                absolute, x_min, _CMP_LT_OQ)
                | _mm512_cmp_pd_mask(absolute, x_max, _CMP_GT_OQ);
            if (outside != 0) {
                _mm256_zeroupper();
                return -3;
            }
        } else {
            absolute = _mm512_max_pd(x_min, _mm512_min_pd(absolute, x_max));
        }
        coordinate = _mm512_mul_pd(
            _mm512_sub_pd(absolute, x_min), coordinate_scale);
        coordinate = _mm512_max_pd(
            zero, _mm512_min_pd(coordinate, last_coordinate));
        endpoint = _mm512_cmp_pd_mask(
            coordinate, last_coordinate, _CMP_GE_OQ)
            | _mm512_cmp_pd_mask(absolute, x_max, _CMP_EQ_OQ);
        raw_indices = _mm512_cvttpd_epi64(coordinate);
        indices = _mm512_mask_mov_epi64(
            raw_indices, endpoint,
            _mm512_set1_epi64((long long)(interval_count - 1)));
        fraction = _mm512_sub_pd(
            coordinate, _mm512_cvtepi64_pd(raw_indices));
        fraction = _mm512_mask_mov_pd(fraction, endpoint, one);
        if (map->method == CWAVEFORM_NONLINEAR_LINEAR) {
            __m512d left = _mm512_i64gather_pd(
                indices, map->coefficients, 8);
            __m512d right = _mm512_i64gather_pd(
                indices, map->coefficients + 1, 8);
            value = _mm512_add_pd(
                left, _mm512_mul_pd(fraction, _mm512_sub_pd(right, left)));
        } else {
            int64_t index_values[8];
            __m256d columns[8];
            int half;
            __m512d coefficient0, coefficient1, coefficient2, coefficient3;
            _mm512_storeu_si512((void *)index_values, indices);
            for (half = 0; half < 2; ++half) {
                const int64_t *ix = index_values + 4 * half;
                __m256d row0 = _mm256_loadu_pd(map->coefficients + 4 * (size_t)ix[0]);
                __m256d row1 = _mm256_loadu_pd(map->coefficients + 4 * (size_t)ix[1]);
                __m256d row2 = _mm256_loadu_pd(map->coefficients + 4 * (size_t)ix[2]);
                __m256d row3 = _mm256_loadu_pd(map->coefficients + 4 * (size_t)ix[3]);
                __m256d lo01 = _mm256_unpacklo_pd(row0, row1);
                __m256d hi01 = _mm256_unpackhi_pd(row0, row1);
                __m256d lo23 = _mm256_unpacklo_pd(row2, row3);
                __m256d hi23 = _mm256_unpackhi_pd(row2, row3);
                columns[4 * half] = _mm256_permute2f128_pd(lo01, lo23, 0x20);
                columns[4 * half + 1] = _mm256_permute2f128_pd(hi01, hi23, 0x20);
                columns[4 * half + 2] = _mm256_permute2f128_pd(lo01, lo23, 0x31);
                columns[4 * half + 3] = _mm256_permute2f128_pd(hi01, hi23, 0x31);
            }
            coefficient0 = _mm512_insertf64x4(_mm512_castpd256_pd512(columns[0]), columns[4], 1);
            coefficient1 = _mm512_insertf64x4(_mm512_castpd256_pd512(columns[1]), columns[5], 1);
            coefficient2 = _mm512_insertf64x4(_mm512_castpd256_pd512(columns[2]), columns[6], 1);
            coefficient3 = _mm512_insertf64x4(_mm512_castpd256_pd512(columns[3]), columns[7], 1);
            value = _mm512_add_pd(
                coefficient2, _mm512_mul_pd(fraction, coefficient3));
            value = _mm512_add_pd(
                coefficient1, _mm512_mul_pd(fraction, value));
            value = _mm512_add_pd(
                coefficient0, _mm512_mul_pd(fraction, value));
        }
        value = _mm512_sub_pd(value, output_offset);
        absolute_magnitude = _mm512_castsi512_pd(
            _mm512_and_si512(_mm512_castpd_si512(value), absolute_mask));
        if (_mm512_cmp_pd_mask(
                absolute_magnitude, finite_limit, _CMP_LE_OQ) != 0xff) {
            _mm256_zeroupper();
            return -3;
        }
        if (dtype == CWAVEFORM_FLOAT64) {
            _mm512_storeu_pd((double *)output + index, value);
        } else {
            double values[8];
            int lane;
            _mm512_storeu_pd(values, value);
            if (dtype == CWAVEFORM_INT16)
                for (lane = 0; lane < 8; ++lane)
                    ((int16_t *)output)[index + (size_t)lane] =
                        wf_quantize16(values[lane], full_scale);
            else
                for (lane = 0; lane < 8; ++lane)
                    ((int32_t *)output)[index + (size_t)lane] =
                        wf_quantize32(values[lane], full_scale);
        }
    }
    _mm256_zeroupper();
    if (index != count) {
        size_t item_size = dtype == CWAVEFORM_FLOAT64 ? sizeof(double)
            : dtype == CWAVEFORM_INT16 ? sizeof(int16_t) : sizeof(int32_t);
        return wf_nonlinear_apply_scalar(
            map, input + index, count - index, dtype, full_scale,
            (uint8_t *)output + index * item_size);
    }
    return 0;
}
#endif
#endif

static int wf_nonlinear_apply_dispatch(
        const cwaveform_nonlinear_map *map, const double *input, size_t count,
        int dtype, double full_scale, void *output) {
#if defined(WF_HAVE_ARM64_NEON) && !defined(WF_DISABLE_NONLINEAR_SIMD)
    if (count >= 8)
        return wf_nonlinear_apply_neon(
            map, input, count, dtype, full_scale, output);
#endif
#if defined(WF_HAVE_X86_SIMD) && !defined(WF_DISABLE_X86_SIMD) \
        && !defined(WF_DISABLE_NONLINEAR_SIMD)
    if (count >= 16) {
#if !defined(WF_DISABLE_AVX512) && !defined(_MSC_VER)
        if (wf_cpu_supports_avx512())
            return wf_nonlinear_apply_avx512(
                map, input, count, dtype, full_scale, output);
#endif
        if (map->point_count <= (size_t)INT32_MAX
                && (map->method == CWAVEFORM_NONLINEAR_LINEAR
                    || map->point_count - 1 <= (size_t)INT32_MAX / 4)
                && wf_cpu_supports_avx2())
            return wf_nonlinear_apply_avx2(
                map, input, count, dtype, full_scale, output);
    }
#endif
    return wf_nonlinear_apply_scalar(
        map, input, count, dtype, full_scale, output);
}

int cwaveform_nonlinear_map_apply(
        const cwaveform_nonlinear_map *map, const double *input, size_t count,
        int dtype, double full_scale, void *output) {
    if (map == NULL || (count != 0 && (input == NULL || output == NULL))
            || (dtype != CWAVEFORM_FLOAT64 && dtype != CWAVEFORM_INT16
                && dtype != CWAVEFORM_INT32)
            || !isfinite(full_scale) || full_scale <= 0.0) {
        return -1;
    }
    if (count == 0) return 0;
    if (dtype != CWAVEFORM_FLOAT64 && count >= 64) {
        const size_t scratch_capacity = count < 4096 ? count : 4096;
        const size_t item_size = dtype == CWAVEFORM_INT16
            ? sizeof(int16_t) : sizeof(int32_t);
        double *scratch = (double *)malloc(
            scratch_capacity * sizeof(*scratch));
        size_t offset = 0;
        if (scratch != NULL) {
            while (offset < count) {
                size_t batch = count - offset;
                int status;
                if (batch > scratch_capacity) batch = scratch_capacity;
                status = wf_nonlinear_apply_dispatch(
                    map, input + offset, batch, CWAVEFORM_FLOAT64,
                    full_scale, scratch);
                if (status == 0)
                    status = wf_quantize_array(
                        scratch, batch, dtype, full_scale,
                        (uint8_t *)output + offset * item_size);
                if (status != 0) {
                    free(scratch);
                    return status;
                }
                offset += batch;
            }
            free(scratch);
            return 0;
        }
    }
    return wf_nonlinear_apply_dispatch(
        map, input, count, dtype, full_scale, output);
}

/* Form the local coordinate before converting to binary64.  Subtracting two
 * already-rounded seconds values loses enough precision to move an exact
 * pulse boundary outside its half-open support. */
static double wf_local_grid_position(int64_t start_tick, size_t index,
                                     int64_t step_numerator,
                                     int64_t step_denominator,
                                     int64_t delay_tick) {
    long double numerator = (((long double)start_tick
                              - (long double)delay_tick)
                             * (long double)step_denominator
                             + (long double)index
                             * (long double)step_numerator);
    return (double)(numerator / ((long double)step_denominator
                                * (long double)wf_ticks_per_second));
}

int cwaveform_wave_sample(
    const cwaveform_wave *wave, int64_t start_tick, size_t count,
    int64_t step_numerator, int64_t step_denominator, int64_t delay_tick,
    double scale, double lower_clip, double upper_clip, int dtype,
    double full_scale, void *output) {
    double *values;
    size_t index;
    if (wave == NULL || output == NULL || step_numerator <= 0
            || step_denominator <= 0 || !isfinite(scale)
            || !isfinite(full_scale) || full_scale <= 0.0) {
        return -1;
    }
    if (dtype != CWAVEFORM_FLOAT64 && dtype != CWAVEFORM_INT16
            && dtype != CWAVEFORM_INT32) return -1;
#if (defined(__APPLE__) || defined(WF_HAVE_X86_SIMD) || defined(_WIN32)) \
        && !defined(WF_DISABLE_BATCH_EVALUATOR)
    if (count >= 256) {
        size_t cursor = 0;
        size_t capacity = wf_vector_chunk_count(wave, count);
        size_t item_size = dtype == CWAVEFORM_INT16
            ? sizeof(int16_t) : dtype == CWAVEFORM_INT32
                ? sizeof(int32_t) : sizeof(double);
        double *positions = (double *)malloc(capacity * sizeof(*positions));
        double *samples = dtype == CWAVEFORM_FLOAT64
            ? NULL : (double *)malloc(capacity * sizeof(*samples));
        if (positions != NULL
                && (dtype == CWAVEFORM_FLOAT64 || samples != NULL)) {
            int status = 0;
            while (cursor < count) {
                size_t batch = count - cursor < capacity
                    ? count - cursor : capacity;
                double *chunk_samples = dtype == CWAVEFORM_FLOAT64
                    ? (double *)output + cursor : samples;
                for (index = 0; index < batch; ++index)
                    positions[index] = wf_local_grid_position(
                        start_tick, cursor + index,
                        step_numerator, step_denominator, delay_tick);
                status = wf_evaluate_many_vector(
                    wave, positions, batch, 0.0, scale,
                    lower_clip, upper_clip, chunk_samples);
                if (status != 0) break;
                if (dtype == CWAVEFORM_FLOAT64)
                    status = wf_values_are_finite(chunk_samples, batch)
                        ? 0 : -3;
                else
                    status = wf_quantize_array(
                        chunk_samples, batch, dtype, full_scale,
                        (uint8_t *)output + cursor * item_size);
                if (status != 0) break;
                cursor += batch;
            }
            free(positions);
            free(samples);
            return status;
        }
        free(positions);
        free(samples);
    }
#endif
    values = (double *)malloc((size_t)wave->node_count * sizeof(*values));
    if (values == NULL) return -2;
    for (index = 0; index < count; ++index) {
        double position = wf_local_grid_position(
            start_tick, index, step_numerator, step_denominator, delay_tick);
        double value = scale * wf_evaluate_one(wave, position, values);
        if (value < lower_clip) value = lower_clip;
        if (value > upper_clip) value = upper_clip;
        if (!isfinite(value)) {
            free(values);
            return -3;
        }
        if (dtype == CWAVEFORM_FLOAT64) {
            ((double *)output)[index] = value;
        } else if (dtype == CWAVEFORM_INT16) {
            ((int16_t *)output)[index] = wf_quantize16(value, full_scale);
        } else if (dtype == CWAVEFORM_INT32) {
            ((int32_t *)output)[index] = wf_quantize32(value, full_scale);
        }
    }
    free(values);
    return 0;
}

static int wf_stack_encode(cwaveform_stack *stack) {
    size_t sizes_size;
    size_t template_bytes = 0;
    size_t event_bytes;
    size_t event_offset;
    size_t total;
    size_t index;
    size_t cursor;
    if (stack->template_count > UINT32_MAX
            || stack->event_count > UINT32_MAX) {
        return -1;
    }
    for (index = 0; index < stack->template_count; ++index) {
        if (stack->templates[index]->data_size > UINT32_MAX - template_bytes) {
            return -1;
        }
        template_bytes += stack->templates[index]->data_size;
    }
    sizes_size = 4 * stack->template_count;
    event_bytes = 20 * stack->event_count;
    if (WF_STACK_HEADER_SIZE > SIZE_MAX - sizes_size
            || WF_STACK_HEADER_SIZE + sizes_size > SIZE_MAX - template_bytes
            || WF_STACK_HEADER_SIZE + sizes_size + template_bytes
               > SIZE_MAX - event_bytes) {
        return -1;
    }
    event_offset = WF_STACK_HEADER_SIZE + sizes_size + template_bytes;
    total = event_offset + event_bytes;
    stack->data = (uint8_t *)calloc(1, total);
    if (stack->data == NULL) return -2;
    stack->data_size = total;
    memcpy(stack->data, "WNS4", 4);
    wf_put_u16(stack->data + 4, WF_VERSION);
    wf_put_u16(stack->data + 6, 0);
    wf_put_u32(stack->data + 8, (uint32_t)stack->template_count);
    wf_put_u32(stack->data + 12, (uint32_t)stack->event_count);
    cursor = WF_STACK_HEADER_SIZE + sizes_size;
    for (index = 0; index < stack->template_count; ++index) {
        size_t size = stack->templates[index]->data_size;
        if (size > UINT32_MAX) return -1;
        wf_put_u32(stack->data + WF_STACK_HEADER_SIZE + 4 * index,
                   (uint32_t)size);
        memcpy(stack->data + cursor, stack->templates[index]->data, size);
        cursor += size;
    }
    for (index = 0; index < stack->event_count; ++index) {
        wf_put_u32(stack->data + event_offset + 4 * index,
                   stack->template_ids[index]);
        wf_put_i64(stack->data + event_offset + 4 * stack->event_count
                   + 8 * index, stack->delays[index]);
        wf_put_f64(stack->data + event_offset + 12 * stack->event_count
                   + 8 * index, stack->scales[index]);
    }
    return 0;
}

static cwaveform_stack *wf_stack_allocate(size_t template_count,
                                         size_t event_count) {
    cwaveform_stack *stack;
    if (template_count > UINT32_MAX || event_count > UINT32_MAX
            || template_count > SIZE_MAX / sizeof(cwaveform_wave *)
            || event_count > SIZE_MAX / sizeof(double)
            || event_count > SIZE_MAX / sizeof(int64_t)
            || event_count > SIZE_MAX / sizeof(uint32_t)) return NULL;
    stack = (cwaveform_stack *)calloc(1, sizeof(*stack));
    if (stack == NULL) return NULL;
    stack->references = 1;
    stack->lower_tick = INT64_MIN;
    stack->upper_tick = INT64_MAX;
    stack->template_count = template_count;
    stack->event_count = event_count;
    stack->templates = (cwaveform_wave **)calloc(template_count,
                                                  sizeof(*stack->templates));
    stack->template_ids = (uint32_t *)malloc(event_count * sizeof(uint32_t));
    stack->delays = (int64_t *)malloc(event_count * sizeof(int64_t));
    stack->scales = (double *)malloc(event_count * sizeof(double));
    if ((template_count && stack->templates == NULL)
            || (event_count && (stack->template_ids == NULL
                || stack->delays == NULL || stack->scales == NULL))) {
        cwaveform_stack_release(stack);
        return NULL;
    }
    return stack;
}

static void wf_stack_include_support(cwaveform_stack *stack,
                                      const cwaveform_wave *wave,
                                      int64_t delay, int *have_support) {
    int64_t lower = wf_add_tick(wave->nodes[wave->root].lower, delay);
    int64_t upper = wf_add_tick(wave->nodes[wave->root].upper, delay);
    if (lower >= upper) return;
    if (!*have_support || lower < stack->lower_tick) stack->lower_tick = lower;
    if (!*have_support || upper > stack->upper_tick) stack->upper_tick = upper;
    *have_support = 1;
}

cwaveform_stack *cwaveform_stack_create(
    cwaveform_wave *const *templates, const uint32_t *template_ids,
    const int64_t *delay_ticks, const double *scales,
    size_t template_count, size_t event_count) {
    cwaveform_stack *stack;
    size_t index;
    int have_support = 0;
    if ((template_count != 0 && templates == NULL)
            || (event_count != 0 && (template_ids == NULL
                || delay_ticks == NULL || scales == NULL))) return NULL;
    stack = wf_stack_allocate(template_count, event_count);
    if (stack == NULL) return NULL;
    for (index = 0; index < template_count; ++index) {
        if (templates[index] == NULL) {
            cwaveform_stack_release(stack);
            return NULL;
        }
        stack->templates[index] = templates[index];
        cwaveform_wave_retain(stack->templates[index]);
    }
    for (index = 0; index < event_count; ++index) {
        if (template_ids[index] >= template_count || !isfinite(scales[index])) {
            cwaveform_stack_release(stack);
            return NULL;
        }
        stack->template_ids[index] = template_ids[index];
        stack->delays[index] = delay_ticks[index];
        stack->scales[index] = scales[index];
        wf_stack_include_support(stack, stack->templates[template_ids[index]],
                                  delay_ticks[index], &have_support);
    }
    if (wf_stack_encode(stack) != 0) {
        cwaveform_stack_release(stack);
        return NULL;
    }
    return stack;
}

cwaveform_stack *cwaveform_stack_materialize(
    const cwaveform_stack *stack, int64_t global_shift, double offset) {
    cwaveform_wave **templates;
    uint32_t *ids;
    int64_t *delays;
    double *scales;
    cwaveform_wave *constant = NULL;
    cwaveform_stack *result;
    size_t template_count;
    size_t event_count;
    size_t index;
    int add_offset;
    if (stack == NULL || !isfinite(offset)) return NULL;
    if (global_shift == 0 && offset == 0.0) {
        cwaveform_stack_retain((cwaveform_stack *)stack);
        return (cwaveform_stack *)stack;
    }
    add_offset = offset != 0.0;
    if (add_offset && (stack->template_count == SIZE_MAX
                       || stack->event_count == SIZE_MAX)) return NULL;
    template_count = stack->template_count + (size_t)add_offset;
    event_count = stack->event_count + (size_t)add_offset;
    templates = (cwaveform_wave **)malloc(template_count * sizeof(*templates));
    ids = (uint32_t *)malloc(event_count * sizeof(*ids));
    delays = (int64_t *)malloc(event_count * sizeof(*delays));
    scales = (double *)malloc(event_count * sizeof(*scales));
    if ((template_count && templates == NULL)
            || (event_count && (ids == NULL || delays == NULL
                                || scales == NULL))) {
        free(templates);
        free(ids);
        free(delays);
        free(scales);
        return NULL;
    }
    for (index = 0; index < stack->template_count; ++index)
        templates[index] = stack->templates[index];
    for (index = 0; index < stack->event_count; ++index) {
        ids[index] = stack->template_ids[index];
        delays[index] = wf_add_tick(stack->delays[index], global_shift);
        scales[index] = stack->scales[index];
    }
    if (add_offset) {
        constant = cwaveform_wave_constant(offset);
        if (constant == NULL) {
            free(templates);
            free(ids);
            free(delays);
            free(scales);
            return NULL;
        }
        templates[stack->template_count] = constant;
        ids[stack->event_count] = (uint32_t)stack->template_count;
        delays[stack->event_count] = 0;
        scales[stack->event_count] = 1.0;
    }
    result = cwaveform_stack_create(templates, ids, delays, scales,
                                    template_count, event_count);
    cwaveform_wave_release(constant);
    free(templates);
    free(ids);
    free(delays);
    free(scales);
    return result;
}

cwaveform_stack *cwaveform_stack_combine(
    const cwaveform_stack *left, int64_t left_shift,
    const cwaveform_stack *right, int64_t right_shift) {
    size_t template_count;
    size_t event_count;
    cwaveform_wave **templates;
    uint32_t *ids;
    int64_t *delays;
    double *scales;
    cwaveform_stack *result;
    size_t index;
    if (left == NULL || right == NULL
            || left->template_count > SIZE_MAX - right->template_count
            || left->event_count > SIZE_MAX - right->event_count
            || left->template_count > UINT32_MAX - right->template_count)
        return NULL;
    template_count = left->template_count + right->template_count;
    event_count = left->event_count + right->event_count;
    templates = (cwaveform_wave **)malloc(template_count * sizeof(*templates));
    ids = (uint32_t *)malloc(event_count * sizeof(*ids));
    delays = (int64_t *)malloc(event_count * sizeof(*delays));
    scales = (double *)malloc(event_count * sizeof(*scales));
    if ((template_count != 0 && templates == NULL)
            || (event_count != 0
                && (ids == NULL || delays == NULL || scales == NULL))) {
        free(templates); free(ids); free(delays); free(scales);
        return NULL;
    }
    for (index = 0; index < left->template_count; ++index)
        templates[index] = left->templates[index];
    for (index = 0; index < right->template_count; ++index)
        templates[left->template_count + index] = right->templates[index];
    for (index = 0; index < left->event_count; ++index) {
        ids[index] = left->template_ids[index];
        delays[index] = wf_add_tick(left->delays[index], left_shift);
        scales[index] = left->scales[index];
    }
    for (index = 0; index < right->event_count; ++index) {
        size_t destination = left->event_count + index;
        ids[destination] = (uint32_t)(left->template_count
                                      + right->template_ids[index]);
        delays[destination] = wf_add_tick(right->delays[index], right_shift);
        scales[destination] = right->scales[index];
    }
    result = cwaveform_stack_create(templates, ids, delays, scales,
                                    template_count, event_count);
    free(templates); free(ids); free(delays); free(scales);
    return result;
}

cwaveform_stack *cwaveform_stack_append(
    const cwaveform_stack *stack, int64_t global_shift,
    const cwaveform_wave *wave, int64_t wave_delay, double wave_scale) {
    cwaveform_wave **templates;
    uint32_t *ids;
    int64_t *delays;
    double *scales;
    cwaveform_stack *result;
    size_t index;
    if (stack == NULL || wave == NULL || !isfinite(wave_scale)
            || stack->template_count == SIZE_MAX
            || stack->event_count == SIZE_MAX
            || stack->template_count >= UINT32_MAX) return NULL;
    templates = (cwaveform_wave **)malloc(
        (stack->template_count + 1) * sizeof(*templates));
    ids = (uint32_t *)malloc((stack->event_count + 1) * sizeof(*ids));
    delays = (int64_t *)malloc((stack->event_count + 1) * sizeof(*delays));
    scales = (double *)malloc((stack->event_count + 1) * sizeof(*scales));
    if (templates == NULL || ids == NULL || delays == NULL || scales == NULL) {
        free(templates); free(ids); free(delays); free(scales);
        return NULL;
    }
    for (index = 0; index < stack->template_count; ++index)
        templates[index] = stack->templates[index];
    templates[stack->template_count] = (cwaveform_wave *)wave;
    for (index = 0; index < stack->event_count; ++index) {
        ids[index] = stack->template_ids[index];
        delays[index] = wf_add_tick(stack->delays[index], global_shift);
        scales[index] = stack->scales[index];
    }
    ids[stack->event_count] = (uint32_t)stack->template_count;
    delays[stack->event_count] = wave_delay;
    scales[stack->event_count] = wave_scale;
    result = cwaveform_stack_create(
        templates, ids, delays, scales,
        stack->template_count + 1, stack->event_count + 1);
    free(templates); free(ids); free(delays); free(scales);
    return result;
}

cwaveform_stack *cwaveform_stack_scale(const cwaveform_stack *stack,
                                       double scale) {
    double *scales;
    cwaveform_stack *result;
    size_t index;
    if (stack == NULL || !isfinite(scale)) return NULL;
    scales = stack->event_count == 0 ? NULL
        : (double *)malloc(stack->event_count * sizeof(*scales));
    if (stack->event_count != 0 && scales == NULL) return NULL;
    for (index = 0; index < stack->event_count; ++index)
        scales[index] = stack->scales[index] * scale;
    result = cwaveform_stack_create(
        stack->templates, stack->template_ids, stack->delays, scales,
        stack->template_count, stack->event_count);
    free(scales);
    return result;
}

cwaveform_stack *cwaveform_stack_from_bytes(const uint8_t *data, size_t size) {
    uint32_t template_count;
    uint32_t event_count;
    size_t event_offset;
    size_t cursor;
    cwaveform_stack *stack = NULL;
    size_t index;
    int have_support = 0;
    if (data == NULL || size < WF_STACK_HEADER_SIZE
            || memcmp(data, "WNS4", 4) != 0
            || wf_get_u16(data + 4) != WF_VERSION
            || wf_get_u16(data + 6) != 0) {
        return NULL;
    }
    template_count = wf_get_u32(data + 8);
    event_count = wf_get_u32(data + 12);
    if ((size_t)template_count > (SIZE_MAX - WF_STACK_HEADER_SIZE) / 4
            || (size_t)event_count > SIZE_MAX / 20
            || WF_STACK_HEADER_SIZE + 4 * (size_t)template_count > size
            || 20 * (size_t)event_count > size) {
        return NULL;
    }
    event_offset = size - 20 * (size_t)event_count;
    cursor = WF_STACK_HEADER_SIZE + 4 * (size_t)template_count;
    if (cursor > event_offset) return NULL;
    stack = wf_stack_allocate(template_count, event_count);
    if (stack == NULL) return NULL;
    for (index = 0; index < template_count; ++index) {
        size_t template_size = wf_get_u32(
            data + WF_STACK_HEADER_SIZE + 4 * index);
        if (template_size > event_offset - cursor) goto invalid;
        stack->templates[index] = cwaveform_wave_from_bytes(data + cursor,
                                                           template_size);
        if (stack->templates[index] == NULL) goto invalid;
        cursor += template_size;
    }
    if (cursor != event_offset) goto invalid;
    for (index = 0; index < event_count; ++index) {
        uint32_t id = wf_get_u32(data + event_offset + 4 * index);
        int64_t delay = wf_get_i64(data + event_offset
                                   + 4 * (size_t)event_count + 8 * index);
        double scale = wf_get_f64(data + event_offset
                                  + 12 * (size_t)event_count + 8 * index);
        if (id >= template_count || !isfinite(scale)) goto invalid;
        stack->template_ids[index] = id;
        stack->delays[index] = delay;
        stack->scales[index] = scale;
        wf_stack_include_support(stack, stack->templates[id], delay,
                                  &have_support);
    }
    /* The input is already a validated WNS4 block. Own a stable copy, without
     * routing through construction and re-encoding the entire event table. */
    stack->data = (uint8_t *)malloc(size);
    if (stack->data == NULL) goto invalid;
    memcpy(stack->data, data, size);
    stack->data_size = size;
    return stack;
invalid:
    cwaveform_stack_release(stack);
    return NULL;
}

void cwaveform_stack_retain(cwaveform_stack *stack) {
    if (stack != NULL) ++stack->references;
}

void cwaveform_stack_release(cwaveform_stack *stack) {
    size_t index;
    if (stack == NULL) return;
    if (stack->references > 1) {
        --stack->references;
        return;
    }
    if (stack->templates != NULL) {
        for (index = 0; index < stack->template_count; ++index) {
            cwaveform_wave_release(stack->templates[index]);
        }
    }
    free(stack->templates);
    free(stack->template_ids);
    free(stack->delays);
    free(stack->scales);
    free(stack->data);
    free(stack);
}

const uint8_t *cwaveform_stack_bytes(const cwaveform_stack *stack,
                                      size_t *size) {
    if (stack == NULL) return NULL;
    if (size != NULL) *size = stack->data_size;
    return stack->data;
}

uint64_t cwaveform_stack_hash(const cwaveform_stack *stack) {
    uint64_t hash;
    if (stack == NULL) return 0;
    /* Sampling never needs a stack hash. Compute it only for explicit hash
     * requests; atomic publication preserves concurrent read-only C use.
     * Zero is a cache-miss sentinel, not a substituted hash: an actual zero
     * hash remains correct and will simply be recomputed on its next use. */
#if defined(_MSC_VER)
    hash = (uint64_t)_InterlockedCompareExchange64(
        (volatile __int64 *)&stack->hash, 0, 0);
#else
    hash = __atomic_load_n(&stack->hash, __ATOMIC_RELAXED);
#endif
    if (hash == 0) {
        hash = wf_hash_bytes(stack->data, stack->data_size);
#if defined(_MSC_VER)
        _InterlockedExchange64((volatile __int64 *)&stack->hash, (__int64)hash);
#else
        __atomic_store_n((uint64_t *)&stack->hash, hash, __ATOMIC_RELAXED);
#endif
    }
    return hash;
}

size_t cwaveform_stack_event_count(const cwaveform_stack *stack) {
    return stack == NULL ? 0 : stack->event_count;
}

size_t cwaveform_stack_template_count(const cwaveform_stack *stack) {
    return stack == NULL ? 0 : stack->template_count;
}

int64_t cwaveform_stack_lower_tick(const cwaveform_stack *stack) {
    return stack == NULL ? INT64_MIN : stack->lower_tick;
}

int64_t cwaveform_stack_upper_tick(const cwaveform_stack *stack) {
    return stack == NULL ? INT64_MAX : stack->upper_tick;
}

static size_t wf_lower_bound(const double *values, size_t count, double target) {
    size_t first = 0;
    while (count != 0) {
        size_t step = count / 2;
        size_t middle = first + step;
        if (values[middle] < target) {
            first = middle + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first;
}

int cwaveform_stack_evaluate(
    const cwaveform_stack *stack, const double *positions, size_t count,
    int64_t global_shift, double offset, double *output) {
    size_t event_index;
    size_t index;
    double ticks = (double)wf_ticks_per_second;
    uint32_t maximum_nodes = 1;
    double *values;
    if (stack == NULL || positions == NULL || output == NULL
            || !isfinite(offset)) return -1;
    for (index = 0; index < stack->template_count; ++index) {
        if (stack->templates[index]->node_count > maximum_nodes)
            maximum_nodes = stack->templates[index]->node_count;
    }
    values = (double *)malloc((size_t)maximum_nodes * sizeof(*values));
    if (values == NULL) return -2;
    for (index = 0; index < count; ++index) output[index] = offset;
    for (event_index = 0; event_index < stack->event_count; ++event_index) {
        cwaveform_wave *wave = stack->templates[stack->template_ids[event_index]];
        int64_t delay_tick = wf_add_tick(stack->delays[event_index], global_shift);
        int64_t lower_tick = wf_add_tick(wave->nodes[wave->root].lower, delay_tick);
        int64_t upper_tick = wf_add_tick(wave->nodes[wave->root].upper, delay_tick);
        double delay = (double)delay_tick / ticks;
        double lower = lower_tick == INT64_MIN ? -DBL_MAX : (double)lower_tick / ticks;
        double upper = upper_tick == INT64_MAX ? DBL_MAX : (double)upper_tick / ticks;
        size_t first = wf_lower_bound(positions, count, lower);
        size_t stop = wf_lower_bound(positions, count, upper);
        /* A global boundary converted to binary64 can round on the opposite
         * side from ``position - delay``.  Include the adjacent candidates;
         * wf_evaluate_one still enforces the exact local half-open support. */
        if (first != 0) --first;
        if (stop != count) ++stop;
        for (index = first; index < stop; ++index) {
            output[index] += stack->scales[event_index]
                * wf_evaluate_one(wave, positions[index] - delay, values);
        }
    }
    free(values);
    return 0;
}

static size_t wf_grid_bound(int64_t boundary, int64_t start_tick,
                            int64_t step_numerator, int64_t step_denominator,
                            size_t count) {
    long double value;
    long double rounded;
    if (boundary == INT64_MIN) return 0;
    if (boundary == INT64_MAX) return count;
    value = (((long double)boundary - (long double)start_tick)
             * (long double)step_denominator) / (long double)step_numerator;
    rounded = ceill(value);
    if (rounded <= 0.0L) return 0;
    if (rounded >= (long double)count) return count;
    return (size_t)rounded;
}

static uint64_t wf_tick_mod(int64_t value, uint64_t modulus) {
    uint64_t magnitude;
    uint64_t remainder;
    if (value >= 0) return (uint64_t)value % modulus;
    magnitude = (uint64_t)(-(value + 1)) + 1;
    remainder = magnitude % modulus;
    return remainder == 0 ? 0 : modulus - remainder;
}

static void wf_plan_scale_group_clear(wf_plan_scale_group *group) {
    if (group == NULL) return;
    free(group->samples);
    free(group->destinations);
    memset(group, 0, sizeof(*group));
}

static void wf_plan_group_clear(wf_plan_group *group) {
    size_t index;
    if (group == NULL) return;
    free(group->samples);
    free(group->destinations);
    free(group->scales);
    for (index = 0; index < group->scale_group_count; ++index)
        wf_plan_scale_group_clear(group->scale_groups + index);
    free(group->scale_groups);
    memset(group, 0, sizeof(*group));
}

static int wf_plan_grow_groups(cwaveform_sample_plan *plan) {
    size_t capacity = plan->group_capacity == 0 ? 4 : plan->group_capacity * 2;
    wf_plan_group *groups;
    if (capacity < plan->group_capacity
            || capacity > SIZE_MAX / sizeof(*groups)) return -1;
    groups = (wf_plan_group *)calloc(capacity, sizeof(*groups));
    if (groups == NULL) return -2;
    if (plan->group_count != 0)
        memcpy(groups, plan->groups,
               plan->group_count * sizeof(*groups));
    free(plan->groups);
    plan->groups = groups;
    plan->group_capacity = capacity;
    return 0;
}

static int wf_plan_group_grow_placements(wf_plan_group *group) {
    size_t capacity = group->placement_capacity == 0
        ? 16 : group->placement_capacity * 2;
    int64_t *destinations;
    double *scales;
    if (capacity < group->placement_capacity
            || capacity > SIZE_MAX / sizeof(*destinations)
            || capacity > SIZE_MAX / sizeof(*scales)) return -1;
    destinations = (int64_t *)malloc(capacity * sizeof(*destinations));
    scales = (double *)malloc(capacity * sizeof(*scales));
    if (destinations == NULL || scales == NULL) {
        free(destinations);
        free(scales);
        return -2;
    }
    if (group->placement_count != 0) {
        memcpy(destinations, group->destinations,
               group->placement_count * sizeof(*destinations));
        memcpy(scales, group->scales,
               group->placement_count * sizeof(*scales));
    }
    free(group->destinations);
    free(group->scales);
    group->destinations = destinations;
    group->scales = scales;
    group->placement_capacity = capacity;
    return 0;
}

static int wf_plan_scale_group_append(wf_plan_scale_group *group,
                                      int64_t destination) {
    if (group->destination_count == group->destination_capacity) {
        size_t capacity = group->destination_capacity == 0
            ? 16 : group->destination_capacity * 2;
        int64_t *destinations;
        if (capacity < group->destination_capacity
                || capacity > SIZE_MAX / sizeof(*destinations)) return -1;
        destinations = (int64_t *)realloc(
            group->destinations, capacity * sizeof(*destinations));
        if (destinations == NULL) return -2;
        group->destinations = destinations;
        group->destination_capacity = capacity;
    }
    group->destinations[group->destination_count++] = destination;
    return 0;
}

static void wf_plan_group_disable_scale_groups(wf_plan_group *group) {
    size_t index;
    for (index = 0; index < group->scale_group_count; ++index)
        wf_plan_scale_group_clear(group->scale_groups + index);
    free(group->scale_groups);
    group->scale_groups = NULL;
    group->scale_group_count = 0;
    group->scale_group_capacity = 0;
    group->grouped_scales = 0;
}

static int wf_plan_group_add_scale_destination(wf_plan_group *group,
                                               double scale,
                                               int64_t destination) {
    size_t index;
    wf_plan_scale_group *scale_group;
    if (!group->grouped_scales) return 0;
    for (index = 0; index < group->scale_group_count; ++index) {
        if (group->scale_groups[index].scale == scale)
            return wf_plan_scale_group_append(group->scale_groups + index,
                                              destination);
    }
    if (group->scale_group_count == 64) {
        wf_plan_group_disable_scale_groups(group);
        return 0;
    }
    if (group->scale_group_count == group->scale_group_capacity) {
        size_t capacity = group->scale_group_capacity == 0
            ? 4 : group->scale_group_capacity * 2;
        wf_plan_scale_group *scale_groups;
        if (capacity > SIZE_MAX / sizeof(*scale_groups)) return -1;
        scale_groups = (wf_plan_scale_group *)calloc(
            capacity, sizeof(*scale_groups));
        if (scale_groups == NULL) return -2;
        if (group->scale_group_count != 0)
            memcpy(scale_groups, group->scale_groups,
                   group->scale_group_count * sizeof(*scale_groups));
        free(group->scale_groups);
        group->scale_groups = scale_groups;
        group->scale_group_capacity = capacity;
    }
    scale_group = group->scale_groups + group->scale_group_count++;
    scale_group->scale = scale;
    if (group->sample_count != 0) {
        if (group->sample_count > SIZE_MAX / sizeof(*scale_group->samples))
            return -1;
        scale_group->samples = (double *)malloc(
            group->sample_count * sizeof(*scale_group->samples));
        if (scale_group->samples == NULL) return -2;
        for (index = 0; index < group->sample_count; ++index)
            scale_group->samples[index] = scale * group->samples[index];
    }
    return wf_plan_scale_group_append(scale_group, destination);
}

static int wf_plan_group_add_placement(wf_plan_group *group,
                                       int64_t destination, double scale) {
    int status;
    if (group->placement_count == group->placement_capacity) {
        status = wf_plan_group_grow_placements(group);
        if (status != 0) return status;
    }
    group->destinations[group->placement_count] = destination;
    group->scales[group->placement_count] = scale;
    ++group->placement_count;
    return wf_plan_group_add_scale_destination(group, scale, destination);
}

static int wf_plan_placement_bounds(
    int64_t destination, size_t template_count, size_t output_count,
    size_t *source_start, size_t *output_start, size_t *copy_count) {
    size_t available;
    *copy_count = 0;
    if (destination < 0) {
        /* Negating INT64_MIN, or adding length to a signed destination, can
         * overflow. Clip using unsigned distances before converting to size_t. */
        uint64_t skipped = (uint64_t)(-(destination + 1)) + 1;
        if (skipped >= template_count) return 0;
        *source_start = (size_t)skipped;
        *output_start = 0;
    } else {
        if ((uint64_t)destination >= output_count) return 0;
        *source_start = 0;
        *output_start = (size_t)destination;
    }
    available = output_count - *output_start;
    *copy_count = template_count - *source_start;
    if (*copy_count > available) *copy_count = available;
    return *copy_count != 0;
}

typedef struct wf_plan_lookup {
    size_t capacity;
    size_t *slots;
} wf_plan_lookup;

static size_t wf_plan_group_hash(uint32_t template_id, uint64_t phase) {
    uint64_t value = phase ^ ((uint64_t)template_id * UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return (size_t)(value ^ (value >> 31));
}

static int wf_plan_lookup_grow(wf_plan_lookup *lookup,
                                const cwaveform_sample_plan *plan) {
    size_t capacity = lookup->capacity == 0 ? 32 : lookup->capacity * 2;
    size_t *slots;
    size_t index;
    if (capacity < lookup->capacity || capacity > SIZE_MAX / sizeof(*slots))
        return -1;
    slots = (size_t *)calloc(capacity, sizeof(*slots));
    if (slots == NULL) return -2;
    for (index = 0; index < plan->group_count; ++index) {
        const wf_plan_group *group = plan->groups + index;
        size_t slot = wf_plan_group_hash(group->template_id, group->phase)
            & (capacity - 1);
        while (slots[slot] != 0) slot = (slot + 1) & (capacity - 1);
        slots[slot] = index + 1;
    }
    free(lookup->slots);
    lookup->slots = slots;
    lookup->capacity = capacity;
    return 0;
}

/* Keep template evaluation out of the per-event plan-building loop. Inlining
 * the vector/scalar alternatives there increases register pressure even when
 * all events reuse one already-sampled template (GCC 9 / Zen 4). */
WF_NOINLINE
static void wf_plan_fill_template(const cwaveform_wave *wave,
                                  wf_plan_group *group, int64_t step_numerator,
                                  double *values) {
    size_t index;
#if (defined(__APPLE__) || (defined(__GLIBC__) && defined(__x86_64__) \
        && defined(WF_HAVE_X86_SIMD) && !defined(WF_DISABLE_X86_SIMD))) \
        && !defined(WF_DISABLE_BATCH_EVALUATOR)
    /* Reuse the sample buffer for positions: each vector chunk consumes all
     * positions before publishing its output. Small/non-DRAG templates keep
     * their scalar preparation path. */
    if (group->sample_count >= 32) {
        uint32_t node_index;
        int has_drag = 0;
        for (node_index = 0; node_index < wave->node_count; ++node_index)
            if (wave->nodes[node_index].op == WF_OP_DRAG) has_drag = 1;
        if (has_drag) {
            for (index = 0; index < group->sample_count; ++index) {
                long double tick = (long double)group->first_tick
                    + (long double)index * (long double)step_numerator;
                group->samples[index] =
                    (double)(tick / (long double)wf_ticks_per_second);
            }
            if (wf_evaluate_many_vector(wave, group->samples,
                    group->sample_count, 0.0, 1.0, -HUGE_VAL, HUGE_VAL,
                    group->samples) == 0) return;
            /* Unsupported nodes/allocation failure: recompute every sample,
             * including any vector chunks which were already written. */
        }
    }
#endif
    for (index = 0; index < group->sample_count; ++index) {
        long double tick = (long double)group->first_tick
            + (long double)index * (long double)step_numerator;
        group->samples[index] = wf_evaluate_one(
            wave, (double)(tick / (long double)wf_ticks_per_second), values);
    }
}

static wf_plan_group *wf_plan_get_group(
    cwaveform_sample_plan *plan, const cwaveform_stack *stack,
    uint32_t template_id, uint64_t phase, int64_t step_numerator,
    wf_plan_lookup *lookup) {
    size_t index;
    size_t slot = 0;
    wf_plan_group *group;
    cwaveform_wave *wave;
    int64_t lower;
    int64_t upper;
    uint64_t lower_phase;
    uint64_t delta;
    long double sample_count;
    double *values;
    if (plan->group_count != 0) {
        /* Consecutive events very often reuse one template and tick phase.
         * Keep that case independent of the lookup-table bookkeeping. */
        group = plan->groups + plan->group_count - 1;
        if (group->template_id == template_id && group->phase == phase)
            return group;
    }
    if (lookup->capacity == 0 && plan->group_count <= 8) {
        /* Avoid a hash table for the usual one/few-template pulse train. */
        for (index = 0; index < plan->group_count; ++index) {
            group = plan->groups + index;
            if (group->template_id == template_id && group->phase == phase)
                return group;
        }
    } else {
        if ((lookup->capacity == 0 || plan->group_count >= lookup->capacity / 2)
                && wf_plan_lookup_grow(lookup, plan) != 0) return NULL;
        slot = wf_plan_group_hash(template_id, phase) & (lookup->capacity - 1);
        while (lookup->slots[slot] != 0) {
            group = plan->groups + lookup->slots[slot] - 1;
            if (group->template_id == template_id && group->phase == phase)
                return group;
            slot = (slot + 1) & (lookup->capacity - 1);
        }
    }
    if (plan->group_count == plan->group_capacity
            && wf_plan_grow_groups(plan) != 0) return NULL;
    group = plan->groups + plan->group_count;
    memset(group, 0, sizeof(*group));
    group->template_id = template_id;
    group->phase = phase;
    group->grouped_scales = 1;
    wave = stack->templates[template_id];
    lower = wave->nodes[wave->root].lower;
    upper = wave->nodes[wave->root].upper;
    if (lower == INT64_MIN || upper == INT64_MAX || lower >= upper)
        return NULL;
    lower_phase = wf_tick_mod(lower, (uint64_t)step_numerator);
    delta = phase >= lower_phase
        ? phase - lower_phase
        : (uint64_t)step_numerator - (lower_phase - phase);
    group->first_tick = wf_add_tick(lower, (int64_t)delta);
    if (group->first_tick >= upper) {
        group->sample_count = 0;
    } else {
        sample_count = ceill(
            ((long double)upper - (long double)group->first_tick)
            / (long double)step_numerator);
        if (sample_count < 0.0L || sample_count > (long double)SIZE_MAX)
            return NULL;
        group->sample_count = (size_t)sample_count;
    }
    if (group->sample_count != 0) {
        if (group->sample_count > SIZE_MAX / sizeof(*group->samples))
            return NULL;
        group->samples = (double *)malloc(
            group->sample_count * sizeof(*group->samples));
        values = (double *)malloc((size_t)wave->node_count * sizeof(*values));
        if (group->samples == NULL || values == NULL) {
            free(values);
            wf_plan_group_clear(group);
            return NULL;
        }
        wf_plan_fill_template(wave, group, step_numerator, values);
        free(values);
    }
    ++plan->group_count;
    if (lookup->capacity != 0) lookup->slots[slot] = plan->group_count;
    return group;
}

cwaveform_sample_plan *cwaveform_sample_plan_create(
    const cwaveform_stack *stack, int64_t start_tick, size_t count,
    int64_t step_numerator, int64_t step_denominator,
    int64_t global_shift) {
    cwaveform_sample_plan *plan;
    size_t event_index;
    size_t previous_start = 0;
    size_t previous_stop = 0;
    int have_previous = 0;
    wf_plan_lookup lookup = {0, NULL};
    if (stack == NULL || step_numerator <= 0 || step_denominator != 1)
        return NULL;
    plan = (cwaveform_sample_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) return NULL;
    plan->references = 1;
    plan->count = count;
    plan->non_overlapping = 1;
    for (event_index = 0; event_index < stack->event_count; ++event_index) {
        uint32_t template_id = stack->template_ids[event_index];
        cwaveform_wave *wave = stack->templates[template_id];
        int64_t delay;
        uint64_t start_phase;
        uint64_t delay_phase;
        uint64_t phase;
        wf_plan_group *group;
        long double destination_value;
        int64_t destination;
        size_t source_start;
        size_t output_start;
        size_t copy_count;
        int status;
        if (stack->scales[event_index] == 0.0
                || wave->nodes[wave->root].lower
                   >= wave->nodes[wave->root].upper) continue;
        if (wave->nodes[wave->root].lower == INT64_MIN
                || wave->nodes[wave->root].upper == INT64_MAX) {
            goto failed;
        }
        delay = wf_add_tick(stack->delays[event_index], global_shift);
        start_phase = wf_tick_mod(start_tick, (uint64_t)step_numerator);
        delay_phase = wf_tick_mod(delay, (uint64_t)step_numerator);
        phase = start_phase >= delay_phase
            ? start_phase - delay_phase
            : (uint64_t)step_numerator - (delay_phase - start_phase);
        group = wf_plan_get_group(plan, stack, template_id, phase,
                                  step_numerator, &lookup);
        if (group == NULL) goto failed;
        destination_value = (
            (long double)delay + (long double)group->first_tick
            - (long double)start_tick) / (long double)step_numerator;
        if (destination_value < (long double)INT64_MIN
                || destination_value > (long double)INT64_MAX) {
            goto failed;
        }
        destination = (int64_t)llroundl(destination_value);
        status = wf_plan_group_add_placement(
            group, destination, stack->scales[event_index]);
        if (status != 0) goto failed;
        if (wf_plan_placement_bounds(
                destination, group->sample_count, count,
                &source_start, &output_start, &copy_count)) {
            size_t stop = output_start + copy_count;
            if (have_previous
                    && (output_start < previous_start
                        || output_start < previous_stop))
                plan->non_overlapping = 0;
            previous_start = output_start;
            previous_stop = stop > previous_stop ? stop : previous_stop;
            have_previous = 1;
        }
    }
    free(lookup.slots);
    return plan;
failed:
    free(lookup.slots);
    cwaveform_sample_plan_release(plan);
    return NULL;
}

void cwaveform_sample_plan_retain(cwaveform_sample_plan *plan) {
    if (plan != NULL) ++plan->references;
}

void cwaveform_sample_plan_release(cwaveform_sample_plan *plan) {
    size_t index;
    if (plan == NULL) return;
    if (plan->references > 1) {
        --plan->references;
        return;
    }
    for (index = 0; index < plan->group_count; ++index)
        wf_plan_group_clear(plan->groups + index);
    free(plan->groups);
    free(plan);
}

size_t cwaveform_sample_plan_count(const cwaveform_sample_plan *plan) {
    return plan == NULL ? 0 : plan->count;
}

size_t cwaveform_sample_plan_group_count(const cwaveform_sample_plan *plan) {
    return plan == NULL ? 0 : plan->group_count;
}

int cwaveform_sample_plan_non_overlapping(
    const cwaveform_sample_plan *plan) {
    return plan != NULL && plan->non_overlapping;
}

static double wf_clip_amplitude(double value, double lower, double upper) {
    if (value < lower) value = lower;
    if (value > upper) value = upper;
    return value;
}

int cwaveform_sos_filter(
    const double *sos, size_t section_count,
    const double *input, size_t count, size_t input_stride,
    double *state, size_t state_stride, double initial,
    double lower, double upper, int dtype, double full_scale,
    void *output, size_t output_stride) {
    double values[256];
    size_t section;
    size_t cursor;
    size_t item_size;
    if ((count != 0 && (input == NULL || output == NULL))
            || (section_count != 0 && (sos == NULL || state == NULL))
            || input_stride == 0 || state_stride == 0 || output_stride == 0
            || !(lower <= upper) || lower == HUGE_VAL || upper == -HUGE_VAL)
        return -1;
    if (dtype == CWAVEFORM_FLOAT64) {
        item_size = sizeof(double);
    } else if (dtype == CWAVEFORM_INT16 || dtype == CWAVEFORM_INT32) {
        if (output_stride != 1 || !isfinite(full_scale) || full_scale <= 0.0)
            return -1;
        item_size = dtype == CWAVEFORM_INT16 ? sizeof(int16_t) : sizeof(int32_t);
    } else return -1;
    for (section = 0; section < section_count; ++section)
        if (sos[6 * section + 3] != 1.0) return -1;

    for (cursor = 0; cursor < count; ) {
        size_t size = count - cursor;
        size_t index;
        if (size > 256) size = 256;
        for (index = 0; index < size; ++index)
            values[index] = input[(cursor + index) * input_stride] - initial;
        /* A single section keeps all coefficients and delays in registers.
         * Cascades advance sample first, allowing the processor to overlap
         * delay updates in successive sections. */
        if (section_count == 1) {
            const double *coeff = sos;
            double b0 = coeff[0], b1 = coeff[1], b2 = coeff[2];
            double a1 = coeff[4], a2 = coeff[5];
            double z0 = state[0];
            double z1 = state[state_stride];
            for (index = 0; index < size; ++index) {
                double x = values[index];
                double y = b0 * x + z0;
                z0 = b1 * x - a1 * y + z1;
                z1 = b2 * x - a2 * y;
                values[index] = y;
            }
            state[0] = z0;
            state[state_stride] = z1;
        } else {
            for (index = 0; index < size; ++index) {
                double x = values[index];
                for (section = 0; section < section_count; ++section) {
                    const double *coeff = sos + 6 * section;
                    double *delay = state + 2 * section * state_stride;
                    double y = coeff[0] * x + delay[0];
                    delay[0] = coeff[1] * x - coeff[4] * y
                             + delay[state_stride];
                    delay[state_stride] = coeff[2] * x - coeff[5] * y;
                    x = y;
                }
                values[index] = x;
            }
        }
        if (dtype == CWAVEFORM_FLOAT64) {
            for (index = 0; index < size; ++index)
                ((double *)output)[(cursor + index) * output_stride]
                    = wf_clip_amplitude(values[index] + initial, lower, upper);
        } else {
            int status;
            for (index = 0; index < size; ++index)
                values[index] = wf_clip_amplitude(
                    values[index] + initial, lower, upper);
            status = wf_quantize_array(values, size, dtype, full_scale,
                (uint8_t *)output + cursor * item_size);
            if (status != 0) return status;
        }
        cursor += size;
    }
    return 0;
}

static void wf_plan_sample_float(const cwaveform_sample_plan *plan,
                                 double offset, double *output,
                                 double lower, double upper) {
    size_t group_index;
    size_t index;
    int limited = lower != -HUGE_VAL || upper != HUGE_VAL;
    double base = plan->non_overlapping
        ? wf_clip_amplitude(offset, lower, upper) : offset;
    for (index = 0; index < plan->count; ++index) output[index] = base;
    for (group_index = 0; group_index < plan->group_count; ++group_index) {
        const wf_plan_group *group = plan->groups + group_index;
        if (group->grouped_scales && plan->non_overlapping) {
            size_t scale_index;
            int transform = offset != 0.0 || limited;
            double *transformed = (!transform || group->sample_count == 0)
                ? NULL
                : (double *)malloc(group->sample_count * sizeof(double));
            if (transform && group->sample_count != 0
                    && transformed == NULL) goto generic;
            for (scale_index = 0;
                    scale_index < group->scale_group_count; ++scale_index) {
                const wf_plan_scale_group *scale_group
                    = group->scale_groups + scale_index;
                const double *source_values = scale_group->samples;
                size_t destination_index;
                if (transform) {
                    for (index = 0; index < group->sample_count; ++index)
                        transformed[index] = wf_clip_amplitude(
                            offset + scale_group->samples[index], lower, upper);
                    source_values = transformed;
                }
                for (destination_index = 0;
                        destination_index < scale_group->destination_count;
                        ++destination_index) {
                    size_t source_start;
                    size_t output_start;
                    size_t copy_count;
                    if (wf_plan_placement_bounds(
                            scale_group->destinations[destination_index],
                            group->sample_count, plan->count,
                            &source_start, &output_start, &copy_count))
                        memcpy(output + output_start,
                               source_values + source_start,
                               copy_count * sizeof(double));
                }
            }
            free(transformed);
            continue;
        }
        if (group->grouped_scales) {
            size_t scale_index;
            for (scale_index = 0;
                    scale_index < group->scale_group_count; ++scale_index) {
                const wf_plan_scale_group *scale_group
                    = group->scale_groups + scale_index;
                size_t destination_index;
                for (destination_index = 0;
                        destination_index < scale_group->destination_count;
                        ++destination_index) {
                    size_t source_start;
                    size_t output_start;
                    size_t copy_count;
                    size_t sample_index;
                    if (!wf_plan_placement_bounds(
                            scale_group->destinations[destination_index],
                            group->sample_count, plan->count,
                            &source_start, &output_start, &copy_count)) continue;
                    for (sample_index = 0; sample_index < copy_count;
                            ++sample_index)
                        output[output_start + sample_index]
                            += scale_group->samples[source_start + sample_index];
                }
            }
            continue;
        }
generic:
        for (index = 0; index < group->placement_count; ++index) {
            size_t source_start;
            size_t output_start;
            size_t copy_count;
            size_t sample_index;
            if (!wf_plan_placement_bounds(
                    group->destinations[index], group->sample_count,
                    plan->count, &source_start, &output_start, &copy_count))
                continue;
            if (plan->non_overlapping) {
                if (limited) {
                    for (sample_index = 0; sample_index < copy_count; ++sample_index)
                        output[output_start + sample_index] = wf_clip_amplitude(
                            offset + group->scales[index]
                            * group->samples[source_start + sample_index], lower, upper);
                } else {
                    for (sample_index = 0; sample_index < copy_count; ++sample_index)
                        output[output_start + sample_index] = offset
                            + group->scales[index]
                            * group->samples[source_start + sample_index];
                }
            } else {
                for (sample_index = 0; sample_index < copy_count; ++sample_index)
                    output[output_start + sample_index] += group->scales[index]
                        * group->samples[source_start + sample_index];
            }
        }
    }
    /* Overlapping events must be accumulated before any amplitude limit. */
    if (limited && !plan->non_overlapping) {
        for (index = 0; index < plan->count; ++index)
            output[index] = wf_clip_amplitude(output[index], lower, upper);
    }
}

static int wf_plan_sample_integer_grouped(
    const cwaveform_sample_plan *plan, const wf_plan_group *group,
    double offset, int dtype, double full_scale, void *output,
    double lower, double upper) {
    size_t scale_index;
    size_t item_size = dtype == CWAVEFORM_INT16
        ? sizeof(int16_t) : sizeof(int32_t);
    void *quantized = group->sample_count == 0 ? NULL
        : malloc(group->sample_count * item_size);
    if (group->sample_count != 0 && quantized == NULL) return -2;
    for (scale_index = 0;
            scale_index < group->scale_group_count; ++scale_index) {
        const wf_plan_scale_group *scale_group
            = group->scale_groups + scale_index;
        size_t index;
        int status = 0;
        if (offset == 0.0 && lower == -HUGE_VAL && upper == HUGE_VAL) {
            status = wf_quantize_array(scale_group->samples,
                group->sample_count, dtype, full_scale, quantized);
        } else {
            double values[256];
            size_t cursor = 0;
            while (cursor < group->sample_count) {
                size_t count = group->sample_count - cursor;
                if (count > 256) count = 256;
                for (index = 0; index < count; ++index)
                    values[index] = wf_clip_amplitude(offset
                        + scale_group->samples[cursor + index], lower, upper);
                status = wf_quantize_array(values, count, dtype, full_scale,
                    (uint8_t *)quantized + cursor * item_size);
                if (status != 0) break;
                cursor += count;
            }
        }
        if (status != 0) {
            free(quantized);
            return status;
        }
        for (index = 0; index < scale_group->destination_count; ++index) {
            size_t source_start;
            size_t output_start;
            size_t copy_count;
            if (wf_plan_placement_bounds(
                    scale_group->destinations[index], group->sample_count,
                    plan->count, &source_start, &output_start, &copy_count))
                memcpy((uint8_t *)output + output_start * item_size,
                       (uint8_t *)quantized + source_start * item_size,
                       copy_count * item_size);
        }
    }
    free(quantized);
    return 0;
}

static int wf_plan_sample_integer_varied_scales(
    const cwaveform_sample_plan *plan, const wf_plan_group *group,
    double offset, int dtype, double full_scale, void *output,
    double lower, double upper) {
    /* Many distinct scales cannot reuse prequantized templates. Small scratch
     * blocks still allow SIMD clipping/conversion without a full float output. */
    double values[256];
    int limited = lower != -HUGE_VAL || upper != HUGE_VAL;
    size_t item_size = dtype == CWAVEFORM_INT16
        ? sizeof(int16_t) : sizeof(int32_t);
    size_t placement_index;
    for (placement_index = 0; placement_index < group->placement_count;
            ++placement_index) {
        size_t source_start;
        size_t output_start;
        size_t copy_count;
        size_t cursor;
        double scale = group->scales[placement_index];
        if (!wf_plan_placement_bounds(
                group->destinations[placement_index], group->sample_count,
                plan->count, &source_start, &output_start, &copy_count)) continue;
        for (cursor = 0; cursor < copy_count; ) {
            size_t count = copy_count - cursor;
            size_t index;
            int status;
            if (count > 256) count = 256;
            if (limited) {
                for (index = 0; index < count; ++index)
                    values[index] = wf_clip_amplitude(offset + scale
                        * group->samples[source_start + cursor + index], lower, upper);
            } else {
                for (index = 0; index < count; ++index)
                    values[index] = offset + scale
                        * group->samples[source_start + cursor + index];
            }
            status = wf_quantize_array(values, count, dtype, full_scale,
                (uint8_t *)output + (output_start + cursor) * item_size);
            if (status != 0) return status;
            cursor += count;
        }
    }
    return 0;
}

int cwaveform_sample_plan_sample(
    const cwaveform_sample_plan *plan, double offset, int dtype,
    double full_scale, void *output) {
    return cwaveform_sample_plan_sample_clipped(
        plan, offset, dtype, full_scale, output, -HUGE_VAL, HUGE_VAL);
}

int cwaveform_sample_plan_sample_clipped(
    const cwaveform_sample_plan *plan, double offset, int dtype,
    double full_scale, void *output, double lower, double upper) {
    size_t index;
    if (plan == NULL || output == NULL || !isfinite(offset)
            || !(lower <= upper) || lower == HUGE_VAL || upper == -HUGE_VAL
            || !isfinite(full_scale) || full_scale <= 0.0) return -1;
    if (dtype == CWAVEFORM_FLOAT64) {
        wf_plan_sample_float(plan, offset, (double *)output, lower, upper);
        return 0;
    }
    if (dtype != CWAVEFORM_INT16 && dtype != CWAVEFORM_INT32) return -1;
    if (!plan->non_overlapping) {
        double *values = plan->count == 0 ? NULL
            : (double *)malloc(plan->count * sizeof(double));
        int status;
        if (plan->count != 0 && values == NULL) return -2;
        wf_plan_sample_float(plan, offset, values, lower, upper);
        status = wf_quantize_array(
            values, plan->count, dtype, full_scale, output);
        free(values);
        return status;
    }
    if (dtype == CWAVEFORM_INT16) {
        int16_t base = wf_quantize16(
            wf_clip_amplitude(offset, lower, upper), full_scale);
        for (index = 0; index < plan->count; ++index)
            ((int16_t *)output)[index] = base;
    } else {
        int32_t base = wf_quantize32(
            wf_clip_amplitude(offset, lower, upper), full_scale);
        for (index = 0; index < plan->count; ++index)
            ((int32_t *)output)[index] = base;
    }
    for (index = 0; index < plan->group_count; ++index) {
        const wf_plan_group *group = plan->groups + index;
        if (group->grouped_scales) {
            int status = wf_plan_sample_integer_grouped(
                plan, group, offset, dtype, full_scale, output, lower, upper);
            if (status != 0) return status;
        } else {
            int status = wf_plan_sample_integer_varied_scales(
                plan, group, offset, dtype, full_scale, output, lower, upper);
            if (status != 0) return status;
        }
    }
    return 0;
}

int cwaveform_stack_sample(
    const cwaveform_stack *stack, int64_t start_tick, size_t count,
    int64_t step_numerator, int64_t step_denominator, int64_t global_shift,
    double offset, int dtype, double full_scale, void *output) {
    size_t event_index;
    size_t index;
    uint32_t maximum_nodes = 1;
    double *values;
    int non_overlapping = 1;
    int64_t previous_lower = INT64_MIN;
    int64_t previous_upper = INT64_MIN;
    if (stack == NULL || output == NULL || step_numerator <= 0
            || step_denominator <= 0 || !isfinite(offset)
            || !isfinite(full_scale) || full_scale <= 0.0) return -1;
    for (index = 0; index < stack->template_count; ++index) {
        if (stack->templates[index]->node_count > maximum_nodes)
            maximum_nodes = stack->templates[index]->node_count;
    }
    values = (double *)malloc((size_t)maximum_nodes * sizeof(*values));
    if (values == NULL) return -2;
    for (event_index = 0; event_index < stack->event_count; ++event_index) {
        cwaveform_wave *wave = stack->templates[stack->template_ids[event_index]];
        int64_t delay = wf_add_tick(stack->delays[event_index], global_shift);
        int64_t lower = wf_add_tick(wave->nodes[wave->root].lower, delay);
        int64_t upper = wf_add_tick(wave->nodes[wave->root].upper, delay);
        if (stack->scales[event_index] == 0.0 || lower >= upper) continue;
        if (lower < previous_lower || lower < previous_upper) non_overlapping = 0;
        previous_lower = lower;
        previous_upper = upper;
    }
    if (dtype == CWAVEFORM_FLOAT64 || !non_overlapping) {
        double *float_output = dtype == CWAVEFORM_FLOAT64
            ? (double *)output : (double *)malloc(count * sizeof(double));
        if (float_output == NULL) {
            free(values);
            return -2;
        }
        for (index = 0; index < count; ++index) float_output[index] = offset;
        for (event_index = 0; event_index < stack->event_count; ++event_index) {
            cwaveform_wave *wave = stack->templates[stack->template_ids[event_index]];
            int64_t delay_tick = wf_add_tick(stack->delays[event_index], global_shift);
            int64_t lower_tick = wf_add_tick(wave->nodes[wave->root].lower, delay_tick);
            int64_t upper_tick = wf_add_tick(wave->nodes[wave->root].upper, delay_tick);
            size_t first = wf_grid_bound(lower_tick, start_tick, step_numerator,
                                         step_denominator, count);
            size_t stop = wf_grid_bound(upper_tick, start_tick, step_numerator,
                                        step_denominator, count);
            for (index = first; index < stop; ++index) {
                double position = wf_local_grid_position(
                    start_tick, index, step_numerator, step_denominator,
                    delay_tick);
                float_output[index] += stack->scales[event_index]
                    * wf_evaluate_one(wave, position, values);
            }
        }
        if (dtype == CWAVEFORM_INT16 || dtype == CWAVEFORM_INT32) {
            int status = wf_quantize_array(
                float_output, count, dtype, full_scale, output);
            if (status != 0) {
                free(float_output);
                free(values);
                return status;
            }
        } else if (dtype != CWAVEFORM_FLOAT64) {
            if (float_output != output) free(float_output);
            free(values);
            return -1;
        }
        if ((void *)float_output != output) free(float_output);
    } else {
        if (dtype == CWAVEFORM_INT16) {
            int16_t base = wf_quantize16(offset, full_scale);
            for (index = 0; index < count; ++index) ((int16_t *)output)[index] = base;
        } else if (dtype == CWAVEFORM_INT32) {
            int32_t base = wf_quantize32(offset, full_scale);
            for (index = 0; index < count; ++index) ((int32_t *)output)[index] = base;
        } else {
            free(values);
            return -1;
        }
        for (event_index = 0; event_index < stack->event_count; ++event_index) {
            cwaveform_wave *wave = stack->templates[stack->template_ids[event_index]];
            int64_t delay_tick = wf_add_tick(stack->delays[event_index], global_shift);
            int64_t lower_tick = wf_add_tick(wave->nodes[wave->root].lower, delay_tick);
            int64_t upper_tick = wf_add_tick(wave->nodes[wave->root].upper, delay_tick);
            size_t first = wf_grid_bound(lower_tick, start_tick, step_numerator,
                                         step_denominator, count);
            size_t stop = wf_grid_bound(upper_tick, start_tick, step_numerator,
                                        step_denominator, count);
            for (index = first; index < stop; ++index) {
                double position = wf_local_grid_position(
                    start_tick, index, step_numerator, step_denominator,
                    delay_tick);
                double value = offset + stack->scales[event_index]
                    * wf_evaluate_one(wave, position, values);
                if (!isfinite(value)) {
                    free(values);
                    return -3;
                }
                if (dtype == CWAVEFORM_INT16)
                    ((int16_t *)output)[index] = wf_quantize16(value, full_scale);
                else
                    ((int32_t *)output)[index] = wf_quantize32(value, full_scale);
            }
        }
    }
    free(values);
    return 0;
}

cwaveform_wave *cwaveform_stack_simplify(const cwaveform_stack *stack,
                                         int64_t global_shift,
                                         double offset) {
    size_t event_index;
    size_t capacity = offset == 0.0 ? 1 : 3;
    size_t parameter_count = 0;
    size_t parameter_cursor = 0;
    uint32_t next = 0;
    uint32_t root = UINT32_MAX;
    wf_node *nodes;
    double *parameters;
    cwaveform_wave *result;
    if (stack == NULL || !isfinite(offset)) return NULL;
    for (event_index = 0; event_index < stack->event_count; ++event_index) {
        cwaveform_wave *wave = stack->templates[stack->template_ids[event_index]];
        if ((size_t)wave->node_count > SIZE_MAX - capacity - 2
                || wave->parameter_count > SIZE_MAX - parameter_count)
            return NULL;
        capacity += wave->node_count + 2;
        parameter_count += wave->parameter_count;
    }
    if (capacity > UINT32_MAX || parameter_count > UINT32_MAX) return NULL;
    nodes = (wf_node *)calloc(capacity, sizeof(*nodes));
    parameters = parameter_count == 0 ? NULL
        : (double *)malloc(parameter_count * sizeof(double));
    if (nodes == NULL || (parameter_count != 0 && parameters == NULL)) {
        free(nodes);
        free(parameters);
        return NULL;
    }
    for (event_index = 0; event_index < stack->event_count; ++event_index) {
        cwaveform_wave *wave = stack->templates[stack->template_ids[event_index]];
        uint32_t node_offset = next;
        uint32_t event_root;
        if (wave->parameter_count != 0)
            memcpy(parameters + parameter_cursor, wave->parameters,
                   wave->parameter_count * sizeof(double));
        event_root = wf_clone_affine(
            wave, nodes, node_offset, (uint32_t)parameter_cursor,
            wf_add_tick(stack->delays[event_index], global_shift),
            stack->scales[event_index], &next);
        parameter_cursor += wave->parameter_count;
        if (root == UINT32_MAX) {
            root = event_root;
        } else {
            wf_node node;
            memset(&node, 0, sizeof(node));
            node.op = WF_OP_ADD;
            node.left = root;
            node.right = event_root;
            wf_union_support(nodes[root].lower, nodes[root].upper,
                             nodes[event_root].lower, nodes[event_root].upper,
                             &node.lower, &node.upper);
            nodes[next] = node;
            root = next++;
        }
    }
    if (offset != 0.0) {
        wf_node constant;
        memset(&constant, 0, sizeof(constant));
        constant.op = WF_OP_CONSTANT;
        constant.p0 = offset;
        constant.lower = INT64_MIN;
        constant.upper = INT64_MAX;
        nodes[next] = constant;
        if (root == UINT32_MAX) {
            root = next++;
        } else {
            wf_node node;
            uint32_t constant_root = next++;
            memset(&node, 0, sizeof(node));
            node.op = WF_OP_ADD;
            node.left = root;
            node.right = constant_root;
            node.lower = INT64_MIN;
            node.upper = INT64_MAX;
            nodes[next] = node;
            root = next++;
        }
    }
    if (root == UINT32_MAX) {
        memset(nodes, 0, sizeof(*nodes));
        nodes[0].op = WF_OP_CONSTANT;
        nodes[0].lower = INT64_MAX;
        nodes[0].upper = INT64_MIN;
        root = 0;
        next = 1;
    }
    result = wf_wave_from_parts(nodes, next, root,
                                parameters, parameter_count);
    free(nodes);
    free(parameters);
    return result;
}

const char *cwaveform_format_description(void) {
    return "WNF4/WNS4 waveform and NLM1 nonlinear-map little-endian immutable blocks; global integer ticks; ABI 2";
}
