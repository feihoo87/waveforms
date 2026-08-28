#define CWAVEFORM_BUILD 1
#include "_cwaveform.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#define WF_DEFAULT_TICKS_PER_SECOND UINT64_C(120000000000)
#define WF_WAVE_HEADER_SIZE 24u
#define WF_NODE_SIZE 28u
#define WF_STACK_HEADER_SIZE 16u
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
};

struct cwaveform_stack {
    uint32_t references;
    size_t template_count;
    size_t event_count;
    cwaveform_wave **templates;
    uint32_t *template_ids;
    int64_t *delays;
    double *scales;
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

static double wf_node_parameter(const cwaveform_wave *wave,
                                const wf_node *node, size_t index);

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

static void wf_make_zero_node(wf_node *node) {
    memset(node, 0, sizeof(*node));
    node->op = WF_OP_CONSTANT;
    node->lower = INT64_MAX;
    node->upper = INT64_MIN;
}

static int wf_carrier_count(const wf_node *nodes, uint32_t index,
                            int *counts, double *frequencies) {
    const wf_node *node;
    int count;
    if (counts[index] >= 0) return counts[index];
    node = nodes + index;
    switch (node->op) {
        case WF_OP_COS:
        case WF_OP_SIN:
            counts[index] = 1;
            frequencies[index] = fabs(node->p0);
            return 1;
        case WF_OP_ADD:
        case WF_OP_MUL: {
            int left = wf_carrier_count(nodes, node->left,
                                        counts, frequencies);
            int right = wf_carrier_count(nodes, node->right,
                                         counts, frequencies);
            count = left + right;
            if (count > 2) count = 2;
            counts[index] = count;
            if (count == 1)
                frequencies[index] = left == 1
                    ? frequencies[node->left] : frequencies[node->right];
            return count;
        }
        case WF_OP_SCALE:
        case WF_OP_POWER:
        case WF_OP_WINDOW:
            count = wf_carrier_count(nodes, node->left,
                                     counts, frequencies);
            counts[index] = count;
            frequencies[index] = frequencies[node->left];
            return count;
        default:
            counts[index] = 0;
            frequencies[index] = 0.0;
            return 0;
    }
}

static void wf_filter_context(wf_node *nodes, uint32_t index,
                              int *counts, double *frequencies,
                              double low, double high) {
    wf_node *node = nodes + index;
    if (node->op == WF_OP_ADD) {
        wf_filter_context(nodes, node->left, counts, frequencies, low, high);
        wf_filter_context(nodes, node->right, counts, frequencies, low, high);
        return;
    }
    if (node->op == WF_OP_SCALE || node->op == WF_OP_WINDOW) {
        wf_filter_context(nodes, node->left, counts, frequencies, low, high);
        return;
    }
    {
        int count = wf_carrier_count(nodes, index, counts, frequencies);
        if ((count == 0 && low > 0.0)
                || (count == 1 && !(low <= frequencies[index]
                                    && frequencies[index] < high)))
            wf_make_zero_node(node);
    }
}

cwaveform_wave *cwaveform_wave_filter(const cwaveform_wave *wave,
                                      double low, double high,
                                      double epsilon) {
    wf_node *nodes;
    int *counts;
    double *frequencies;
    cwaveform_wave *result;
    uint32_t index;
    (void)epsilon;
    if (wave == NULL || !isfinite(low) || isnan(high) || low < 0.0
            || high < low) return NULL;
    nodes = (wf_node *)malloc((size_t)wave->node_count * sizeof(*nodes));
    counts = (int *)malloc((size_t)wave->node_count * sizeof(*counts));
    frequencies = (double *)calloc(wave->node_count, sizeof(*frequencies));
    if (nodes == NULL || counts == NULL || frequencies == NULL) {
        free(nodes); free(counts); free(frequencies);
        return NULL;
    }
    memcpy(nodes, wave->nodes, (size_t)wave->node_count * sizeof(*nodes));
    for (index = 0; index < wave->node_count; ++index) counts[index] = -1;
    wf_filter_context(nodes, wave->root, counts, frequencies, low, high);
    for (index = 0; index < wave->node_count; ++index) {
        if (wf_prepare_decoded_node(nodes + index, nodes, index,
                                    wave->parameter_count) != 0) {
            free(nodes); free(counts); free(frequencies);
            return NULL;
        }
    }
    result = wf_wave_from_parts(nodes, wave->node_count, wave->root,
                                wave->parameters, wave->parameter_count);
    free(nodes); free(counts); free(frequencies);
    if (result != NULL) {
        cwaveform_wave *compact = cwaveform_wave_simplify(result, epsilon);
        cwaveform_wave_release(result);
        result = compact;
    }
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

static double wf_drag_sin_value(const cwaveform_wave *wave,
                                const wf_node *node, double local,
                                int sinx) {
    size_t total = (size_t)node->parameter_count + 1;
    size_t block_count = total - 7;
    size_t order = block_count + 1;
    size_t power = ((block_count + 2) >> 1) << 1;
    size_t basis_count;
    double t0 = wf_node_parameter(wave, node, 0);
    double frequency = wf_node_parameter(wave, node, 1);
    double width = wf_node_parameter(wave, node, 2);
    double delta = wf_node_parameter(wave, node, 3);
    double phase = wf_node_parameter(wave, node, 4);
    double plateau = wf_node_parameter(wave, node, 5);
    double tab = wf_node_parameter(wave, node, 6);
    double angular;
    double *transform;
    double *derivatives;
    double *basis;
    double *values;
    double components[2] = {0.0, 0.0};
    double normalization = 1.0;
    size_t index;
    if (!(width > 0.0) || block_count > 64) return NAN;
    if (power < 2) power = 2;
    basis_count = power + 1;
    angular = 3.14159265358979323846 / width;
    transform = (double *)calloc(order * 4, sizeof(double));
    derivatives = (double *)calloc(order * basis_count, sizeof(double));
    basis = (double *)calloc(basis_count, sizeof(double));
    values = (double *)calloc(order, sizeof(double));
    if (transform == NULL || derivatives == NULL || basis == NULL
            || values == NULL) {
        free(transform); free(derivatives); free(basis); free(values);
        return NAN;
    }
    transform[0] = 1.0;
    transform[3] = 1.0;
    for (index = 0; index < block_count; ++index) {
        double block_frequency = wf_node_parameter(wave, node, 7 + index);
        double coefficient = 1.0 / (2.0 * 3.14159265358979323846
                                    * (block_frequency - delta));
        size_t row = index + 1;
        while (row-- != 0) {
            double *target = transform + (row + 1) * 4;
            const double *source = transform + row * 4;
            target[0] += -coefficient * source[1];
            target[1] += coefficient * source[0];
            target[2] += -coefficient * source[3];
            target[3] += coefficient * source[2];
        }
    }
    derivatives[power] = 1.0;
    for (index = 1; index < order; ++index) {
        size_t exponent;
        if (index & 1) {
            for (exponent = 0; exponent < power; ++exponent)
                derivatives[index * basis_count + exponent] =
                    derivatives[(index - 1) * basis_count + exponent + 1]
                    * (double)(exponent + 1) * angular;
        } else {
            for (exponent = 0; exponent + 2 <= power; ++exponent)
                derivatives[index * basis_count + exponent] =
                    derivatives[(index - 2) * basis_count + exponent + 2]
                    * (double)(exponent + 1) * (double)(exponent + 2);
            for (exponent = 0; exponent <= power; ++exponent)
                derivatives[index * basis_count + exponent] -=
                    derivatives[(index - 2) * basis_count + exponent]
                    * (double)(exponent * exponent);
            for (exponent = 0; exponent <= power; ++exponent)
                derivatives[index * basis_count + exponent]
                    *= angular * angular;
        }
    }
    {
        double midpoint = t0 + width / 2.0;
        double plateau_stop = midpoint + plateau;
        double adjusted = local >= plateau_stop ? local - plateau : local;
        double angle = angular * (adjusted - t0);
        double sine = sin(angle);
        double cosine = cos(angle);
        int in_plateau = local > midpoint && local < plateau_stop;
        size_t exponent;
        for (exponent = 0; exponent <= power; ++exponent) {
            basis[exponent] = pow(sine, (double)exponent);
            if (exponent & 1) basis[exponent] *= cosine;
            if (in_plateau) basis[exponent] = 0.0;
        }
        for (index = 0; index < order; ++index) {
            for (exponent = 0; exponent <= power; ++exponent)
                values[index] += derivatives[index * basis_count + exponent]
                    * basis[exponent];
        }
        if (in_plateau) values[0] = 1.0;
        if (sinx) {
            int left = local >= midpoint - tab * width / 2.0
                && local <= midpoint;
            int right = local >= plateau_stop
                && local <= plateau_stop + tab * width / 2.0;
            if (left || right) {
                double boundary_angle = angular
                    * ((left ? (1.0 - tab) : (1.0 + tab)) * width / 2.0);
                double boundary_sine = sin(boundary_angle);
                double boundary_cosine = cos(boundary_angle);
                double *boundary_basis = basis;
                double *edge_values = (double *)calloc(order, sizeof(double));
                double *polynomial = (double *)calloc(2 * order, sizeof(double));
                double edge_x = (left ? -tab : tab) * width / 2.0;
                double x = left ? local - midpoint : local - plateau_stop;
                if (edge_values == NULL || polynomial == NULL) {
                    free(edge_values); free(polynomial);
                    free(transform); free(derivatives); free(basis); free(values);
                    return NAN;
                }
                for (exponent = 0; exponent <= power; ++exponent) {
                    boundary_basis[exponent] = pow(boundary_sine,
                                                    (double)exponent);
                    if (exponent & 1) boundary_basis[exponent]
                        *= boundary_cosine;
                }
                for (index = 0; index < order; ++index)
                    for (exponent = 0; exponent <= power; ++exponent)
                        edge_values[index] +=
                            derivatives[index * basis_count + exponent]
                            * boundary_basis[exponent];
                if (wf_edge_polynomial(edge_values, order, edge_x,
                                       polynomial) != 0) {
                    free(edge_values); free(polynomial);
                    free(transform); free(derivatives); free(basis); free(values);
                    return NAN;
                }
                for (index = 0; index < order; ++index)
                    values[index] = wf_polynomial_derivative_value(
                        polynomial, 2 * order, index, x);
                free(edge_values);
                free(polynomial);
            }
        }
    }
    if (!sinx) {
        double peak_components[2] = {0.0, 0.0};
        size_t exponent;
        memset(basis, 0, basis_count * sizeof(double));
        for (exponent = 0; exponent <= power; exponent += 2)
            basis[exponent] = 1.0;
        memset(values, 0, order * sizeof(double));
        for (index = 0; index < order; ++index)
            for (exponent = 0; exponent <= power; ++exponent)
                values[index] += derivatives[index * basis_count + exponent]
                    * basis[exponent];
        for (index = 0; index < order; ++index) {
            peak_components[0] += transform[index * 4 + 0] * values[index];
            peak_components[1] += transform[index * 4 + 2] * values[index];
        }
        normalization = hypot(peak_components[0], peak_components[1]);
        /* Restore the actual derivatives after the peak calculation. */
        {
            double midpoint = t0 + width / 2.0;
            double plateau_stop = midpoint + plateau;
            double adjusted = local >= plateau_stop ? local - plateau : local;
            double angle = angular * (adjusted - t0);
            double sine = sin(angle), cosine = cos(angle);
            int in_plateau = local > midpoint && local < plateau_stop;
            memset(values, 0, order * sizeof(double));
            for (index = 0; index <= power; ++index) {
                basis[index] = pow(sine, (double)index);
                if (index & 1) basis[index] *= cosine;
                if (in_plateau) basis[index] = 0.0;
            }
            for (index = 0; index < order; ++index) {
                size_t exponent;
                for (exponent = 0; exponent <= power; ++exponent)
                    values[index] += derivatives[index * basis_count + exponent]
                        * basis[exponent];
            }
            if (in_plateau) values[0] = 1.0;
        }
    }
    for (index = 0; index < order; ++index) {
        components[0] += transform[index * 4 + 0] * values[index];
        components[1] += transform[index * 4 + 2] * values[index];
    }
    components[0] /= normalization;
    components[1] /= normalization;
    {
        double carrier = 2.0 * 3.14159265358979323846
            * (frequency + delta) * local
            - (2.0 * 3.14159265358979323846 * delta * t0 + phase);
        double result = components[0] * cos(carrier)
            + components[1] * sin(carrier);
        free(transform); free(derivatives); free(basis); free(values);
        return result;
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
                values[index] = wf_drag_sin_value(wave, node, local, 0);
                break;
            case WF_OP_DRAG_SINX:
                local = position - (double)node->shift / ticks;
                values[index] = wf_drag_sin_value(wave, node, local, 1);
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

/* Evaluate one node at a time so vForce can process transcendental functions
 * in wide batches.  The portable path below deliberately stays scalar. */
static int wf_evaluate_many_apple(
    const cwaveform_wave *wave, const double *positions, size_t count,
    double delay, double scale, double lower_clip, double upper_clip,
    double *output) {
    double *matrix;
    double *scratch;
    uint32_t node_index;
    size_t index;
    if (count != 0 && (size_t)wave->node_count > SIZE_MAX / count) return -2;
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
                for (index = 0; index < count; ++index) {
                    double position = positions[index] - delay;
                    row[index] = position >= lower && position < upper
                        ? node->p0 : 0.0;
                }
                break;
            case WF_OP_GAUSSIAN:
                for (index = 0; index < count; ++index) {
                    double position = positions[index] - delay;
                    double local = (position - shift) / node->p1;
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
                    int batch = count > (size_t)INT_MAX ? INT_MAX : (int)count;
                    size_t cursor = 0;
                    while (cursor < count) {
                        batch = count - cursor > (size_t)INT_MAX
                            ? INT_MAX : (int)(count - cursor);
                        vvsin(row + cursor, scratch + cursor, &batch);
                        cursor += (size_t)batch;
                    }
                }
                if (lower != -DBL_MAX || upper != DBL_MAX) {
                    for (index = 0; index < count; ++index) {
                        double position = positions[index] - delay;
                        if (position < lower || position >= upper) row[index] = 0.0;
                    }
                }
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
            default:
                free(matrix);
                free(scratch);
                return -3;
        }
    }
    {
        const double *root = matrix + (size_t)wave->root * count;
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
#if defined(__APPLE__)
    if (count >= 256) {
        int status = wf_evaluate_many_apple(
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
    scaled = nearbyint(value * (32768.0 / full_scale));
    if (scaled <= -32768.0) return INT16_MIN;
    if (scaled >= 32767.0) return INT16_MAX;
    return (int16_t)scaled;
}

static int32_t wf_quantize32(double value, double full_scale) {
    double scaled;
    if (value <= -full_scale) return INT32_MIN;
    if (value >= full_scale) return INT32_MAX;
    scaled = nearbyint(value * (2147483648.0 / full_scale));
    if (scaled <= -2147483648.0) return INT32_MIN;
    if (scaled >= 2147483647.0) return INT32_MAX;
    return (int32_t)scaled;
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
        } else {
            free(values);
            return -1;
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
    stack->hash = wf_hash_bytes(stack->data, total);
    return 0;
}

cwaveform_stack *cwaveform_stack_create(
    cwaveform_wave *const *templates, const uint32_t *template_ids,
    const int64_t *delay_ticks, const double *scales,
    size_t template_count, size_t event_count) {
    cwaveform_stack *stack;
    size_t index;
    if ((template_count != 0 && templates == NULL)
            || (event_count != 0 && (template_ids == NULL
                || delay_ticks == NULL || scales == NULL))) {
        return NULL;
    }
    stack = (cwaveform_stack *)calloc(1, sizeof(*stack));
    if (stack == NULL) return NULL;
    stack->references = 1;
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
    cwaveform_wave **templates = NULL;
    uint32_t *ids = NULL;
    int64_t *delays = NULL;
    double *scales = NULL;
    cwaveform_stack *stack = NULL;
    uint32_t index;
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
    templates = (cwaveform_wave **)calloc(template_count, sizeof(*templates));
    ids = (uint32_t *)malloc((size_t)event_count * sizeof(*ids));
    delays = (int64_t *)malloc((size_t)event_count * sizeof(*delays));
    scales = (double *)malloc((size_t)event_count * sizeof(*scales));
    if ((template_count && templates == NULL)
            || (event_count && (ids == NULL || delays == NULL || scales == NULL))) {
        goto done;
    }
    for (index = 0; index < template_count; ++index) {
        size_t template_size = wf_get_u32(
            data + WF_STACK_HEADER_SIZE + 4 * index);
        if (template_size > event_offset - cursor) goto done;
        templates[index] = cwaveform_wave_from_bytes(data + cursor,
                                                      template_size);
        if (templates[index] == NULL) goto done;
        cursor += template_size;
    }
    if (cursor != event_offset) goto done;
    for (index = 0; index < event_count; ++index) {
        ids[index] = wf_get_u32(data + event_offset + 4 * index);
        delays[index] = wf_get_i64(data + event_offset + 4 * event_count
                                   + 8 * index);
        scales[index] = wf_get_f64(data + event_offset + 12 * event_count
                                   + 8 * index);
    }
    stack = cwaveform_stack_create(templates, ids, delays, scales,
                                   template_count, event_count);
done:
    if (templates != NULL) {
        for (index = 0; index < template_count; ++index) {
            cwaveform_wave_release(templates[index]);
        }
    }
    free(templates);
    free(ids);
    free(delays);
    free(scales);
    return stack;
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
    return stack == NULL ? 0 : stack->hash;
}

size_t cwaveform_stack_event_count(const cwaveform_stack *stack) {
    return stack == NULL ? 0 : stack->event_count;
}

size_t cwaveform_stack_template_count(const cwaveform_stack *stack) {
    return stack == NULL ? 0 : stack->template_count;
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
    long double start = (long double)destination;
    long double stop = start + (long double)template_count;
    long double clipped_start = start < 0.0L ? 0.0L : start;
    long double clipped_stop = stop > (long double)output_count
        ? (long double)output_count : stop;
    if (clipped_start >= clipped_stop || clipped_stop <= 0.0L
            || clipped_start >= (long double)output_count) {
        *copy_count = 0;
        return 0;
    }
    *source_start = (size_t)(clipped_start - start);
    *output_start = (size_t)clipped_start;
    *copy_count = (size_t)(clipped_stop - clipped_start);
    return 1;
}

static wf_plan_group *wf_plan_get_group(
    cwaveform_sample_plan *plan, const cwaveform_stack *stack,
    uint32_t template_id, uint64_t phase, int64_t step_numerator) {
    size_t index;
    wf_plan_group *group;
    cwaveform_wave *wave;
    int64_t lower;
    int64_t upper;
    uint64_t lower_phase;
    uint64_t delta;
    long double sample_count;
    double *values;
    for (index = 0; index < plan->group_count; ++index) {
        group = plan->groups + index;
        if (group->template_id == template_id && group->phase == phase)
            return group;
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
        for (index = 0; index < group->sample_count; ++index) {
            long double tick = (long double)group->first_tick
                + (long double)index * (long double)step_numerator;
            group->samples[index] = wf_evaluate_one(
                wave,
                (double)(tick / (long double)wf_ticks_per_second),
                values);
        }
        free(values);
    }
    ++plan->group_count;
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
            cwaveform_sample_plan_release(plan);
            return NULL;
        }
        delay = wf_add_tick(stack->delays[event_index], global_shift);
        start_phase = wf_tick_mod(start_tick, (uint64_t)step_numerator);
        delay_phase = wf_tick_mod(delay, (uint64_t)step_numerator);
        phase = start_phase >= delay_phase
            ? start_phase - delay_phase
            : (uint64_t)step_numerator - (delay_phase - start_phase);
        group = wf_plan_get_group(plan, stack, template_id, phase,
                                  step_numerator);
        if (group == NULL) {
            cwaveform_sample_plan_release(plan);
            return NULL;
        }
        destination_value = (
            (long double)delay + (long double)group->first_tick
            - (long double)start_tick) / (long double)step_numerator;
        if (destination_value < (long double)INT64_MIN
                || destination_value > (long double)INT64_MAX) {
            cwaveform_sample_plan_release(plan);
            return NULL;
        }
        destination = (int64_t)llroundl(destination_value);
        status = wf_plan_group_add_placement(
            group, destination, stack->scales[event_index]);
        if (status != 0) {
            cwaveform_sample_plan_release(plan);
            return NULL;
        }
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
    return plan;
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

static void wf_plan_sample_float(const cwaveform_sample_plan *plan,
                                 double offset, double *output) {
    size_t group_index;
    size_t index;
    for (index = 0; index < plan->count; ++index) output[index] = offset;
    for (group_index = 0; group_index < plan->group_count; ++group_index) {
        const wf_plan_group *group = plan->groups + group_index;
        if (group->grouped_scales && plan->non_overlapping) {
            size_t scale_index;
            double *transformed = (offset == 0.0 || group->sample_count == 0)
                ? NULL
                : (double *)malloc(group->sample_count * sizeof(double));
            if (offset != 0.0 && group->sample_count != 0
                    && transformed == NULL) goto generic;
            for (scale_index = 0;
                    scale_index < group->scale_group_count; ++scale_index) {
                const wf_plan_scale_group *scale_group
                    = group->scale_groups + scale_index;
                const double *source_values = scale_group->samples;
                size_t destination_index;
                if (offset != 0.0) {
                    for (index = 0; index < group->sample_count; ++index)
                        transformed[index] = offset
                            + scale_group->samples[index];
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
                for (sample_index = 0; sample_index < copy_count; ++sample_index)
                    output[output_start + sample_index] = offset
                        + group->scales[index]
                        * group->samples[source_start + sample_index];
            } else {
                for (sample_index = 0; sample_index < copy_count; ++sample_index)
                    output[output_start + sample_index] += group->scales[index]
                        * group->samples[source_start + sample_index];
            }
        }
    }
}

static int wf_plan_sample_integer_grouped(
    const cwaveform_sample_plan *plan, const wf_plan_group *group,
    double offset, int dtype, double full_scale, void *output) {
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
        for (index = 0; index < group->sample_count; ++index) {
            double value = offset + scale_group->samples[index];
            if (dtype == CWAVEFORM_INT16)
                ((int16_t *)quantized)[index] = wf_quantize16(value, full_scale);
            else
                ((int32_t *)quantized)[index] = wf_quantize32(value, full_scale);
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

int cwaveform_sample_plan_sample(
    const cwaveform_sample_plan *plan, double offset, int dtype,
    double full_scale, void *output) {
    size_t index;
    if (plan == NULL || output == NULL || !isfinite(offset)
            || !isfinite(full_scale) || full_scale <= 0.0) return -1;
    if (dtype == CWAVEFORM_FLOAT64) {
        wf_plan_sample_float(plan, offset, (double *)output);
        return 0;
    }
    if (dtype != CWAVEFORM_INT16 && dtype != CWAVEFORM_INT32) return -1;
    if (!plan->non_overlapping) {
        double *values = plan->count == 0 ? NULL
            : (double *)malloc(plan->count * sizeof(double));
        if (plan->count != 0 && values == NULL) return -2;
        wf_plan_sample_float(plan, offset, values);
        for (index = 0; index < plan->count; ++index) {
            if (dtype == CWAVEFORM_INT16)
                ((int16_t *)output)[index] = wf_quantize16(values[index], full_scale);
            else
                ((int32_t *)output)[index] = wf_quantize32(values[index], full_scale);
        }
        free(values);
        return 0;
    }
    if (dtype == CWAVEFORM_INT16) {
        int16_t base = wf_quantize16(offset, full_scale);
        for (index = 0; index < plan->count; ++index)
            ((int16_t *)output)[index] = base;
    } else {
        int32_t base = wf_quantize32(offset, full_scale);
        for (index = 0; index < plan->count; ++index)
            ((int32_t *)output)[index] = base;
    }
    for (index = 0; index < plan->group_count; ++index) {
        const wf_plan_group *group = plan->groups + index;
        if (group->grouped_scales) {
            int status = wf_plan_sample_integer_grouped(
                plan, group, offset, dtype, full_scale, output);
            if (status != 0) return status;
        } else {
            size_t placement_index;
            for (placement_index = 0;
                    placement_index < group->placement_count;
                    ++placement_index) {
                size_t source_start;
                size_t output_start;
                size_t copy_count;
                size_t sample_index;
                if (!wf_plan_placement_bounds(
                        group->destinations[placement_index],
                        group->sample_count, plan->count,
                        &source_start, &output_start, &copy_count)) continue;
                for (sample_index = 0; sample_index < copy_count; ++sample_index) {
                    double value = offset + group->scales[placement_index]
                        * group->samples[source_start + sample_index];
                    if (dtype == CWAVEFORM_INT16)
                        ((int16_t *)output)[output_start + sample_index]
                            = wf_quantize16(value, full_scale);
                    else
                        ((int32_t *)output)[output_start + sample_index]
                            = wf_quantize32(value, full_scale);
                }
            }
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
        if (dtype == CWAVEFORM_INT16) {
            for (index = 0; index < count; ++index)
                ((int16_t *)output)[index] = wf_quantize16(float_output[index], full_scale);
        } else if (dtype == CWAVEFORM_INT32) {
            for (index = 0; index < count; ++index)
                ((int32_t *)output)[index] = wf_quantize32(float_output[index], full_scale);
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
    return "WNF4/WNS4 little-endian immutable waveform blocks; global integer ticks; ABI 2";
}
