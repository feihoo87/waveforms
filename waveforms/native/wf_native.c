#define WF_NATIVE_BUILD 1
#include "wf_native.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#define WF_TICKS_PER_SECOND UINT64_C(120000000000)
#define WF_WAVE_HEADER_SIZE 24u
#define WF_NODE_SIZE 28u
#define WF_STACK_HEADER_SIZE 16u
#define WF_VERSION 1u

enum wf_op {
    WF_OP_CONSTANT = 1,
    WF_OP_GAUSSIAN = 2,
    WF_OP_COS = 3,
    WF_OP_SIN = 4,
    WF_OP_SQUARE = 5,
    WF_OP_ADD = 16,
    WF_OP_MUL = 17,
    WF_OP_SCALE = 18
};

typedef struct wf_node {
    uint8_t op;
    uint32_t left;
    uint32_t right;
    int64_t shift;
    double p0;
    double p1;
    double p2;
    int64_t lower;
    int64_t upper;
} wf_node;

struct wf_native_wave {
    uint32_t references;
    uint32_t node_count;
    uint32_t root;
    uint64_t hash;
    size_t data_size;
    uint8_t *data;
    wf_node *nodes;
};

struct wf_native_stack {
    uint32_t references;
    size_t template_count;
    size_t event_count;
    wf_native_wave **templates;
    uint32_t *template_ids;
    int64_t *delays;
    double *scales;
    uint64_t hash;
    size_t data_size;
    uint8_t *data;
};

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
    long double scaled = (long double)value * (long double)WF_TICKS_PER_SECOND;
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
    wf_put_u32(data + 4, node->left);
    wf_put_u32(data + 8, node->right);
    wf_put_i64(data + 12, node->shift);
    wf_put_f64(data + 20, node->p0);
}

static void wf_decode_node(wf_node *node, const uint8_t *data) {
    memset(node, 0, sizeof(*node));
    node->op = data[0];
    node->left = wf_get_u32(data + 4);
    node->right = wf_get_u32(data + 8);
    node->shift = wf_get_i64(data + 12);
    node->p0 = wf_get_f64(data + 20);
}

static int wf_prepare_decoded_node(wf_node *node, const wf_node *nodes,
                                   uint32_t index) {
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
                / (double)WF_TICKS_PER_SECOND;
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
        default:
            return -1;
    }
    return 0;
}

static wf_native_wave *wf_wave_from_nodes(const wf_node *nodes,
                                           uint32_t node_count,
                                           uint32_t root) {
    wf_native_wave *wave;
    size_t size;
    uint32_t index;
    if (nodes == NULL || node_count == 0 || root >= node_count) {
        return NULL;
    }
    if ((size_t)node_count > (SIZE_MAX - WF_WAVE_HEADER_SIZE) / WF_NODE_SIZE) {
        return NULL;
    }
    size = WF_WAVE_HEADER_SIZE + (size_t)node_count * WF_NODE_SIZE;
    wave = (wf_native_wave *)calloc(1, sizeof(*wave));
    if (wave == NULL) {
        return NULL;
    }
    wave->data = (uint8_t *)calloc(1, size);
    wave->nodes = (wf_node *)malloc((size_t)node_count * sizeof(*wave->nodes));
    if (wave->data == NULL || wave->nodes == NULL) {
        wf_native_wave_release(wave);
        return NULL;
    }
    memcpy(wave->nodes, nodes, (size_t)node_count * sizeof(*nodes));
    wave->references = 1;
    wave->node_count = node_count;
    wave->root = root;
    wave->data_size = size;
    memcpy(wave->data, "WNF4", 4);
    wf_put_u16(wave->data + 4, WF_VERSION);
    wf_put_u16(wave->data + 6, 0);
    wf_put_u32(wave->data + 8, node_count);
    wf_put_u32(wave->data + 12, root);
    wf_put_u64(wave->data + 16, WF_TICKS_PER_SECOND);
    for (index = 0; index < node_count; ++index) {
        wf_encode_node(wave->data + WF_WAVE_HEADER_SIZE
                       + (size_t)index * WF_NODE_SIZE, nodes + index);
    }
    wave->hash = wf_hash_bytes(wave->data, size);
    return wave;
}

static wf_native_wave *wf_wave_single(uint8_t op, int64_t lower, int64_t upper,
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

uint64_t wf_native_ticks_per_second(void) {
    return WF_TICKS_PER_SECOND;
}

wf_native_wave *wf_native_wave_constant(double value) {
    int64_t lower = value == 0.0 ? INT64_MAX : INT64_MIN;
    int64_t upper = value == 0.0 ? INT64_MIN : INT64_MAX;
    if (!isfinite(value)) {
        return NULL;
    }
    return wf_wave_single(WF_OP_CONSTANT, lower, upper, 0, value, 0.0);
}

wf_native_wave *wf_native_wave_gaussian(double width_seconds) {
    int64_t lower;
    int64_t upper;
    if (!isfinite(width_seconds) || width_seconds <= 0.0) {
        return wf_native_wave_constant(0.0);
    }
    lower = wf_seconds_to_tick(-0.75 * width_seconds);
    upper = wf_seconds_to_tick(0.75 * width_seconds);
    return wf_wave_single(WF_OP_GAUSSIAN, lower, upper, 0,
                          width_seconds,
                          (double)wf_seconds_to_tick(
                              width_seconds / 3.3302184446307908)
                          / (double)WF_TICKS_PER_SECOND);
}

wf_native_wave *wf_native_wave_cos(double angular_frequency, double phase) {
    if (!isfinite(angular_frequency) || !isfinite(phase)) {
        return NULL;
    }
    if (angular_frequency == 0.0) {
        return wf_native_wave_constant(cos(phase));
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

wf_native_wave *wf_native_wave_sin(double angular_frequency, double phase) {
    if (!isfinite(angular_frequency) || !isfinite(phase)) {
        return NULL;
    }
    if (angular_frequency == 0.0) {
        return wf_native_wave_constant(sin(phase));
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

wf_native_wave *wf_native_wave_square(double width_seconds) {
    int64_t lower;
    int64_t upper;
    if (!isfinite(width_seconds) || width_seconds <= 0.0) {
        return wf_native_wave_constant(0.0);
    }
    lower = wf_seconds_to_tick(-0.5 * width_seconds);
    upper = wf_seconds_to_tick(0.5 * width_seconds);
    return wf_wave_single(WF_OP_SQUARE, lower, upper, 0,
                          width_seconds, 0.0);
}

wf_native_wave *wf_native_wave_from_bytes(const uint8_t *data, size_t size) {
    wf_native_wave *wave;
    uint32_t node_count;
    uint32_t root;
    uint32_t index;
    if (data == NULL || size < WF_WAVE_HEADER_SIZE
            || memcmp(data, "WNF4", 4) != 0
            || wf_get_u16(data + 4) != WF_VERSION
            || wf_get_u16(data + 6) != 0
            || wf_get_u64(data + 16) != WF_TICKS_PER_SECOND) {
        return NULL;
    }
    node_count = wf_get_u32(data + 8);
    root = wf_get_u32(data + 12);
    if (node_count == 0 || root >= node_count
            || size != WF_WAVE_HEADER_SIZE + (size_t)node_count * WF_NODE_SIZE) {
        return NULL;
    }
    wave = (wf_native_wave *)calloc(1, sizeof(*wave));
    if (wave == NULL) {
        return NULL;
    }
    wave->data = (uint8_t *)malloc(size);
    wave->nodes = (wf_node *)malloc((size_t)node_count * sizeof(*wave->nodes));
    if (wave->data == NULL || wave->nodes == NULL) {
        wf_native_wave_release(wave);
        return NULL;
    }
    memcpy(wave->data, data, size);
    for (index = 0; index < node_count; ++index) {
        wf_decode_node(wave->nodes + index,
                       data + WF_WAVE_HEADER_SIZE + (size_t)index * WF_NODE_SIZE);
        if (wf_prepare_decoded_node(wave->nodes + index, wave->nodes,
                                    index) != 0) {
            wf_native_wave_release(wave);
            return NULL;
        }
    }
    wave->references = 1;
    wave->node_count = node_count;
    wave->root = root;
    wave->data_size = size;
    wave->hash = wf_hash_bytes(data, size);
    return wave;
}

static uint32_t wf_clone_affine(const wf_native_wave *source, wf_node *target,
                                uint32_t offset, int64_t delay, double scale,
                                uint32_t *next) {
    uint32_t index;
    uint32_t root;
    for (index = 0; index < source->node_count; ++index) {
        wf_node node = source->nodes[index];
        if (node.op == WF_OP_ADD || node.op == WF_OP_MUL) {
            node.left += offset;
            node.right += offset;
        } else if (node.op == WF_OP_SCALE) {
            node.left += offset;
        } else if (node.op != WF_OP_CONSTANT) {
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

static wf_native_wave *wf_combine_affine(
    const wf_native_wave *left, int64_t left_delay, double left_scale,
    const wf_native_wave *right, int64_t right_delay, double right_scale,
    uint8_t operation) {
    uint32_t capacity;
    uint32_t next = 0;
    uint32_t left_root;
    uint32_t right_root;
    wf_node *nodes;
    wf_node node;
    wf_native_wave *result;
    if (left == NULL || right == NULL || !isfinite(left_scale)
            || !isfinite(right_scale)) {
        return NULL;
    }
    if (left->node_count > UINT32_MAX - right->node_count - 3) {
        return NULL;
    }
    capacity = left->node_count + right->node_count + 3;
    nodes = (wf_node *)calloc(capacity, sizeof(*nodes));
    if (nodes == NULL) {
        return NULL;
    }
    left_root = wf_clone_affine(left, nodes, 0, left_delay, left_scale, &next);
    right_root = wf_clone_affine(right, nodes, next, right_delay, right_scale,
                                 &next);
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
    result = wf_wave_from_nodes(nodes, next + 1, next);
    free(nodes);
    return result;
}

wf_native_wave *wf_native_wave_add_affine(
    const wf_native_wave *left, int64_t left_delay, double left_scale,
    const wf_native_wave *right, int64_t right_delay, double right_scale) {
    return wf_combine_affine(left, left_delay, left_scale, right, right_delay,
                             right_scale, WF_OP_ADD);
}

wf_native_wave *wf_native_wave_mul_affine(
    const wf_native_wave *left, int64_t left_delay, double left_scale,
    const wf_native_wave *right, int64_t right_delay, double right_scale) {
    return wf_combine_affine(left, left_delay, left_scale, right, right_delay,
                             right_scale, WF_OP_MUL);
}

wf_native_wave *wf_native_wave_materialize(const wf_native_wave *wave,
                                            int64_t delay, double scale) {
    wf_node *nodes;
    uint32_t next = 0;
    uint32_t root;
    wf_native_wave *result;
    if (wave == NULL || !isfinite(scale)) {
        return NULL;
    }
    nodes = (wf_node *)calloc((size_t)wave->node_count + 1, sizeof(*nodes));
    if (nodes == NULL) {
        return NULL;
    }
    root = wf_clone_affine(wave, nodes, 0, delay, scale, &next);
    result = wf_wave_from_nodes(nodes, next, root);
    free(nodes);
    return result;
}

void wf_native_wave_retain(wf_native_wave *wave) {
    if (wave != NULL) {
        ++wave->references;
    }
}

void wf_native_wave_release(wf_native_wave *wave) {
    if (wave == NULL) {
        return;
    }
    if (wave->references > 1) {
        --wave->references;
        return;
    }
    free(wave->data);
    free(wave->nodes);
    free(wave);
}

const uint8_t *wf_native_wave_bytes(const wf_native_wave *wave, size_t *size) {
    if (wave == NULL) {
        return NULL;
    }
    if (size != NULL) {
        *size = wave->data_size;
    }
    return wave->data;
}

uint64_t wf_native_wave_hash(const wf_native_wave *wave) {
    return wave == NULL ? 0 : wave->hash;
}

int wf_native_wave_equal(const wf_native_wave *left,
                         const wf_native_wave *right) {
    return left != NULL && right != NULL && left->hash == right->hash
        && left->data_size == right->data_size
        && memcmp(left->data, right->data, left->data_size) == 0;
}

int64_t wf_native_wave_lower_tick(const wf_native_wave *wave) {
    return wave == NULL ? INT64_MAX : wave->nodes[wave->root].lower;
}

int64_t wf_native_wave_upper_tick(const wf_native_wave *wave) {
    return wave == NULL ? INT64_MIN : wave->nodes[wave->root].upper;
}

uint32_t wf_native_wave_node_count(const wf_native_wave *wave) {
    return wave == NULL ? 0 : wave->node_count;
}

static double wf_evaluate_one(const wf_native_wave *wave, double position,
                              double *values) {
    uint32_t index;
    const double ticks = (double)WF_TICKS_PER_SECOND;
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
                local /= node->p1;
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
            case WF_OP_ADD:
                values[index] = values[node->left] + values[node->right];
                break;
            case WF_OP_MUL:
                values[index] = values[node->left] * values[node->right];
                break;
            case WF_OP_SCALE:
                values[index] = node->p0 * values[node->left];
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
    const wf_native_wave *wave, const double *positions, size_t count,
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
            : (double)node->lower / (double)WF_TICKS_PER_SECOND;
        double upper = node->upper == INT64_MAX ? DBL_MAX
            : (double)node->upper / (double)WF_TICKS_PER_SECOND;
        double shift = (double)node->shift / (double)WF_TICKS_PER_SECOND;
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

int wf_native_wave_evaluate(
    const wf_native_wave *wave, const double *positions, size_t count,
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
            (double)delay_tick / (double)WF_TICKS_PER_SECOND,
            scale, lower_clip, upper_clip, output);
        if (status == 0) return 0;
    }
#endif
    values = (double *)malloc((size_t)wave->node_count * sizeof(*values));
    if (values == NULL) {
        return -2;
    }
    delay = (double)delay_tick / (double)WF_TICKS_PER_SECOND;
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
                                * (long double)WF_TICKS_PER_SECOND));
}

int wf_native_wave_sample(
    const wf_native_wave *wave, int64_t start_tick, size_t count,
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
        if (dtype == WF_NATIVE_FLOAT64) {
            ((double *)output)[index] = value;
        } else if (dtype == WF_NATIVE_INT16) {
            ((int16_t *)output)[index] = wf_quantize16(value, full_scale);
        } else if (dtype == WF_NATIVE_INT32) {
            ((int32_t *)output)[index] = wf_quantize32(value, full_scale);
        } else {
            free(values);
            return -1;
        }
    }
    free(values);
    return 0;
}

static int wf_stack_encode(wf_native_stack *stack) {
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

wf_native_stack *wf_native_stack_create(
    wf_native_wave *const *templates, const uint32_t *template_ids,
    const int64_t *delay_ticks, const double *scales,
    size_t template_count, size_t event_count) {
    wf_native_stack *stack;
    size_t index;
    if ((template_count != 0 && templates == NULL)
            || (event_count != 0 && (template_ids == NULL
                || delay_ticks == NULL || scales == NULL))) {
        return NULL;
    }
    stack = (wf_native_stack *)calloc(1, sizeof(*stack));
    if (stack == NULL) return NULL;
    stack->references = 1;
    stack->template_count = template_count;
    stack->event_count = event_count;
    stack->templates = (wf_native_wave **)calloc(template_count,
                                                  sizeof(*stack->templates));
    stack->template_ids = (uint32_t *)malloc(event_count * sizeof(uint32_t));
    stack->delays = (int64_t *)malloc(event_count * sizeof(int64_t));
    stack->scales = (double *)malloc(event_count * sizeof(double));
    if ((template_count && stack->templates == NULL)
            || (event_count && (stack->template_ids == NULL
                || stack->delays == NULL || stack->scales == NULL))) {
        wf_native_stack_release(stack);
        return NULL;
    }
    for (index = 0; index < template_count; ++index) {
        if (templates[index] == NULL) {
            wf_native_stack_release(stack);
            return NULL;
        }
        stack->templates[index] = templates[index];
        wf_native_wave_retain(stack->templates[index]);
    }
    for (index = 0; index < event_count; ++index) {
        if (template_ids[index] >= template_count || !isfinite(scales[index])) {
            wf_native_stack_release(stack);
            return NULL;
        }
        stack->template_ids[index] = template_ids[index];
        stack->delays[index] = delay_ticks[index];
        stack->scales[index] = scales[index];
    }
    if (wf_stack_encode(stack) != 0) {
        wf_native_stack_release(stack);
        return NULL;
    }
    return stack;
}

wf_native_stack *wf_native_stack_materialize(
    const wf_native_stack *stack, int64_t global_shift, double offset) {
    wf_native_wave **templates;
    uint32_t *ids;
    int64_t *delays;
    double *scales;
    wf_native_wave *constant = NULL;
    wf_native_stack *result;
    size_t template_count;
    size_t event_count;
    size_t index;
    int add_offset;
    if (stack == NULL || !isfinite(offset)) return NULL;
    if (global_shift == 0 && offset == 0.0) {
        wf_native_stack_retain((wf_native_stack *)stack);
        return (wf_native_stack *)stack;
    }
    add_offset = offset != 0.0;
    if (add_offset && (stack->template_count == SIZE_MAX
                       || stack->event_count == SIZE_MAX)) return NULL;
    template_count = stack->template_count + (size_t)add_offset;
    event_count = stack->event_count + (size_t)add_offset;
    templates = (wf_native_wave **)malloc(template_count * sizeof(*templates));
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
        constant = wf_native_wave_constant(offset);
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
    result = wf_native_stack_create(templates, ids, delays, scales,
                                    template_count, event_count);
    wf_native_wave_release(constant);
    free(templates);
    free(ids);
    free(delays);
    free(scales);
    return result;
}

wf_native_stack *wf_native_stack_from_bytes(const uint8_t *data, size_t size) {
    uint32_t template_count;
    uint32_t event_count;
    size_t event_offset;
    size_t cursor;
    wf_native_wave **templates = NULL;
    uint32_t *ids = NULL;
    int64_t *delays = NULL;
    double *scales = NULL;
    wf_native_stack *stack = NULL;
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
    templates = (wf_native_wave **)calloc(template_count, sizeof(*templates));
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
        templates[index] = wf_native_wave_from_bytes(data + cursor,
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
    stack = wf_native_stack_create(templates, ids, delays, scales,
                                   template_count, event_count);
done:
    if (templates != NULL) {
        for (index = 0; index < template_count; ++index) {
            wf_native_wave_release(templates[index]);
        }
    }
    free(templates);
    free(ids);
    free(delays);
    free(scales);
    return stack;
}

void wf_native_stack_retain(wf_native_stack *stack) {
    if (stack != NULL) ++stack->references;
}

void wf_native_stack_release(wf_native_stack *stack) {
    size_t index;
    if (stack == NULL) return;
    if (stack->references > 1) {
        --stack->references;
        return;
    }
    if (stack->templates != NULL) {
        for (index = 0; index < stack->template_count; ++index) {
            wf_native_wave_release(stack->templates[index]);
        }
    }
    free(stack->templates);
    free(stack->template_ids);
    free(stack->delays);
    free(stack->scales);
    free(stack->data);
    free(stack);
}

const uint8_t *wf_native_stack_bytes(const wf_native_stack *stack,
                                      size_t *size) {
    if (stack == NULL) return NULL;
    if (size != NULL) *size = stack->data_size;
    return stack->data;
}

uint64_t wf_native_stack_hash(const wf_native_stack *stack) {
    return stack == NULL ? 0 : stack->hash;
}

size_t wf_native_stack_event_count(const wf_native_stack *stack) {
    return stack == NULL ? 0 : stack->event_count;
}

size_t wf_native_stack_template_count(const wf_native_stack *stack) {
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

int wf_native_stack_evaluate(
    const wf_native_stack *stack, const double *positions, size_t count,
    int64_t global_shift, double offset, double *output) {
    size_t event_index;
    size_t index;
    double ticks = (double)WF_TICKS_PER_SECOND;
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
        wf_native_wave *wave = stack->templates[stack->template_ids[event_index]];
        int64_t delay_tick = wf_add_tick(stack->delays[event_index], global_shift);
        int64_t lower_tick = wf_add_tick(wave->nodes[wave->root].lower, delay_tick);
        int64_t upper_tick = wf_add_tick(wave->nodes[wave->root].upper, delay_tick);
        double delay = (double)delay_tick / ticks;
        double lower = lower_tick == INT64_MIN ? -DBL_MAX : (double)lower_tick / ticks;
        double upper = upper_tick == INT64_MAX ? DBL_MAX : (double)upper_tick / ticks;
        size_t first = wf_lower_bound(positions, count, lower);
        size_t stop = wf_lower_bound(positions, count, upper);
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

int wf_native_stack_sample(
    const wf_native_stack *stack, int64_t start_tick, size_t count,
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
        wf_native_wave *wave = stack->templates[stack->template_ids[event_index]];
        int64_t delay = wf_add_tick(stack->delays[event_index], global_shift);
        int64_t lower = wf_add_tick(wave->nodes[wave->root].lower, delay);
        int64_t upper = wf_add_tick(wave->nodes[wave->root].upper, delay);
        if (stack->scales[event_index] == 0.0 || lower >= upper) continue;
        if (lower < previous_lower || lower < previous_upper) non_overlapping = 0;
        previous_lower = lower;
        previous_upper = upper;
    }
    if (dtype == WF_NATIVE_FLOAT64 || !non_overlapping) {
        double *float_output = dtype == WF_NATIVE_FLOAT64
            ? (double *)output : (double *)malloc(count * sizeof(double));
        if (float_output == NULL) {
            free(values);
            return -2;
        }
        for (index = 0; index < count; ++index) float_output[index] = offset;
        for (event_index = 0; event_index < stack->event_count; ++event_index) {
            wf_native_wave *wave = stack->templates[stack->template_ids[event_index]];
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
        if (dtype == WF_NATIVE_INT16) {
            for (index = 0; index < count; ++index)
                ((int16_t *)output)[index] = wf_quantize16(float_output[index], full_scale);
        } else if (dtype == WF_NATIVE_INT32) {
            for (index = 0; index < count; ++index)
                ((int32_t *)output)[index] = wf_quantize32(float_output[index], full_scale);
        } else if (dtype != WF_NATIVE_FLOAT64) {
            if (float_output != output) free(float_output);
            free(values);
            return -1;
        }
        if ((void *)float_output != output) free(float_output);
    } else {
        if (dtype == WF_NATIVE_INT16) {
            int16_t base = wf_quantize16(offset, full_scale);
            for (index = 0; index < count; ++index) ((int16_t *)output)[index] = base;
        } else if (dtype == WF_NATIVE_INT32) {
            int32_t base = wf_quantize32(offset, full_scale);
            for (index = 0; index < count; ++index) ((int32_t *)output)[index] = base;
        } else {
            free(values);
            return -1;
        }
        for (event_index = 0; event_index < stack->event_count; ++event_index) {
            wf_native_wave *wave = stack->templates[stack->template_ids[event_index]];
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
                if (dtype == WF_NATIVE_INT16)
                    ((int16_t *)output)[index] = wf_quantize16(value, full_scale);
                else
                    ((int32_t *)output)[index] = wf_quantize32(value, full_scale);
            }
        }
    }
    free(values);
    return 0;
}

wf_native_wave *wf_native_stack_simplify(const wf_native_stack *stack,
                                         int64_t global_shift,
                                         double offset) {
    size_t event_index;
    size_t capacity = offset == 0.0 ? 1 : 3;
    uint32_t next = 0;
    uint32_t root = UINT32_MAX;
    wf_node *nodes;
    wf_native_wave *result;
    if (stack == NULL || !isfinite(offset)) return NULL;
    for (event_index = 0; event_index < stack->event_count; ++event_index) {
        wf_native_wave *wave = stack->templates[stack->template_ids[event_index]];
        if ((size_t)wave->node_count > SIZE_MAX - capacity - 2) return NULL;
        capacity += wave->node_count + 2;
    }
    if (capacity > UINT32_MAX) return NULL;
    nodes = (wf_node *)calloc(capacity, sizeof(*nodes));
    if (nodes == NULL) return NULL;
    for (event_index = 0; event_index < stack->event_count; ++event_index) {
        wf_native_wave *wave = stack->templates[stack->template_ids[event_index]];
        uint32_t event_root = wf_clone_affine(
            wave, nodes, next,
            wf_add_tick(stack->delays[event_index], global_shift),
            stack->scales[event_index], &next
        );
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
    result = wf_wave_from_nodes(nodes, next, root);
    free(nodes);
    return result;
}

const char *wf_native_format_description(void) {
    return "WNF4/WNS4 little-endian immutable waveform blocks; 120 GHz ticks; ABI 1";
}
