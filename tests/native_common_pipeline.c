/* Standalone allocator-failure / sanitizer checks for private core helpers.
 * macOS: clang -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
 *   tests/native_common_pipeline.c -framework Accelerate -o /tmp/wf-common-test
 * Linux: use -lm instead of -framework Accelerate.
 */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static long allocation_budget = -1;

static int fail_allocation(void) {
    if (allocation_budget < 0) return 0;
    if (allocation_budget == 0) return 1;
    --allocation_budget;
    return 0;
}

static void *checked_malloc(size_t size) {
    return fail_allocation() ? NULL : malloc(size);
}

static void *checked_calloc(size_t count, size_t size) {
    return fail_allocation() ? NULL : calloc(count, size);
}

static void *checked_realloc(void *pointer, size_t size) {
    return fail_allocation() ? NULL : realloc(pointer, size);
}

#define malloc checked_malloc
#define calloc checked_calloc
#define realloc checked_realloc
#include "../waveforms/_cwaveform.c"
#undef malloc
#undef calloc
#undef realloc

static void check_bounds(void) {
    size_t source, output, count;
    assert(!wf_plan_placement_bounds(INT64_MIN, 17, 100, &source, &output, &count));
    assert(!wf_plan_placement_bounds(INT64_MAX, 17, 100, &source, &output, &count));
    assert(!wf_plan_placement_bounds(-17, 17, 100, &source, &output, &count));
    assert(!wf_plan_placement_bounds(0, 17, 0, &source, &output, &count));
    assert(wf_plan_placement_bounds(-3, 17, 10, &source, &output, &count));
    assert(source == 3 && output == 0 && count == 10);
    assert(wf_plan_placement_bounds(9, 17, 10, &source, &output, &count));
    assert(source == 0 && output == 9 && count == 1);
    if (SIZE_MAX > UINT32_MAX) {
        assert(wf_plan_placement_bounds(INT64_MIN, SIZE_MAX, 3,
                                         &source, &output, &count));
        assert(source == (UINT64_C(1) << 63) && output == 0 && count == 3);
        assert(wf_plan_placement_bounds(INT64_MAX, SIZE_MAX, SIZE_MAX,
                                         &source, &output, &count));
        assert(source == 0 && output == (size_t)INT64_MAX
               && count == SIZE_MAX - (size_t)INT64_MAX);
    }
}

int main(void) {
    cwaveform_wave *templates[40];
    uint32_t ids[200];
    int64_t delays[200];
    double scales[200];
    cwaveform_stack *stack;
    const uint8_t *bytes;
    uint8_t *mutated;
    size_t size;
    size_t index;
    long budget;
    int succeeded = 0;
    check_bounds();
    for (index = 0; index < 40; ++index) {
        templates[index] = cwaveform_wave_square((100.0 + index) / 120e9);
        assert(templates[index] != NULL);
    }
    for (index = 0; index < 200; ++index) {
        ids[index] = (uint32_t)(index % 40);
        delays[index] = (int64_t)(500 * index + index % 50);
        scales[index] = .25;
    }
    stack = cwaveform_stack_create(templates, ids, delays, scales, 40, 200);
    assert(stack != NULL);
    assert(stack->hash == 0);  /* No eager work on the sample-only path. */
    bytes = cwaveform_stack_bytes(stack, &size);
    mutated = (uint8_t *)malloc(size);
    assert(mutated != NULL);
    for (index = 0; index < size; ++index) {
        cwaveform_stack *restored;
        memcpy(mutated, bytes, size);
        mutated[index] ^= 0xff;
        restored = cwaveform_stack_from_bytes(mutated, size);
        cwaveform_stack_release(restored);
    }
    for (budget = 0; budget < 1000; ++budget) {
        cwaveform_stack *restored;
        allocation_budget = budget;
        restored = cwaveform_stack_from_bytes(bytes, size);
        allocation_budget = -1;
        if (restored != NULL) {
            assert(restored->hash == 0);
            assert(cwaveform_stack_hash(restored) == cwaveform_stack_hash(stack));
            assert(memcmp(restored->data, bytes, size) == 0);
            succeeded = 1;
        }
        cwaveform_stack_release(restored);
        if (succeeded) break;
    }
    assert(succeeded);
    succeeded = 0;
    for (budget = 0; budget < 5000; ++budget) {
        cwaveform_sample_plan *plan;
        allocation_budget = budget;
        plan = cwaveform_sample_plan_create(stack, 0, 2000, 50, 1, 7);
        allocation_budget = -1;
        if (plan != NULL) {
            double samples[2000];
            assert(cwaveform_sample_plan_sample(plan, 0, 0, 1, samples) == 0);
            succeeded = 1;
        }
        cwaveform_sample_plan_release(plan);
        if (succeeded) break;
    }
    assert(succeeded);
    free(mutated);
    cwaveform_stack_release(stack);
    for (index = 0; index < 40; ++index) cwaveform_wave_release(templates[index]);
    puts("native common-pipeline checks passed");
    return 0;
}
