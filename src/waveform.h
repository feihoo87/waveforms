#ifndef WAVEFORM_H
#define WAVEFORM_H

#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288 /* pi */
#endif

typedef uint8_t bool;

#define True 1
#define False 0

typedef int64_t Time;
typedef int64_t Frequency;
typedef uint64_t Phase;

static const Time s = 1000000000000000;
static const Time ms = 1000000000000;
static const Time us = 1000000000;
static const Time ns = 1000000;
static const Time ps = 1000;
static const Time fs = 1;

static const Phase pi2_3 = 0x5555555555555555;
static const Phase pi = 0x8000000000000000;
static const Phase pi2 = 0x4000000000000000;
static const Phase pi3 = 0x2aaaaaaaaaaaaaab;
static const Phase pi4 = 0x2000000000000000;
static const Phase pi6 = 0x1555555555555556;
static const Phase pi8 = 0x1000000000000000;

static const Frequency mHz = 1;
static const Frequency Hz = 1000;
static const Frequency kHz = 1000000;
static const Frequency MHz = 1000000000;
static const Frequency GHz = 1000000000000;
static const Frequency THz = 1000000000000000;

typedef union
{
    double real;
    int64_t integer;
    Phase phase;
    Time time;
    Frequency freq;
} Number;

typedef struct
{
    uint32_t func; // function index
    uint32_t argc; // number of arguments
    Number *argv;  // arguments
    Time shift;    // time shift
    int32_t n;     // numerator
    int32_t d;     // denominator
} Wave1;

typedef struct
{
    size_t size;
    Wave1 *waves;
    double amplitude;
} Wave2;

typedef struct
{
    size_t size;
    Wave2 *waves;
    Time *upper_bound;
} Wave3;

typedef struct
{
    size_t size;
    uint64_t sample_rate;
    double max;
    double min;
    Time start;
    Time stop;
    Wave3 *waves;
} Waveform;

typedef double (*FuncPtr)(double, size_t, Number *);

static inline int32_t gcd(int32_t a, int32_t b)
{
    if (a < 0)
    {
        a = -a;
    }
    if (b < 0)
    {
        b = -b;
    }
    while (b != 0)
    {
        int32_t c = a % b;
        a = b;
        b = c;
    }
    return a;
}

static inline size_t bisect_left(const void *array, const void *const value,
                                 size_t lo, size_t hi,
                                 size_t type_size, int (*cmp)(const void *, const void *))
{
    while (lo < hi)
    {
        size_t mid = (lo + hi) / 2;
        void *p = (char *)array + mid * type_size;
        if (cmp(p, value) < 0)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

static inline Phase mul_freq_time(Frequency freq, Time time)
{
    Phase p = 2 * ((freq % s) * (time % s) % s);
    p = p * 0x400000000000 / 30517578125 * 4 + p * 0x400000000000 % 30517578125 * 4 / 30517578125;
    return p;
}

static inline Time dev_phase_freq(Phase phase, Frequency freq)
{
    Time t = phase * s / freq / 2 / pi;
    return t;
}

static inline Frequency dev_phase_time(Phase phase, Time time)
{
    Frequency f = phase * s / time / 2 / pi;
    return f;
}

static inline Phase double_2_phase(double d)
{
    return (Phase)(d / M_PI * pi);
}

static inline double phase_2_double(Phase p)
{
    return (double)p * M_PI / pi;
}

static inline Time double_2_time(double d)
{
    return (Time)(d * s);
}

static inline double time_2_double(Time t)
{
    return (double)t / s;
}

static inline Frequency double_2_freq(double d)
{
    return (Frequency)d;
}

static inline double freq_2_double(Frequency f)
{
    return (double)f;
}

#endif // WAVEFORM_H
