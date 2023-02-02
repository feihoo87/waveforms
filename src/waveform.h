#ifndef WAVEFORM_H
#define WAVEFORM_H

#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288 /* pi */
#endif

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

typedef double (*FuncPtr)(double, size_t, const Number *);

static inline int64_t mul_mod(int64_t a, int64_t b, int64_t m)
{
    int64_t r = 0;
    while (b)
    {
        if (b & 1)
        {
            r = (r + a) % m;
        }
        a = (a + a) % m;
        b >>= 1;
    }
    return r;
}

static inline uint64_t gcd(uint64_t a, uint64_t b)
{
    uint64_t c;
    while (b)
    {
        c = a % b;
        a = b;
        b = c;
    }
    return a;
}

static inline Phase mul_freq_time(Frequency freq, Time time)
{
    Phase p = (Phase)mul_mod(freq, time, Hz * s);
    p = 9 * p + p / 0x5 + 2 * p / 0x7d + 4 * p / 0x271 + 3 * p / 0xc35 + 4 * p / 0x5f5e1 + 3 * p / 0x1dcd65 + 2 * p / 0x9502f9 + 2 * p / 0x2e90edd + 3 * p / 0xe8d4a51 + 3 * p / 0x48c27395 + 2 * p / 0x16bcc41e9 + 3 * p / 0x2386f26fc1 + p / 0xb1a2bc2ec5 + 2 * p / 0x3782dace9d9;
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

static inline Time one_over_freq(Frequency f)
{
    return s * Hz / f;
}

static inline Frequency one_over_time(Time t)
{
    return s * Hz / t;
}

#endif // WAVEFORM_H
