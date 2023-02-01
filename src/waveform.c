#include <Python.h>
#include <stdlib.h>
#include <math.h>
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

static inline Time one_over_freq(Frequency f)
{
    return s * Hz / f;
}

static inline Frequency one_over_time(Time t)
{
    return s * Hz / t;
}

#define FUNC_TAB_SIZE 128

static size_t wave_func_count = 0;
FuncPtr wave_function_table[FUNC_TAB_SIZE];

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

static int cmp_basic_fun(uint32_t func, const Wave1 *a, const Wave1 *b);

static int cmp_wav1(const void *a, const void *b)
{
    Wave1 *wa = (Wave1 *)a;
    Wave1 *wb = (Wave1 *)b;
    if (wa->func > wb->func)
    {
        return 1;
    }
    else if (wa->func < wb->func)
    {
        return -1;
    }
    return cmp_basic_fun(wa->func, wa, wb);
}

static int cmp_wav2(const void *a, const void *b)
{
    Wave2 *wa = (Wave2 *)a;
    Wave2 *wb = (Wave2 *)b;
    for (size_t i = 0; i < wa->size && i < wb->size; i++)
    {
        if (wa->waves[i].func > wb->waves[i].func)
        {
            return 1;
        }
        else if (wa->waves[i].func < wb->waves[i].func)
        {
            return -1;
        }
    }
    return wa->size > wb->size ? 1 : wa->size < wb->size ? -1
                                                         : 0;
}

static int cmp_wav3(const void *a, const void *b)
{
    Wave3 *wa = (Wave3 *)a;
    Wave3 *wb = (Wave3 *)b;
    return wa->upper_bound > wb->upper_bound ? 1 : wa->upper_bound < wb->upper_bound ? -1
                                                                                     : 0;
}

static double wave_basic_func_linear(double t, size_t argc, const Number *argv)
{
    return t;
}

static double wave_basic_func_gaussian(double t, size_t argc, const Number *argv)
{
    double x = t / argv[0].real;
    return exp(-x * x);
}

static double wave_basic_func_erf(double t, size_t argc, const Number *argv)
{
    double x = t / argv[0].real;
    return erf(x);
}

static double wave_basic_func_cos(double t, size_t argc, const Number *argv)
{
    double x = t * argv[0].real;
    return cos(x);
}

static double wave_basic_func_sinc(double t, size_t argc, const Number *argv)
{
    double x = t * argv[0].real;
    if (x == 0.0)
    {
        return 0.0;
    }
    return sin(x) / x;
}

static double wave_basic_func_exp(double t, size_t argc, const Number *argv)
{
    double x = t * argv[0].real;
    return exp(x);
}

static double wave_apply_wave1(Time t, const Wave1 *wave)
{
    double x = time_2_double(t - wave->shift);
    double y = wave_function_table[wave->func](x, wave->argc, wave->argv);
    if (wave->n != 1 || wave->d != 1)
    {
        y = pow(y, (double)wave->n / wave->d);
    }
    return y;
}

static double wave_apply_wave2(Time t, const Wave2 *wave)
{
    double y = wave->amplitude;
    for (size_t i = 0; i < wave->size; i++)
    {
        y *= wave_apply_wave1(t, wave->waves + i);
    }
    return y;
}

static double wave_apply_wave3(Time t, const Wave3 *wave)
{
    double y = 0.0;
    for (size_t i = 0; i < wave->size; i++)
    {
        y += wave_apply_wave2(t, wave->waves + i);
    }
    return y;
}

static double wave_apply_waveform(Time t, const Waveform *wave)
{
    size_t i = bisect_left(wave->waves, t, 0, wave->size, sizeof(Wave3), cmp_wav3);
    if (i == wave->size)
    {
        return 0.0;
    }
    return wave_apply_wave3(t, wave->waves + i);
}

static void wave_sample_waveform(const Waveform *wave, double *samples, size_t size, Time start, Time dt)
{
    Time t = start;
    size_t i = 0;
    for (size_t j = 0; j < size; j++)
    {
        i = bisect_left(wave->waves, t, i, wave->size, sizeof(Wave3), cmp_wav3);
        if (i == wave->size)
        {
            for (; j < size; j++)
            {
                samples[j] = 0.0;
            }
            break;
        }
        samples[j] = wave_apply_wave3(t, wave->waves + i);
        t += dt;
    }
}

static void wave_sample_waveform_tlist(const Waveform *wave, size_t size, const Time *tlist, double *samples)
{
    Time t = tlist[0];
    size_t i = 0;
    for (size_t j = 0; j < size; j++)
    {
        i = bisect_left(wave->waves, t, i, wave->size, sizeof(Wave3), cmp_wav3);
        if (i == wave->size)
        {
            for (; j < size; j++)
            {
                samples[j] = 0.0;
            }
            break;
        }
        samples[j] = wave_apply_wave3(t, wave->waves + i);
        t = tlist[j + 1];
    }
}

static size_t wave_register_basic_func(FuncPtr func)
{
    wave_function_table[wave_func_count] = func;
    return wave_func_count++;
}

static void wave_init_basic_func_table()
{
    wave_register_basic_func(wave_basic_func_linear);
    wave_register_basic_func(wave_basic_func_gaussian);
    wave_register_basic_func(wave_basic_func_erf);
    wave_register_basic_func(wave_basic_func_cos);
    wave_register_basic_func(wave_basic_func_sinc);
    wave_register_basic_func(wave_basic_func_exp);
}

static PyObject *method_fputs(PyObject *self, PyObject *args)
{
    char *str, *filename = NULL;
    int bytes_copied = -1;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args, "ss", &str, &filename))
    {
        return NULL;
    }

    // FILE *fp = fopen(filename, "w");
    // bytes_copied = fputs(str, fp);
    // fclose(fp);

    return PyLong_FromLong(bytes_copied);
}

static PyMethodDef Methods[] = {
    {"fputs", method_fputs, METH_VARARGS, "Python interface for fputs C library function"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_waveform",
    "Python interface for the _waveform C library",
    -1,
    Methods};

PyMODINIT_FUNC PyInit__waveform(void)
{
    wave_init_basic_func_table();
    return PyModule_Create(&module);
}
