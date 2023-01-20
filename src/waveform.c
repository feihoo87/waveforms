#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define FUNC_TAB_SIZE 128

typedef int64_t Time;
typedef int64_t Frequency;
typedef uint64_t Phase;

const Time s = 1000000000000000;
const Time ms = 1000000000000;
const Time us = 1000000000;
const Time ns = 1000000;
const Time ps = 1000;
const Time fs = 1;

const Phase pi = 0x8000000000000000;
const Phase pi2 = 0x4000000000000000;
const Phase pi4 = 0x2000000000000000;
const Phase pi8 = 0x1000000000000000;

int32_t gcd(int32_t a, int32_t b)
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
                                 size_t size, int (*cmp)(const void *, const void *))
{
    while (lo < hi)
    {
        size_t mid = (lo + hi) / 2;
        if (cmp(array + mid * size, value) < 0)
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
    double *amp;
} Wave3;

typedef struct
{
    size_t size;
    Wave3 *waves;
    double *upper_bounds;
} Waveform;

typedef double (*FuncPtr)(double, size_t, Number *);

size_t wave_func_count = 0;
FuncPtr wave_function_table[FUNC_TAB_SIZE];

double wave_basic_func_linear(double t, size_t argc, Number *argv)
{
    return t;
}

double wave_basic_func_gaussian(double t, size_t argc, Number *argv)
{
    double x = t / argv[0].real;
    return exp(-x * x);
}

double wave_basic_func_erf(double t, size_t argc, Number *argv)
{
    double x = t / argv[0].real;
    return erf(x);
}

double wave_basic_func_cos(double t, size_t argc, Number *argv)
{
    double x = t * argv[0].real;
    return cos(x);
}

double wave_basic_func_sinc(double t, size_t argc, Number *argv)
{
    double x = t * argv[0].real;
    if (x == 0.0)
    {
        return 0.0;
    }
    return sin(x) / x;
}

double wave_basic_func_exp(double t, size_t argc, Number *argv)
{
    double x = t * argv[0].real;
    return exp(x);
}

double wave_apply_wave1(Time t, Wave1 *wave)
{
    double x = time_2_double(t - wave->shift);
    return wave_function_table[wave->func](x, wave->argc, wave->argv);
}

double wave_apply_wave2(Time t, Wave2 *wave)
{
    double y = 1.0;
    for (size_t i = 0; i < wave->size; i++)
    {
        y *= wave_apply_wave1(t, wave->waves + i);
    }
    return y;
}

double wave_apply_wave3(Time t, Wave3 *wave)
{
    double y = 0.0;
    for (size_t i = 0; i < wave->size; i++)
    {
        y += wave->amp[i] * wave_apply_wave2(t, wave->waves + i);
    }
    return y;
}

size_t wave_register_basic_func(FuncPtr func)
{
    wave_function_table[wave_func_count] = func;
    return wave_func_count++;
}

void wave_init_basic_func_table()
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

    FILE *fp = fopen(filename, "w");
    bytes_copied = fputs(str, fp);
    fclose(fp);

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
