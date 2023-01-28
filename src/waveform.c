#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "waveform.h"

#define FUNC_TAB_SIZE 128

static size_t wave_func_count = 0;
FuncPtr wave_function_table[FUNC_TAB_SIZE];

static int cmp_wav1(const void *a, const void *b)
{
    Wave1 *wa = (Wave1 *)a;
    Wave1 *wb = (Wave1 *)b;
    return wa->func > wb->func ? 1 : wa->func < wb->func ? -1
                                                         : 0;
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
    double y = wave_function_table[wave->func](x, wave->argc, wave->argv);
    if (wave->n != 1 || wave->d != 1)
    {
        y = pow(y, (double)wave->n / wave->d);
    }
    return y;
}

double wave_apply_wave2(Time t, Wave2 *wave)
{
    double y = wave->amplitude;
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
        y += wave_apply_wave2(t, wave->waves + i);
    }
    return y;
}

double wave_apply_waveform(Time t, Waveform *wave)
{
    size_t i = bisect_left(wave->waves, t, 0, wave->size, sizeof(Wave3), cmp_wav3);
    if (i == wave->size)
    {
        return 0.0;
    }
    return wave_apply_wave3(t, wave->waves + i);
}

void wave_sample_waveform(Waveform *wave, double *samples, size_t size, Time start, Time dt)
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

void wave_sample_waveform_tlist(Waveform *wave, size_t size, const Time *tlist, double *samples)
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
