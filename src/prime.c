#include <stdint.h>
#include <Python.h>

#define true 1
#define false 0
#define SIEVE_LIMIT 50000

typedef uint8_t bool;

uint64_t primes[SIEVE_LIMIT];

static inline uint64_t isqrt(uint64_t n, uint64_t lo)
{
    while (lo * lo < n)
    {
        lo++;
    }
    return lo;
}

void sieve(size_t size, uint64_t *primes)
{
    primes[0] = 2;
    size_t i = 1;
    uint64_t n = 3;
    uint64_t limit = 2;
    while (i < size)
    {
        bool is_prime = true;
        uint64_t limit = isqrt(n, limit);
        for (uint64_t *prime = primes; *prime <= limit; prime++)
        {
            if (n % (*prime) == 0)
            {
                is_prime = false;
                break;
            }
        }
        if (is_prime)
        {
            primes[i] = n;
            i++;
        }
        n += 2;
    }
}

static inline uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t n)
{
    uint64_t x = 0;
    while (b != 0)
    {
        if ((b & 1) == 1)
        {
            x = (x + a) % n;
        }
        a = (a + a) % n;
        b >>= 1;
    }
    return x;
}

static inline uint64_t power_mod(uint64_t a, uint64_t b, uint64_t n)
{
    uint64_t x = 1;
    while (b != 0)
    {
        if ((b & 1) == 1)
        {
            x = mul_mod(x, a, n);
        }
        a = mul_mod(a, a, n);
        b >>= 1;
    }
    return x;
}

bool miller_rabin(uint64_t n, uint64_t a)
{
    uint64_t d = n - 1;
    while ((d & 1) == 0)
    {
        d >>= 1;
    }
    uint64_t t = power_mod(a, d, n);
    while (d != n - 1 && t != n - 1 && t != 1)
    {
        t = mul_mod(t, t, n);
        d <<= 1;
    }
    return t == n - 1 || (d & 1) == 1;
}

bool is_prime(uint64_t q)
{
    if (q < 1373653)
    {
        return miller_rabin(q, 2) && miller_rabin(q, 3);
    }
    else if (q < 9080191)
    {
        return miller_rabin(q, 31) && miller_rabin(q, 73);
    }
    else if (q < 4759123141)
    {
        return miller_rabin(q, 2) && miller_rabin(q, 7) && miller_rabin(q, 61);
    }
    else if (q < 2152302898747)
    {
        return (miller_rabin(q, 2) && miller_rabin(q, 3) && miller_rabin(q, 5) && miller_rabin(q, 7) && miller_rabin(q, 11));
    }
    else if (q < 3474749660383)
    {
        return (miller_rabin(q, 2) && miller_rabin(q, 3) && miller_rabin(q, 5) && miller_rabin(q, 7) && miller_rabin(q, 11) && miller_rabin(q, 13));
    }
    else if (q < 341550071728321)
    {
        return (miller_rabin(q, 2) && miller_rabin(q, 3) && miller_rabin(q, 5) && miller_rabin(q, 7) && miller_rabin(q, 11) && miller_rabin(q, 13) && miller_rabin(q, 17));
    }
    else if (q < 3825123056546413051)
    {
        return (miller_rabin(q, 2) && miller_rabin(q, 3) && miller_rabin(q, 5) && miller_rabin(q, 7) && miller_rabin(q, 11) && miller_rabin(q, 13) && miller_rabin(q, 17) && miller_rabin(q, 19) && miller_rabin(q, 23));
    }
    else
    {
        return (miller_rabin(q, 2) && miller_rabin(q, 3) && miller_rabin(q, 5) && miller_rabin(q, 7) && miller_rabin(q, 11) && miller_rabin(q, 13) && miller_rabin(q, 17) && miller_rabin(q, 19) && miller_rabin(q, 23) && miller_rabin(q, 29) && miller_rabin(q, 31) && miller_rabin(q, 37));
    }
}

/*
static PyObject *method_mul_mod(PyObject *self, PyObject *args)
{
    uint64_t a, b, n;

    if (!PyArg_ParseTuple(args, "lll", &a, &b, &n))
    {
        return NULL;
    }
    return PyLong_FromLong(mul_mod(a, b, n));
}

static PyObject *method_power_mod(PyObject *self, PyObject *args)
{
    uint64_t a, b, n;

    if (!PyArg_ParseTuple(args, "lll", &a, &b, &n))
    {
        return NULL;
    }
    return PyLong_FromLong(power_mod(a, b, n));
}
*/

static PyObject *method_miller_rabin(PyObject *self, PyObject *args)
{
    uint64_t q, a;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args, "ll", &q, &a))
    {
        return NULL;
    }
    return PyBool_FromLong(miller_rabin(q, a));
}

static PyObject *method_is_prime(PyObject *self, PyObject *args)
{
    uint64_t q;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args, "l", &q))
    {
        return NULL;
    }
    return PyBool_FromLong(is_prime(q));
}

static PyObject *method_prime(PyObject *self, PyObject *args)
{
    uint64_t i;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args, "i", &i))
    {
        return NULL;
    }
    if (i < 1)
    {
        i = 1;
    }
    if (i > SIEVE_LIMIT)
    {
        i = SIEVE_LIMIT;
    }
    return PyLong_FromLong(primes[i - 1]);
}

static PyMethodDef Methods[] = {
    // {"mul_mod", method_mul_mod, METH_VARARGS, "mul mod"},
    // {"power_mod", method_power_mod, METH_VARARGS, "power mod"},
    {"miller_rabin", method_miller_rabin, METH_VARARGS, "Miller-Rabin test"},
    {"is_prime", method_is_prime, METH_VARARGS, "Miller-Rabin test"},
    {"prime", method_prime, METH_VARARGS, "nth prime"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_prime",
    "",
    -1,
    Methods};

PyMODINIT_FUNC PyInit__prime(void)
{
    sieve(SIEVE_LIMIT, primes);
    PyObject *num = PyLong_FromLong(SIEVE_LIMIT);
    PyObject *m = PyModule_Create(&module);
    PyModule_AddObject(m, "SIEVE_LIMIT", num);
    return m;
}
