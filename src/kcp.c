#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "ikcp.h"
#include <stdint.h>

static PyObject *KCPError;

char kcp_doc[] = "";

static int output(const char *buf, int len, ikcpcb *kcp, void *user)
{
    PyObject *buff = Py_BuildValue("y#", buf, (Py_ssize_t)len);
    PyObject *kcp_obj = (PyObject *)user;
    PyObject_CallMethod(kcp_obj, "output", "O", buff);
    Py_DECREF(buff);
    return 0;
}

static PyObject *kcp_create(PyObject *self, PyObject *args)
{
    int conv;
    PyObject *user;
    if (!PyArg_ParseTuple(args, "iO", &conv, &user))
        return NULL;
    ikcpcb *kcp = ikcp_create((IUINT32)conv, (void *)user);
    kcp->output = output;
    return PyLong_FromLong((int64_t)kcp);
}

static PyObject *kcp_release(PyObject *self, PyObject *args)
{
    int64_t kcp;
    if (!PyArg_ParseTuple(args, "l", &kcp))
        return NULL;
    ikcp_release((ikcpcb *)kcp);
    return PyLong_FromLong(0);
}

static PyObject *kcp_update(PyObject *self, PyObject *args)
{
    int64_t kcp;
    IUINT32 current;
    if (!PyArg_ParseTuple(args, "li", &kcp, &current))
        return NULL;
    ikcp_update((ikcpcb *)kcp, current);
    return PyLong_FromLong(0);
}

static PyObject *kcp_check(PyObject *self, PyObject *args)
{
    int64_t kcp;
    IUINT32 current;
    if (!PyArg_ParseTuple(args, "li", &kcp, &current))
        return NULL;
    int ret = ikcp_check((ikcpcb *)kcp, current);
    return PyLong_FromLong(ret);
}

static PyObject *kcp_input(PyObject *self, PyObject *args)
{
    int64_t kcp;
    const char *data;
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "ly#", &kcp, &data, &size))
        return NULL;
    int ret = ikcp_input((ikcpcb *)kcp, data, (long)size);
    return PyLong_FromLong(ret);
}

static PyObject *kcp_setmtu(PyObject *self, PyObject *args)
{
    int64_t kcp;
    int mtu;
    if (!PyArg_ParseTuple(args, "li", &kcp, &mtu))
        return NULL;
    int ret = ikcp_setmtu((ikcpcb *)kcp, mtu);
    return PyLong_FromLong(ret);
}

static PyObject *kcp_wndsize(PyObject *self, PyObject *args)
{
    int64_t kcp;
    int sndwnd, rcvwnd;
    if (!PyArg_ParseTuple(args, "lii", &kcp, &sndwnd, &rcvwnd))
        return NULL;
    int ret = ikcp_wndsize((ikcpcb *)kcp, sndwnd, rcvwnd);
    return PyLong_FromLong(ret);
}

const char kcp_nodelay_doc[] = "kcp_nodelay\n\
\n\
    nodelay: 0:disable(default), 1:enable\n\
    interval: internal update timer interval in millisec, default is 100ms\n\
    resend: 0:disable fast resend(default), 1:enable fast resend\n\
    nc: 0:normal congestion control(default), 1:disable congestion control\n\
\n\
    fastest: kcp_nodelay(kcp, 1, 20, 2, 1)\n";
static PyObject *kcp_nodelay(PyObject *self, PyObject *args)
{
    int64_t kcp;
    int nodelay, interval, resend, nc;
    if (!PyArg_ParseTuple(args, "liiii", &kcp, &nodelay, &interval, &resend, &nc))
        return NULL;
    int ret = ikcp_nodelay((ikcpcb *)kcp, nodelay, interval, resend, nc);
    return PyLong_FromLong(ret);
}

const char kcp_recv_doc[] = "user/upper level recv: returns bytes, returns below zero for EAGAIN";
static PyObject *kcp_recv(PyObject *self, PyObject *args)
{
    int64_t kcp;
    char buf[1500];
    int ret;
    PyObject *ret_obj;

    if (!PyArg_ParseTuple(args, "l", &kcp))
        return NULL;

    ret = ikcp_recv((ikcpcb *)kcp, buf, 1500);
    if (ret > 0)
        return Py_BuildValue("y#", buf, ret);

    int size = ikcp_peeksize((ikcpcb *)kcp);
    if (size <= 0)
        return PyLong_FromLong(size);

    char *buffer = (char *)malloc(size);
    if (buffer == NULL)
        return PyLong_FromLong(-4);
    ret = ikcp_recv((ikcpcb *)kcp, buffer, size);
    if (ret < 0)
    {
        free(buffer);
        return PyLong_FromLong(ret);
    }
    ret_obj = Py_BuildValue("y#", buffer, ret);
    free(buffer);
    return ret_obj;
}

const char kcp_send_doc[] = "user/upper level send, returns below zero for error";
static PyObject *kcp_send(PyObject *self, PyObject *args)
{
    int64_t kcp;
    const char *data;
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "ly#", &kcp, &data, &size))
        return NULL;
    int ret = ikcp_send((ikcpcb *)kcp, data, (int)size);
    return PyLong_FromLong(ret);
}

static PyMethodDef KCPMethods[] = {
    {"kcp_create", kcp_create, METH_VARARGS, ""},
    {"kcp_release", kcp_release, METH_VARARGS, ""},
    {"kcp_update", kcp_update, METH_VARARGS, ""},
    {"kcp_check", kcp_check, METH_VARARGS, ""},
    {"kcp_input", kcp_input, METH_VARARGS, ""},
    {"kcp_setmtu", kcp_setmtu, METH_VARARGS, ""},
    {"kcp_wndsize", kcp_wndsize, METH_VARARGS, ""},
    {"kcp_nodelay", kcp_nodelay, METH_VARARGS, kcp_nodelay_doc},
    {"kcp_recv", kcp_recv, METH_VARARGS, kcp_recv_doc},
    {"kcp_send", kcp_send, METH_VARARGS, kcp_send_doc},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef kcpmodule = {
    PyModuleDef_HEAD_INIT,
    "_kcp",  /* name of module */
    kcp_doc, /* module documentation, may be NULL */
    -1,      /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    KCPMethods};

PyMODINIT_FUNC
PyInit__kcp(void)
{
    PyObject *m;

    m = PyModule_Create(&kcpmodule);
    if (m == NULL)
        return NULL;
    return m;
}
