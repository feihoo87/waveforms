#from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


def get_extensions():
    #from numpy.distutils.misc_util import get_numpy_include_dirs
    #from pathlib import Path

    extensions = [
        Extension('waveforms._waveform', ['src/waveform.c'],
                  include_dirs=['src']),
        Extension('waveforms.math._prime', ['src/prime.c'],
                  include_dirs=['src']),
        Extension('waveforms.sys.net._kcp', ['src/kcp.c', 'src/ikcp.c'],
                  include_dirs=['src']),
        # Extension(
        #     'waveforms.math.npufunc',
        #     ['src/multi_type_logit.c'],
        #     include_dirs=get_numpy_include_dirs(),
        #     library_dirs=[str(Path(p).parent / 'lib') for p in get_numpy_include_dirs()],
        #     libraries=["npymath"],
        #     # extra_compile_args=['-std=c99'],
        # ),
        # cythonize("waveforms/math/prime.pyx"),
    ]

    return extensions


setup(packages=find_packages(), ext_modules=get_extensions())
