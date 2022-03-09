#!/usr/bin/env python
# encoding: utf-8
"""
setup_helpers.py

Created by Matthew Loper on 2012-10-10.
Copyright (c) 2012 MPI. All rights reserved.
"""
from __future__ import print_function

import distutils.sysconfig as config
import os
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from distutils.core import setup, Extension

from numpy import get_include as numpy_include


@contextmanager
def suppress_output():
    """Prevent error message spew during test-compiles.
    Compiler errors when checking for support of various options can look like build failures,
    which is misleading. Since they happen outside of python code, we temporarily redirect the
    standard output & error file descriptors to dev/null while doing test-compilation.
    """
    null = os.open(os.devnull, os.O_WRONLY)
    stdout, stderr = os.dup(1), os.dup(2)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        yield
    finally:
        os.dup2(stdout, 1)
        os.dup2(stderr, 2)
        os.close(null)


def suppess_warnings():
    """Get rid of stupid warnings about strict prototypes"""
    cvars = config.get_config_vars()
    print(cvars['CFLAGS'])
    #    import pdb; pdb.set_trace()
    suppressed_options = ('-Wstrict-prototypes', '-Wshorten-64-to-32', '-arch i386', '-mno-fused-madd', '-Wall')
    for k, v in cvars.items():
        if isinstance(v, str):
            for option in suppressed_options:
                cvars[k] = cvars[k].replace(option, '')
    print(cvars['CFLAGS'])


suppess_warnings()


def setup_extended(parallel=True, numpy_includes=True, usr_local_includes=True, **kwargs):
    """Like "setup" from distutils, but which tries to compile in parallelism (if requested),
    and adds some other includes we always use.
    """

    kwargs = deepcopy(kwargs)
    if numpy_includes:
        for m in kwargs['ext_modules']:
            m.include_dirs.append(numpy_include())

    if usr_local_includes:
        for m in kwargs['ext_modules']:
            m.include_dirs.append('/usr/local/include')

    if parallel and _omp_available() and _tbb_available():
        kwargs['ext_modules'] = [_with_omp(_with_tbb(m)) for m in kwargs['ext_modules']]
    elif parallel and _omp_available():
        kwargs['ext_modules'] = [_with_omp(m) for m in kwargs['ext_modules']]
    elif parallel and _tbb_available():
        kwargs['ext_modules'] = [_with_tbb(m) for m in kwargs['ext_modules']]

    setup(**kwargs)


def _omp_available():
    if not hasattr(_omp_available, 'r'):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c') as fp:
            _omp_available.r = _module_compiles(_with_omp(_empty_module(fp)))
    return _omp_available.r


def _tbb_available():
    if not hasattr(_tbb_available, 'r'):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c') as fp:
            _tbb_available.r = _module_compiles(_with_tbb(_empty_module(fp)))
    return _tbb_available.r


def _with_omp(module):
    module = deepcopy(module)
    module.libraries.append('gomp')
    module.extra_compile_args.append('-fopenmp')
    module.define_macros.append(('HAVE_OPENMP', '1'))
    return module


def _with_tbb(module):
    module = deepcopy(module)
    module.libraries.append('tbb')
    module.define_macros.append(('HAVE_TBB', '1'))
    return module


def _empty_module(f):
    empty_extension = """
    #include <Python.h>

    static PyObject * setuptest_noop(PyObject *self, PyObject *args) { return NULL; }
    PyMODINIT_FUNC initsetuptest(void) {}

    static PyMethodDef SetuptestMethods[] = {
        {"no-op",  setuptest_noop, METH_VARARGS, "Do nothing."},
        {NULL, NULL, 0, NULL}
    };
    """

    f.write(empty_extension)
    f.flush()
    kwargs = {'name': 'test_setup', 'sources': [f.name], 'extra_compile_args': ['-w']}
    return Extension(**kwargs)


def _module_compiles(module):
    print('Checking support for args {extra_compile_args} and libraries {libraries} ...'.format(**module.__dict__))
    try:
        with suppress_output():
            setup(name='TestSetup', version='1.0', description='test_setup', ext_modules=[module])
        print('Success!')
        return True
    except SystemExit:
        print('Failed!')
    except Exception as e:
        print('Failed with unexpected error:', e)

    return False
