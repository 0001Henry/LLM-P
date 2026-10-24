"""Rebuild Gensim 4.4.0's dot wrapper without the erroneous -1 sentinel.

BLAS dot products may legitimately return -1. The shipped Cython-generated
wrapper treats that value as an exception and replaces it with zero.
Only the two generated sentinel checks are removed; training logic is unchanged.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def dot_regression():
    import gensim.models.word2vec_inner as inner
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.argtypes = [ctypes.py_object]
    get_name.restype = ctypes.c_char_p
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    capsule = inner.__pyx_capi__['our_dot']
    address = get_pointer(capsule, get_name(capsule))
    function_address = ctypes.c_void_p.from_address(address).value
    ip, fp = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
    function = ctypes.CFUNCTYPE(ctypes.c_float, ip, fp, ip, fp, ip)(function_address)
    n, step = ctypes.c_int(1), ctypes.c_int(1)
    x, y = ctypes.c_float(-1), ctypes.c_float(1)
    actual = float(function(ctypes.byref(n), ctypes.byref(x), ctypes.byref(step), ctypes.byref(y), ctypes.byref(step)))
    if actual != -1.0:
        raise AssertionError(f'BLAS dot(-1, 1) must be -1, got {actual}')
    return actual


def main():
    if '--check' in sys.argv:
        print('dot(-1, 1) =', dot_regression())
        return
    import gensim
    import numpy
    from setuptools import Extension, setup
    if gensim.__version__ != '4.4.0':
        raise RuntimeError('This narrowly scoped repair only supports Gensim 4.4.0')
    package = Path(gensim.__file__).parent
    record_path = ROOT / 'results/dependency_patch.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        import gensim.models.word2vec_inner as inner
        if hashlib.sha256(Path(inner.__file__).read_bytes()).hexdigest() == record['binary_sha256']:
            print('Existing repair verified:', dot_regression())
            return
    source_path = package / 'models/word2vec_inner.c'
    source = source_path.read_text()
    # The float and double variants have exactly these two Cython error checks.
    pattern = r"(__pyx_t_1 = __pyx_v_6gensim_6models_14word2vec_inner_(?:s|ds)dot\([^;]+;)( if \(unlikely\(__pyx_t_1 == \(\((?:float|double)\)-1\.0\)\)\) __PYX_ERR\([^\n]+)"
    patched, count = re.subn(pattern, r'\1', source)
    if count != 2:
        raise RuntimeError(f'Expected two sentinel checks, found {count}; refusing to change dependency')
    build = ROOT / '.cache/gensim-build'
    build.mkdir(parents=True, exist_ok=True)
    patched_path = build / 'word2vec_inner.c'
    patched_path.write_text(patched)
    extension = Extension('gensim.models.word2vec_inner', [str(patched_path)],
                          include_dirs=[numpy.get_include(), str(package / 'models')])
    setup(name='hw2-gensim-dot-fix', ext_modules=[extension],
          script_args=['build_ext', '--build-lib', str(build / 'lib'), '--build-temp', str(build / 'temp')])
    candidates = list((build / 'lib/gensim/models').glob('word2vec_inner*.so'))
    if len(candidates) != 1:
        raise RuntimeError('Expected exactly one compiled extension')
    compiled = candidates[0]
    destination = package / 'models' / compiled.name
    temp = destination.with_suffix('.replacement')
    shutil.copy2(compiled, temp)
    os.replace(temp, destination)
    # A fresh interpreter is essential: the builder already imported the old binary.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--check'], check=True)
    record = {'gensim': gensim.__version__, 'source': 'gensim/models/word2vec_inner.c',
              'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'patched_source_sha256': hashlib.sha256(patched.encode()).hexdigest(),
              'binary_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
              'removed_sentinel_checks': count, 'regression_dot_minus_one': -1.0}
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/dependency_patch.json').write_text(json.dumps(record, indent=2)+'\n')
    print('Installed and verified local Gensim dot-wrapper repair.')


if __name__ == '__main__':
    main()
