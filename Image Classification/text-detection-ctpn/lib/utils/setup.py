from Cython.Build import cythonize
import numpy as np
from distutils.core import setup,Extension

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()



ext_modules = [
    Extension(
        "utils.bbox",
        ["bbox.c"],
        #extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
    Extension(
        "utils.cython_nms",
        ["cython_nms.c"],
        #extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
]
setup(
    #ext_modules=cythonize(["bbox.pyx","cython_nms.pyx"],include_dirs=[numpy_include]),
    #ext_modules=cythonize(["bbox.pyx","cython_nms.pyx"]),
    ext_modules = ext_modules
)
