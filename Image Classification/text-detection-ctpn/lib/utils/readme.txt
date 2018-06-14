https://github.com/eragonruan/text-detection-ctpn/issues/43

CPU Setting:
(1) Set "USE_GPU_NMS " in the file ./ctpn/text.yml as "False"
(2) Set the "__C.USE_GPU_NMS" in the file ./lib/fast_rcnn/config.py as "False";
(3) Comment out the line "from lib.utils.gpu_nms import gpu_nms" in the file ./lib/fast_rcnn/nms_wrapper.py;
(4) To rebuild the setup.py in path "[path]/text-detection-ctpn/lib/utils/setup.py":
(5) cd xxx/text-detection-ctpn-master/lib/utils
and execute:python setup.py build
(6) copy the .so file from the "build" directory to the
xxx/text-detection-ctpn-master/lib/utils.
(7) cd xxx/text-detection-ctpn-master
and execute: python ./ctpn/demo.py

ps: no need to set env variable "CFLAGS"

you may also need to modify some python file, just follow the error, it is pretty straight-forward

setup.py is like blow:

from Cython.Build import cythonize
import numpy as np
from distutils.core import setup
from distutils.extension import Extension

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        'bbox',
        sources=['bbox.c'],
        include_dirs = [numpy_include]
    ),
    Extension(
        'cython_nms',
        sources=['cython_nms.c'],
        include_dirs = [numpy_include]
    )
]
setup(
    ext_modules=ext_modules
)
