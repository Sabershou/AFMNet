from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='others',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='others.ext',
            sources=[
                'others/extensions/extra/cloud/cloud.cpp',
                'others/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'others/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'others/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'others/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'others/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
