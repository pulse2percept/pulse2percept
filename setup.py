import os
import numpy as np
from setuptools import find_packages, setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension('pulse2percept.fast_retina', ['pulse2percept/fast_retina.pyx'],
              include_dirs=[np.get_include()],
              extra_compile_args=['-O3'])
]

# Get version and release info, which is all stored in pulse2percept/version.py
ver_file = os.path.join('pulse2percept', 'version.py')
with open(ver_file) as f:
    exec(f.read())

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=find_packages(),
            ext_modules=cythonize(extensions),
            install_requires=REQUIRES)


if __name__ == '__main__':
    setup(**opts)
