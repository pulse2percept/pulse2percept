Installation
============

********************
Windows Installation
********************

Pre-Installations
#################
-use either conda or pip, but not both, to download things.

Anaconda
*********************`<http://www.python.org/>`_
download anaconda ('Anaconda <https://www.anaconda.com/distribution/>'_)
Check if it is downloaded: look in folder that you installed it in. Run conda --version.
Check which version of Python your machine defaults to with python --version, and if it is not >= 3.4,
go to system properties -> environment variables -> Path (under System variables) -> add a path to the
desired Python version (this requies you to have a viable python version downloaded). 

Numpy
*********************
install numpy (pip/conda install numpy)
Check version / if it is on your machine: python -c "import numpy; print(numpy.__version__)" to see 
that it is installed.

Cython
*********************
download cython (' Cython <https://github.com/cython/cython/wiki/InstallingOnWindows>'_) ensure that you also install
a C compiler, which is linked under the heading 'Using Anaconda').

VS 2019
*********************
download visual studio 2019 (make sure you get the build tools. Desktop development with C++ should work)

Recommended: Install make
* Go to ezwinports: 'make <https://sourceforge.net/projects/ezwinports/files/>'_
* Download make-4.1-2-without-guile-w32-bin.zip (get the version without guile)
* Extract zip
* Now you can install pulse2percept by typing make instead of the python setup.py stuff!

Installation
############

(pip /conda) install pulse2percept

git clone https://github.com/uwescience/pulse2percept.git
cd pulse2percept
python setup.py install (otherwise use make)