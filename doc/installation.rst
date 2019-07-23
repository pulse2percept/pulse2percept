Installation
============

*******
Windows
*******

Prerequisites
#################
* use **either** conda **or** pip to download

Anaconda
*********************
http://www.python.org/

download anaconda: https://www.anaconda.com/distribution/

check if it is downloaded: look in the folder that you installed it in

run conda --version

check which version of Python your machine defaults to with ''python --version'', and if it is not >= 3.4, go to system properties -> environment variables -> Path (under System variables) -> add a path to the desired Python version (this requies you to have a viable python version downloaded)

Numpy
*********************
install numpy: ``pip/conda install numpy``

check version / if it is on your machine: ``python -c "import numpy; print(numpy.__version__)"`` to see that it is installed

Cython
*********************
download cython: <https://github.com/cython/cython/wiki/InstallingOnWindows>
 
ensure that you also install a C compiler, which is linked under the heading 'Using Anaconda'

VS 2019
*********************
download visual studio 2019 (make sure you get the build tools. Desktop development with C++ should work)

recommended: install make

* go to ezwinports: 'make <https://sourceforge.net/projects/ezwinports/files/>'_

* download make-4.1-2-without-guile-w32-bin.zip (get the version without guile)

* extract zip

* now you can install pulse2percept by typing make instead of the python setup.py

Installation
############

Stable
*********************
the latest stable release

``(pip /conda) install pulse2percept``

Development
*********************
the most recent pulse2percept

``(pip /conda) install pulse2percept``

``git clone https://github.com/uwescience/pulse2percept.git``

``cd pulse2percept``

``python setup.py install`` (otherwise use make)

***
Mac
***

*****
Linux
*****