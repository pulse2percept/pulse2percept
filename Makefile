# simple makefile to simplify repetitive build env management tasks under posix

PYTHON ?= python
PYTEST ?= pytest
FLAKE ?= flake8

all: inplace install

clean:
	$(PYTHON) setup.py clean
	rm -rf build

distclean: clean
	rm -rf dist

inplace:
	$(PYTHON) setup.py build_ext -i

install:
	$(PYTHON) setup.py install

doc: install
	$(MAKE) -C doc html

tests: inplace install
	$(PYTEST) --showlocals -v pulse2percept --durations=20

flake:
	$(FLAKE) --ignore N802,N806,W504 --select W503 `find . -name \*.py | grep -v setup.py | grep -v __init__.py | grep -v /doc/`

help:
	@ echo 
	@ echo "pulse2percept Makefile options:"
	@ echo 
	@ echo "make               Compiles pulse2percept"
	@ echo "make tests         Compiles pulse2percept and runs the test suite"
	@ echo "make doc           Compiles pulse2percept and generates the documentation"
	@ echo "make clean         Cleans out all build files"
	@ echo "make distclean     Cleans out all build and dist files"
	@ echo "make help          Brings up this message"
