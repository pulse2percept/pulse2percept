# simple makefile to simplify repetitive build env management tasks under posix

PIP ?= pip3
PYTHON ?= python3
PYTEST ?= pytest
FLAKE ?= flake8

all: install

clean:
	$(PYTHON) setup.py clean
	rm -rf build

distclean: clean
	rm -rf dist

install:
	$(PIP) install -e .

uninstall:
	$(PIP) uninstall pulse2percept -y

doc: install
	$(MAKE) -C doc html

tests: install
	$(PYTEST) --doctest-modules --showlocals -v pulse2percept --durations=20

flake:
	$(FLAKE) --ignore N802,N806,W504 --select W503 `find . -name \*.py | grep -v setup.py | grep -v __init__.py | grep -v /doc/`

help:
	@ echo 
	@ echo "pulse2percept Makefile options:"
	@ echo 
	@ echo "make               Installs pulse2percept"
	@ echo "make uninstall     Uninstalls pulse2percept"       
	@ echo "make tests         Installs pulse2percept and runs the test suite"
	@ echo "make doc           Installs pulse2percept and generates the documentation"
	@ echo "make clean         Cleans out all build files"
	@ echo "make distclean     Cleans out all build and dist files"
	@ echo "make help          Brings up this message"
