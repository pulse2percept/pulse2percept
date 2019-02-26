#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y freetype freetype-devel libpng-devel gcc

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ ${PYBIN} = *"cp2"* ]]; then
        "${PYBIN}/pip" install numpy==1.6.2
        "${PYBIN}/pip" install matplotlib==1.3.1
    fi
    if [[ ${PYBIN} = *"cp3"* ]]; then
        "${PYBIN}/pip" install numpy==1.7
        "${PYBIN}/pip" install matplotlib==1.3.1
    fi
    "${PYBIN}/pip" install -r /io/requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/pulse2percept*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install pulse2percept --no-index -f /io/wheelhouse
    #(cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
done
