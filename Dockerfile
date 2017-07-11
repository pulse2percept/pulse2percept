FROM andrewosh/binder-base

MAINTAINER Michael Beyeler <mbeyeler@uw.edu> 

USER main

# Install p2p for both Py2 and Py3 kernels
RUN pip install pulse2percept
RUN /bin/bash -c "source activate python3 && pip install pulse2percept && source deactivate"
