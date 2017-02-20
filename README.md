[![Coverage Status](https://coveralls.io/repos/github/uwescience/pulse2percept/badge.svg?branch=master)](https://coveralls.io/github/uwescience/pulse2percept?branch=master)


# pulse2percept: Models for Retinal Prosthetics

## Summary

By 2020 roughly 200 million people will suffer from retinal diseases such as
macular degeneration or retinitis pigmentosa. Consequently, a variety of retinal
sight restoration procedures are being developed to target these diseases.
Electronic prostheses (currently being implanted in patients) directly stimulate
remaining retinal cells using electrical current, analogous to a cochlear
implant. Optogenetic prostheses (soon to be implanted in human) use optogenetic
proteins to make remaining retinal cells responsive to light, then use light
diodes (natural illumination is inadequate) implanted in the eye to stimulate
these light sensitive cells.

However, these devices do not restore anything resembling natural vision:
Interactions between the electronics and the underlying neurophysiology result
in significant distortions of the perceptual experience.

We have developed a computer model that has the goal of predicting the perceptual
experience of retinal prosthesis patients.
The model was developed using a variety of patient data describing the brightness
and shape of phosphenes elicited by stimulating single electrodes, and validated
against an independent set of behavioral measures examining spatiotemporal
interactions across multiple electrodes.

The model takes as input a series of (simulated) electrical pulse trains---one pulse
train per electrode in the array---and converts them into an image sequence that
corresponds to the predicted perceptual experience of a patient:

<img src="doc/_static/model.jpg" width="100%"/>



## Installation

### Prerequisites
pulse2percept requires the following software installed for your platform:

1. [Python](http://www.python.org) 2.7 or >= 3.4

2. [NumPy](http://www.numpy.org)

3. [SciPy](http://www.scipy.org)

4. [Numba](http://numba.pydata.org/)

Optional packages:

1. [JobLib](https://github.com/joblib/joblib) for parallel processing.

2. [Dask](https://github.com/dask/dask) for parallel processing
   (a joblib alternative).

3. [scikit-image](http://scikit-image.org/) if you want to use
   the `image2pulsetrain` function.

4. [ffmpeg codec](http://adaptivesamples.com/how-to-install-ffmpeg-on-windows)
   if you're on Windows and want to use functions in the `files`
   module.


### Development version

In order to get the latest version of pulse2percept,
use the commands:

```
$ git clone https://github.com/uwescience/pulse2percept.git
$ cd pulse2percept
$ python setup.py install
```

To test pulse2percept after installation, execute in Python:
```
>>> import pulse2percept
```

### Getting started

A number of useful examples can be found in the "examples/notebooks"
folder, including the following:

- [0.0-example-usage.ipynb](https://github.com/uwescience/pulse2percept/blob/master/examples/notebooks/0.0-example-usage.ipynb): How to use the model.

- [0.1-image2percept.ipynb](https://github.com/uwescience/pulse2percept/blob/master/examples/notebooks/0.1-image2percept.ipynb): How to convert an image to a percept.

- [1.0-exp-nanduri2012.ipynb](https://github.com/uwescience/pulse2percept/blob/master/examples/notebooks/1.0-exp-nanduri2012.ipynb): A notebook reproducing the findings described
   in Nanduri et al. (2012).

Detailed documentation can be found at [uwescience.github.io/pulse2percept](https://uwescience.github.io/pulse2percept).
