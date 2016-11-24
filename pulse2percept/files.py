# -*- npy2savedformats.py -*-
"""
Creates avi files will work on a 64x Windows as well as a PC provided that the
ffmpeg folder is included in the right location

Most uptodate version of ffmpeg can be found here:
https://ffmpeg.zeranoe.com/builds/win64/static/

Used instructions from here with modifications:
http://adaptivesamples.com/how-to-install-ffmpeg-on-windows/

written Ione Fine 7/2016
"""

import subprocess
import numpy as np
import scipy.io as sio


def savemoviefiles(filestr, data, path='savedImages/'):
    np.save(path + filestr, data)  # save as npy
    sio.savemat(path + filestr + '.mat', dict(mov=data))  # save as matfile
    npy2movie(path + filestr + '.avi', data)  # save as avi


def npy2movie(filename, movie, rate=30):
    from PIL import Image
    cmdstring = ('ffmpeg.exe',
                 '-y',
                 '-r', '%d' % rate,
                 '-f', 'image2pipe',
                 '-vcodec', 'mjpeg',
                 '-i', 'pipe:',
                 '-vcodec', 'libxvid',
                 filename
                 )
    print(filename)
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)

    for i in range(movie.shape[-1]):
        im = Image.fromarray(np.uint8(scale(movie[:, :, i], 0, 255)))
        p.stdin.write(im.tobytes('jpeg', 'L'))
    # p.communicate(im.tostring('jpeg','L'))

    p.stdin.close()


def scale(inarray, newmin=0, newmax=1):
    """Scales an image such that its lowest value attains newmin and
    it's highest value attains newmax.
    written by Ione Fine, based on code from Rick Anthony
    6/5/2015
    """

    oldmin = inarray.min()
    oldmax = inarray.max()

    delta = (newmax - newmin) / (oldmax - oldmin)
    outarray = delta * (inarray - oldmin) + newmin
    return outarray
