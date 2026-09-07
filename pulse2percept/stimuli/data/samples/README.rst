Bundled sample assets
=====================

Assets loaded by :py:mod:`pulse2percept.stimuli.samples`. None of them are
covered by pulse2percept's BSD license, which applies to the source code only;
each entry below states its own terms.

``big-buck-bunny.mp4``
----------------------

*Big Buck Bunny*, © copyright 2008, Blender Foundation /
www.bigbuckbunny.org.

:License: Creative Commons Attribution 3.0
          (https://creativecommons.org/licenses/by/3.0/), as stated by the
          Peach open movie project at https://peach.blender.org/about/
:Source: ``BigBuckBunny_640x360.m4v`` from the official Blender Foundation
         release, https://download.blender.org/peach/bigbuckbunny_movies/
         (linked from the project download page,
         https://peach.blender.org/download/)
:Modifications: Frames 2711-2825 of the original (1:52.958 to 1:57.750 at
                24 fps), just after the cut to the camera view, with the audio
                track dropped. The video was stream-copied, not re-encoded:
                all 115 frames are bit-identical to the source. 4.79 s of
                H.264 in an MP4 container. The 640x359 frame size is the
                original's -- the official release is 640x359 despite being
                named 640x360.

This clip remains under CC BY 3.0. It is **not** relicensed under
pulse2percept's BSD license, and redistributing it -- including as part of a
pulse2percept release -- requires the attribution above.
:py:func:`~pulse2percept.stimuli.samples.big_buck_bunny` carries a short form
of it in ``metadata``.

``bionic-vision-lab.png``, ``ucsb.png``
---------------------------------------

Logos of the Bionic Vision Lab and of the University of California, Santa
Barbara.

:License: None granted. These are the marks of their respective owners,
          included with permission for use in pulse2percept's own demos,
          docs, and tests.
:Source: Supplied by the trademark owners.

Redistributing pulse2percept carries these files along, but that is not a
license to use either logo for any other purpose. In particular, the UCSB
logo is a University of California trademark: reuse outside pulse2percept
requires permission from the University, not from pulse2percept.
