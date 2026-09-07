Bundled sample assets
=====================

Assets loaded by :py:mod:`pulse2percept.stimuli.samples`. pulse2percept's BSD
license covers the source code; it does not automatically extend to the files
in this directory. Each entry below states its own terms.

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

``bvl-cake.jpg``
----------------

Photograph of a cake decorated with the Bionic Vision Lab logo, 435x495 RGB.

:License: BSD 3-Clause, the same terms as pulse2percept itself, as granted by
          the copyright holder. No separate third-party license applies.
:Source: Supplied by the Bionic Vision Lab.
:Modifications: None; the file is bundled as supplied.

The logo depicted on the cake is still the lab's mark; see the entry above.

``ucsb-surf.jpg``
-----------------

Frame of the UCSB coastline, 845x476 RGB.

:License: U.S. government work, in the public domain in the United States.
          This file is **not** covered by pulse2percept's BSD license.
:Attribution: Courtesy of the National Library of Medicine.
:Source: A frame extracted from *Towards a Smart Bionic Eye*
         (https://www.youtube.com/watch?v=m2WXEjewqho), produced by the
         National Library of Medicine / National Institutes of Health. The
         timestamp of the frame within the video was not recorded.
:Modifications: Extracted as a single frame and saved as JPEG; not cropped,
                resized, or color-corrected.

:py:func:`~pulse2percept.stimuli.samples.ucsb_surf` carries the attribution in
``metadata``.

``cajal-retina.jpg``
--------------------

Santiago Ramón y Cajal's drawing of the layered structure of the retina,
500x745 RGB.

:License: Public domain. Cajal died in 1934, so the drawing is out of
          copyright worldwide. This file is **not** covered by
          pulse2percept's BSD license, but no permission is needed to use it.
:Source: https://commons.wikimedia.org/wiki/File:Cajal_Retina.jpg
:Modifications: None; the file is bundled as downloaded.

``zebrafish-retina.jpg``
------------------------

*Sunrise in the eye: zebrafish retina*, a fluorescence micrograph, 760x544
RGB.

:License: Creative Commons Attribution 4.0 International
          (https://creativecommons.org/licenses/by/4.0/)
:Attribution: Dr Kara Cerveny & Dr Steve Wilson. Source: Wellcome Collection.
:Source: https://wellcomecollection.org/works/tkeg5zqn/images?id=ca6k4v9j
:Modifications: None; the file is bundled as downloaded.

This micrograph remains under CC BY 4.0. It is **not** relicensed under
pulse2percept's BSD license, and redistributing it -- including as part of a
pulse2percept release -- requires the attribution above.
:py:func:`~pulse2percept.stimuli.samples.zebrafish_retina` carries a short
form of it in ``metadata``.
