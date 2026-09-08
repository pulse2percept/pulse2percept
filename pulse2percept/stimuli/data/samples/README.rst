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

``ucsb-bike.jpg``
-----------------

Photograph of a cyclist, a pedestrian, a crosswalk, and a stop sign on the
UCSB campus, 900x600 RGB.

:License: BSD 3-Clause, the same terms as pulse2percept itself, as granted by
          the copyright holder. No separate third-party license applies.
:Source: Supplied by the Bionic Vision Lab.
:Modifications: None; the file is bundled as supplied.

``ucsb-surf.jpg``, ``ucsb-flyover.mp4``, ``ucsb-pedestrians.mp4``
-----------------------------------------------------------------

A frame of the UCSB coastline (845x476 RGB), an aerial flyover of the campus
and shoreline (640x346, 53 frames), and a shot of pedestrians on campus
(640x346, 45 frames).

:License: In the public domain in the United States as a work of the U.S.
          government, per NLM's own policy
          (https://www.nlm.nih.gov/web_policies.html#copyright): "Works
          produced by the U.S. government are not subject to copyright
          protection in the United States. Any such works found on National
          Library of Medicine (NLM) Web sites may be freely used or
          reproduced without permission in the U.S." That policy asks for
          the acknowledgement given below, and these files are **not**
          covered by pulse2percept's BSD license.
:Attribution: Courtesy of the National Library of Medicine. This is the
              wording NLM's policy requests.
:Source: Cut from *Towards a Smart Bionic Eye*
         (https://www.youtube.com/watch?v=m2WXEjewqho), produced by the
         National Library of Medicine / National Institutes of Health. The
         timestamps within the video were not recorded.
:Caveats: The basis above is NLM's site-wide policy; no item-level rights
          statement was located for this video, so the claim rests on NLM /
          NIH being credited as its producer. The same policy warns that NLM
          sites also carry third-party content that *is* copyrighted, and
          individual shots within the source video were not separately
          cleared. Section 105 of the U.S. Copyright Act is a United States
          rule, so no claim is made about the status of these files
          elsewhere.
:Modifications: The still was extracted as a single frame and saved as JPEG.
                The two clips were cut without re-encoding and their audio
                tracks dropped. Nothing was cropped, resized, or
                color-corrected. Both clips inherit the source's variable
                frame rate, so a reader resamples them to a constant rate and
                reports 53 and 45 frames rather than the 43 and 36 distinct
                ones they hold.

:py:func:`~pulse2percept.stimuli.samples.ucsb_surf`,
:py:func:`~pulse2percept.stimuli.samples.ucsb_flyover`, and
:py:func:`~pulse2percept.stimuli.samples.ucsb_pedestrians` carry the
attribution in ``metadata``.

``cajal-retina.jpg``
--------------------

Santiago Ramón y Cajal's drawing of the layered structure of the retina,
500x745 RGB.

:License: Public domain, as designated by the source. Wikimedia Commons
          carries the file under its ``PD-old-70`` tag: "in the public
          domain in its country of origin and other countries and areas
          where the copyright term is the author's life plus 70 years or
          fewer." Cajal died in 1934. This file is **not** covered by
          pulse2percept's BSD license.
:Source: https://commons.wikimedia.org/wiki/File:Cajal_Retina.jpg, uploaded
         2006-03-04 and described there as "From 'Structure of the Mammalian
         Retina' c.1900 By Santiago Ramon y Cajal".
:Caveats: The designation is Commons', not a rights statement from a holding
          institution: the file page has no machine-readable author or source
          field, and no separate United States public-domain tag. The
          ``PD-old-70`` term is conditional on the jurisdiction, so no claim
          is made that the drawing is out of copyright everywhere.
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
