# SatQuery intro asset

`satquery-intro.mp4` is the newest matching generated video from Downloads:
`Camera_descending_from_space_to_202609070058.mp4` (7 September 2026).

Copied unchanged. H.264, 1280 × 720, 24 fps, approximately 10 seconds,
10.7 MB. Keyframes occur at 0, 3, 6, and 9 seconds. Production code uses
only `/videos/satquery-intro.mp4`; no external asset requests are needed.

The long keyframe interval is a potential reverse-seeking bottleneck.
Browser seek performance has not been verified. If slow/reverse scrolling
shows delayed frames, re-encode with frequent keyframes and faststart before
increasing smoothing. Preserve the original source outside public assets.
