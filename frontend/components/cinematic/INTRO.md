# Video introduction

The root route renders `CinematicIntro`; the existing product routes and
auth-independent workspace remain intact. The previous procedural scene
components are retained but are no longer loaded by the root route.

`animation/introStory.ts` owns all normalized story ranges. Anime.js v4 paused
animations are sought from scroll progress. Video seeking is independent:
passive scroll updates refs, RAF interpolates at a frame-rate-independent
equivalent of 0.16 per frame, and settles at the exact target after 110 ms
without another scroll event. Concurrent seeks are avoided; `seeked` resumes
pending work. The final target is duration minus 45 ms to retain the last frame.

Desktop uses 700vh; narrow screens use 600vh and contain the video to preserve
its composition. Reduced motion uses one viewport, a static final frame, and
immediate login access. Video errors also expose the entrance immediately.

No authentication exists in this repository. The email/password controls are
a visual shell; sign-in is disabled and demo workspace access is a separate
link. No passwords are stored or sent. Replace this form with the real auth
adapter when it exists. Skip/replay share the same scroll endpoint helper;
future returning-user behavior can call that helper after an explicit policy
is adopted. No viewing history is currently persisted.

Verification: production build, TypeScript, and diff whitespace checks pass.
There is no frontend test script. The inherited `next lint` script is invalid
on Next 16. No browser was connected during implementation, so viewport
screenshots and slow/fast/reverse/trackpad performance remain unverified.

Manual review checkpoints: 0%, 20%, 35%, 50%, 65%, 80%, 90%, 100%.
At each checkpoint verify the decoded frame, readable copy, no overlaps,
and development-only debug progress. Also test Skip Intro, reverse from login,
keyboard input, demo link, mobile portrait, reduced motion, failed video,
and back navigation. Dev debug information never appears in production.
