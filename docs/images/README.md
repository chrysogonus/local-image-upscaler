# Documentation images

[`AGENTS.md`](../../AGENTS.md) forbids committing the output of a local job, with one bounded
exception for curated showcase and documentation images: they live here, they are the
committer's to publish or permissively licensed, they contain no identifiable people, and they
ship sized for display rather than at full 4K/8K. This file is the record that the exception
was met, so that a reader does not have to take it on trust.

## What is in this directory

| File | Dimensions | Shows |
| --- | --- | --- |
| `illustration-creature-comparison.webp` | 3412 × 1165 | Illustration mode at 1:1 output pixels |
| `illustration-waterfall-comparison.webp` | 3407 × 1172 | Illustration mode at 1:1 output pixels |

Each is a two-panel strip cropped at 1:1 output pixels. The **left** panel is this
application's 4K Illustration result. The **right** panel is the unmodified source at the same
zoom, which is why it covers more of the scene: it has half as many pixels across. Both are
WebP-compressed for the web, so the application's own comparison view remains the place to
judge sharpness.

## Provenance

Both source images were created by the maintainer and are published here under this
repository's [licence](../../LICENSE). No third-party photograph, artwork, or stock image is
used, nothing depicts a real person or place, and no identifiable person appears in either.
The 4K panels were produced from those sources by this application's Illustration mode —
Real-ESRGAN's x4 anime model through a local ComfyUI — as described in
[Deployment](../deployment.md#illustration-mode).
