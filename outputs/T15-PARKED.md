# T15 flood U-Net parked

Soham parked the U-Net after the preregistered gates failed.

- S1-only eight-chip IoU: 0.897. Gate: >0.9.
- T15d D1 FAIL: S1+S2 eight-chip IoU was 0.965 (>0.9), but worst-chip IoU was 0.744 (required >0.95). S2-only worst-chip IoU was 0.842.
- D2 verdict: pipeline cause by the registered rule. The specific cause remains unresolved.
- T15d preregistration: `7daad8d`. Diagnostic branch and report-reader reference: `7fd5d12`.

Open leads: audit worst-chip labels, alignment, and nodata. Check S1 scaling because S1+S2 performed worse than S2 alone on the worst chip.

Flood answers use per-scene `flood-otsu`, which remains `experimental_unvalidated`. Its India-test IoU was 0.631 against a 0.65 gate. Shipped B1 scored 0.704.

Resume only by Soham's decision; gates unchanged.
