# Versioned Stage-1 patch initialization

For a **new experiment**, copy its configuration to a new file and add:

```yaml
patch_embed_initialization: rgb_preserving_zero9_v1
```

The shared Stage-1 builder copies the complete pretrained temporal Conv3d R/G/B
filters into B04/B03/B02 (indices 3/2/1) and zeros the other nine channel filters.
Bias, convolution settings, dtype, device and gradient flags are preserved by
conversion. The builder's existing trainable-master dtype policy still applies.
Omitting the option retains historical mean-and-repeat initialization
(`mean_repeat_v1`); existing configuration hashes are unchanged.

The explicit option participates in both smoke and full-run configuration
identities. New checkpoints also record `patch_embed_initialization`. Explicit
configurations require matching initialization metadata and configuration hashes
on resume; legacy configurations retain their existing identity handling.
Conversion occurs once on the three-channel base projection, before trained
weights are loaded. Conversion rejects an already converted projection.
Do not change the initialization identity to resume an older experiment.

CPU tests use explicitly synthetic filters and model doubles. They verify actual
Qwen temporal/spatial packing and projection equivalence, not pretrained numerical
equivalence. The latter remains pending the pinned local pretrained weights.
Historical configurations and artifacts are unchanged.

Preserved experiment packages are not updated by this change. Build a new package
from the current source before running a new experiment; do not reuse an older
package runner for this option.
