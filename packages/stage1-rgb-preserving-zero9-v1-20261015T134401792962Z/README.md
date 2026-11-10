# A2 RGB-preserving initialization package (unexecuted)

Current working-tree source snapshot, not a clean-commit release. No weights included.
Use `config-rgb-preserving-zero9-v1.yaml` explicitly for the new experiment.
It derives from the historical last-four/fp32-vision config, preserving its optimizer
and schedule. It adds the versioned initialization and disables implicit resume.
Historical configs are retained byte-for-byte as controls; default/all-mode config
selection still selects historical controls. No training is authorized by packaging.

Example for a separately authorized future experiment:

```
python run.py smoke --config config-rgb-preserving-zero9-v1.yaml --checkpoint <pinned-local-weights> --patch-root <patches> --output <fresh-output.json>
```

Initialization is included in smoke/full config identities and checkpoint metadata.
New explicit configs reject absent or mismatched initialization/config identities.
Conversion runs before trained-state loading and rejects a 12-channel projection.
CPU equivalence tests use synthetic filters only. Pretrained equivalence is pending
locally unavailable pinned weights; GPU/training/resume execution is unverified.

`package-manifest.json` records the current source hashes and payload checksums;
its own bytes are excluded from that inventory. Historical revision.json only
specifies the pinned base-weight contract; historical experiment evidence does
not validate this new initialization. Copied config git_sha fields are historical;
source_snapshot in the manifest describes this package's actual code provenance.
