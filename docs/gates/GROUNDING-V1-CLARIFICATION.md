# Grounding v1 baseline controls clarification

This clarification is written before the grounding v1 baseline run.

The registered adapter controls and decision requirements in GROUNDING-V1.md remain in force. The baseline run also records shuffled-expression and constant-grey outputs for base Qwen and Grounding DINO. These baseline controls are descriptive and do not add or change a promotion criterion.

For each dataset, use the registered parent-hash ordering and next-row rotation for the shuffled expressions. Use RGB (128, 128, 128) images at the original dimensions for the grey condition. For DINO, record the same fixed threshold grid for each condition and use the threshold selected on the original select split when reporting eval controls.
