# Grid-cell conformal isometry

[日本語版 README](README_ja.md)

This repository is based on the official implementation of
[On Conformal Isometry of Grid Cells: Learning Distance-Preserving Position Embedding](https://arxiv.org/abs/2405.16865).

The current implementation focuses on one training condition that reliably
learns hexagonal grid-cell activity:

- a continuous SIREN position encoder with non-negative Softplus output;
- a direction-conditioned MLP that generates the transformation matrix from
  `[cos(theta), sin(theta)]`;
- one random movement direction per sample;
- a fixed isometry displacement of `dr = 5`;
- a squared-distance isometry loss;
- a loss-based activity norm constraint.

Legacy table encoders, ordinary MLP encoders, angle bins, variable-distance
isometry losses, and multi-direction experiment branches have been removed so
that the training path is explicit in the code.

## Setup

Python 3.12 is supported. Install dependencies with:

```bash
uv sync
```

## Training

Run the supported configuration with:

```bash
uv run python main.py \
  --config=configs/siren_scale10.py \
  --workdir=./logs
```

Each invocation creates a timestamped run directory under `logs`. Training
metrics are written to `metrics.csv`. PyTorch checkpoints are written every
500 steps to `ckpt/checkpoint-step*.pth`; their configuration and NumPy random
state are stored in the matching JSON files. The `.pth` files can be loaded
with `torch.load(..., weights_only=True)`.

## Visualization

After training, replace `RUN_DIR` with the timestamped run directory:

```bash
uv run python visualize_results.py RUN_DIR
```

To visualize a specific checkpoint, append `--step 5000`. The script writes
the receptive fields, generated direction-dependent matrices, losses, activity
diagnostics, and module-norm diagnostics below `RUN_DIR/visualizations`.

Measure hexagonal and square gridness with:

```bash
uv run python analyze_gridness.py RUN_DIR
```

The CSV scores and spatial-autocorrelation plots are written below
`RUN_DIR/gridness`.
