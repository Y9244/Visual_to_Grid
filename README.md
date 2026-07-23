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
- one shared movement sample for the transformation and isometry losses;
- a configurable fixed or uniform movement-distance distribution;
- an isometry loss that directly compares Euclidean activity distances;
- a loss-based activity norm constraint.

Legacy table encoders, ordinary MLP encoders, angle bins, separate
loss-specific movement samples, and multi-direction experiment branches have
been removed so that the training path is explicit in the code.

## Setup

Python 3.12 is supported. Install dependencies with:

```bash
uv sync
```

## Training

Run the supported configuration with:

```bash
uv run python main.py \
  --config=configs/siren.py \
  --workdir=./logs
```

Set `seed` in the config, or override it with `--config.seed=1`, to control
both NumPy movement sampling and PyTorch model initialization. Runs with the
same seed and deterministic backend operations reproduce the same random
sequence and usually the same grid orientation.

Each invocation creates a timestamped run directory under `logs`. Training
metrics are written to `metrics.csv`. PyTorch checkpoints are written every
500 steps to `ckpt/checkpoint-step*.pth`; their configuration and NumPy random
state are stored in the matching JSON files. The `.pth` files can be loaded
with `torch.load(..., weights_only=True)`.

Training runs for 20,000 steps. The learning rate warms up to `0.003` over
500 steps, remains there through step 12,000, and then follows cosine decay to
`0.0001`. The learning rate is included in `metrics.csv`.

`data.movement_dr` is the representative movement distance. In uniform mode,
samples are drawn from `movement_dr ± movement_dr_half_width`; fixed mode uses
exactly `movement_dr`. The model automatically sets

```text
s_fixed = target_activity_distance * environment_size / movement_dr
```

so that the target activity distance at the representative movement remains
`target_activity_distance`. Empirically in this model, larger
`movement_dr` selects a longer spatial period (wider grid spacing), while
smaller `movement_dr` selects a shorter period (narrower grid spacing). This is
an empirical property of the finite-displacement objective, not a guarantee
that the lattice remains hexagonal for every setting.

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
