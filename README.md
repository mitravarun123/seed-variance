# Seed Variance in Grokking Circuits

## Phase 4: controlled robustness

Phase 4 varies one factor at a time around the Phase 1/2 baseline and stores all artifacts under `results/phase4/` and `checkpoints/phase4/`. The shared baseline is included once in each preset, even when several factor lists contain its value. The default five model seeds are run sequentially. A completed run is skipped on resume; an interrupted run resumes from its latest checkpoint, and failures are retained in `results/phase4/aggregate/failed_runs.csv`.

Preview preset sizes and storage/runtime estimates without training:

```bash
python main.py --config configs/phase4.yaml --phase4-preset core --dry-run
python main.py --config configs/phase4.yaml --phase4-preset architecture --dry-run
python main.py --config configs/phase4.yaml --phase4-preset optimization --dry-run
python main.py --config configs/phase4.yaml --phase4-preset task --dry-run
python main.py --config configs/phase4.yaml --phase4-preset data --dry-run
```

Start a bounded test with reduced seeds and horizon by temporarily editing the Phase 4 config or using `--max-steps`. Plans above `phase4.max_runs_without_confirmation` require `--confirm-large-run` after reviewing the dry-run. `--force` deliberately replaces artifacts for selected Phase 4 condition/seed runs. Tier 1 uses Phase 3's Fourier and cross-seed representation analyses on initial/final checkpoints only; it runs only after all selected seeds for a condition complete. No causal ablations or activation patching are started automatically.

Example for one factor value:

```bash
python main.py --config configs/phase4.yaml --phase4-factor operation --factor-value subtraction --dry-run
```

Add `--model-seed 2` to run a single seed (or `--model-seeds 0 1` for a small replication), and pass `--resume /path/to/latest.pt` only when selecting one condition and one seed. Automatic resume from that condition's Phase 4 `latest.pt` is used by default. Tier 1 runs Fourier and representation CKA for all completed conditions. Set `phase4.analysis_tier` to `2` or `3` and list selected deterministic IDs under `tier2_condition_ids` or `tier3_condition_ids` to run causal ablations or detailed Phase 3 analysis for only those conditions. Tier 2 omits activation patching; Tier 3 includes it. Their outputs are isolated under each condition's `analysis/tier2/` and `analysis/tier3/` directories.

This repository studies whether independently initialized small Transformers learn similar internal algorithms when trained on modular arithmetic. Phase 1 provides a reproducible single-seed baseline; later phases extend it to multi-seed comparisons and mechanistic analysis.

## Phase 1 research task

The baseline learns modular addition:

```text
(a + b) mod p = c
```

For `p = 113`, the dataset contains all `113² = 12,769` ordered input pairs. The code deterministically shuffles those pairs with `data.data_seed`, then assigns the configured fraction to training and the rest to test. The split and its SHA-256 identifiers are saved with the run.

Inputs use four tokens `[a, +, b, =]`. A small causal Transformer is implemented directly in PyTorch. Its attention projections, per-head values, attention patterns, residual stream, MLP activations, and intervention points are exposed for later analysis. It does not use pretrained weights or a transformer framework.

Training uses AdamW and cross-entropy. Full-batch training is the default; microbatches bound activation memory while accumulating gradients for one optimizer step. Metrics are evaluated at the configured interval. Memorization and grokking are recorded separately. Grokking requires the configured training and test accuracy thresholds to hold for a set number of consecutive evaluations, after memorization has been observed. Training then continues for `post_grok_steps`, subject to `max_steps`.

## macOS setup

Python 3.10 or newer is recommended. PyTorch uses Apple MPS when available and otherwise falls back to CPU. The program does not initialize CUDA.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the baseline

```bash
python main.py --config configs/base.yaml
```

The YAML seed defaults to 42. `--seed 123` overrides the model seed and, unless explicitly overridden, the dataset split seed too:

```bash
python main.py --config configs/base.yaml --seed 123
```

To hold the split fixed while changing model initialization:

```bash
python main.py --config configs/base.yaml --seed 123 --data-seed 42
```

The actual experiment is intentionally not run automatically during setup. The command above starts the configured 100,000-step run when you choose to run it.

## Resume an interrupted run

Use the matching configuration and the run's latest checkpoint:

```bash
python main.py --config configs/base.yaml --resume checkpoints/seed_42/latest.pt
```

The checkpoint records model and optimizer state, step, configuration identity, split hash, metric history, and random-number-generator states. Resume rejects mismatched experiment settings, seed, or data split.

## Outputs

Each run is written to `results/seed_<seed>/` and `checkpoints/seed_<seed>/`.

The result directory contains the resolved `config.yaml`, `manifest.json`, `metrics.csv`, `summary.json`, and high-resolution `training_curves.png`. The CSV records step, train/test loss and accuracy, elapsed time, and learning rate. The figure shows accuracy and loss curves and marks confirmed grokking when detected.

The checkpoint directory contains `initial.pt`, `latest.pt`, `best.pt`, and, when applicable, `memorization.pt`, `grokking.pt`, `final.pt`, and selected milestone checkpoints. Existing run artifacts are protected from accidental replacement; use a new seed/output path, resume a checkpoint, or pass `--force` when deliberately starting over.

## Hardware and reproducibility

The device selector checks `torch.backends.mps.is_available()` and chooses MPS or CPU. The run summary records software/platform details, parameter count, seeds, data split hashes, training duration, and whether grokking was observed. Deterministic PyTorch algorithms are requested where supported, with warnings for operations that do not offer deterministic implementations.

The default model has **426,496 trainable parameters**. The count is printed at startup and stored in `summary.json`.

## Project phases

- **Phase 1:** establish a single-seed behavioral baseline and preserve checkpoints/curves. A run that does not grok within its configured horizon is recorded as censored; that outcome is not treated as a software failure.
- **Phase 2:** vary model seeds while holding a deterministic dataset split and the other experiment settings fixed, then quantify seed-to-seed variation.
- **Phase 3:** use saved checkpoints for Fourier, representation, attention/MLP, causal ablation, activation-patching, and cross-seed circuit analyses.
- **Phase 4:** test whether behavioral and mechanistic findings persist across controlled changes to data fraction, prime, architecture, operation, and optimization settings.

Later phase infrastructure in the repository is not launched by the Phase 1 command. No grokking result should be claimed until a real training run produces it.

## Phase 2: multi-seed study

Phase 2 holds the task, split, architecture, optimizer, and training settings fixed while varying only `model_seed`. `configs/multi_seed.yaml` uses the Phase 1 settings, fixes `data_seed: 42`, and lists model seeds 0 through 9. Each seed is trained sequentially to avoid memory pressure on an M1. Initial model hashes and dataset split hashes are checked and recorded in the experiment manifest.

Run the full configured seed list with:

```bash
python main.py --config configs/multi_seed.yaml --multi-seed
```

Run one listed seed independently:

```bash
python main.py --config configs/multi_seed.yaml --model-seed 7
```

To deliberately use a shorter horizon for a smoke run, add `--max-steps N` and use separate roots, for example `--results-root results/smoke_tests --checkpoints-root checkpoints/smoke_tests`. Smoke runs are not research results.

Restart the same `--multi-seed` command to skip completed seeds, resume seeds with `latest.pt`, and continue remaining seeds. An individual seed uses the same behavior with `--model-seed 7`. To resume from a specific checkpoint explicitly, pass both `--model-seed 7 --resume checkpoints/multi_seed/seed_7/latest.pt`.

Per-seed metrics and plots are stored in `results/multi_seed/seed_<seed>/`; checkpoints are isolated in `checkpoints/multi_seed/seed_<seed>/`. Aggregate CSV/JSON, the seed table, trajectory plots, grokking timing/delay figures, distribution, and empirical right-censoring curve are written under `results/multi_seed/`. Runs that do not grok by `max_steps` remain in the aggregate and are marked as not observed within the training horizon. They are not treated as proof that grokking can never occur.

The aggregate reports descriptive variation across seeds. Ten runs provide an initial view, not strong grounds for significance claims. Saved initial, memorization, grokking, and final model checkpoints are the inputs to Phase 3 mechanistic analysis.

## Phase 3: checkpoint analysis

Phase 3 reads Phase 2 checkpoints without training or modifying them. It evaluates number-token embeddings and output vectors in the real Fourier basis of `Z_p`, tracks stage-dependent Fourier energy in residual, attention, and MLP activations, and computes pairwise frequency-profile similarities. It compares residual representations with linear CKA, which is invariant to orthogonal changes of hidden coordinates. Attention signatures summarize query-to-token patterns and head output norms; those descriptions are correlational. Head and MLP ablations and targeted activation patching provide separate causal evidence, measured with accuracy, loss, correct-answer logits, and logit margins. Hungarian head matching uses these functional descriptors and does not treat equal head indices as identities.

The analysis processes one checkpoint at a time. It caches only aligned initial/final residual representations and selected final head activations under `results/phase3/cache/`; cache metadata includes the split, model, sample, and analysis identities. Fourier profiles, stage inventories, ablations, similarities, signatures, and figures are written under `results/phase3/`. Missing memorization/grokking checkpoints remain explicitly missing; an unobserved grokking checkpoint is not fabricated or replaced with another stage.

Run one analysis family or the full configured analysis with:

```bash
python main.py --config configs/phase3.yaml --analysis fourier
python main.py --config configs/phase3.yaml --analysis activations
python main.py --config configs/phase3.yaml --analysis ablation
python main.py --config configs/phase3.yaml --analysis cross-seed
python main.py --config configs/phase3.yaml --analysis all
```

Use `--seed 0` to restrict the run to one seed and `--checkpoint final` to inspect only one stage. The full analysis can be expensive; the commands do not run automatically during setup.
