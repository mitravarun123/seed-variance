# Laptop pilot results

## Scope

This is a fast exploratory pilot, not the planned p=113 study. It uses addition mod 11, 30% training data (36 ordered input pairs), a 16-wide one-layer transformer with two heads, AdamW (`lr=0.001`, `weight_decay=1.0`), three Phase 2 model seeds, and a 3,000-step horizon. The 85 remaining pairs form the test set. All pilot seeds used the same data split (split hash `fac9aa2a3cbb68a5`). Results below are descriptive; the small task, short horizon, and three seeds do not support broad claims.

## Phase 1

Model seed 42 completed 3,000 steps. It memorized the training set at step 580 and ended with 100% training accuracy, 12.9% test accuracy, and 21.2% best test accuracy. It did not meet the 95% test-accuracy grokking criterion and is right-censored at the 3,000-step horizon.

## Phase 2

Three model seeds completed against the fixed data split. All memorized their training examples and none grokked by step 3,000.

| Model seed | Memorization step | Best test accuracy | Final test accuracy | Grokking by 3,000? |
| ---: | ---: | ---: | ---: | :--- |
| 0 | 190 | 29.4% | 16.5% | No; right-censored |
| 1 | 300 | 7.1% | 2.4% | No; right-censored |
| 2 | 270 | 14.1% | 5.9% | No; right-censored |

Across seeds, mean memorization occurred at step 253. Mean best test accuracy was 16.9%; mean final test accuracy was 8.2%. The result is consistent with memorization without generalization in this short pilot. “Did not grok by 3,000” does not mean these models could never grok with more training.

## Phase 3

Phase 3 completed Fourier, representation, head/MLP ablation, activation-patching, and cross-seed analyses on each seed's initial and final checkpoints. Since none of the models grokked, these describe undertrained/non-generalizing models rather than converged modular-addition circuits.

- Final embedding Fourier-profile cosine averaged 0.913 across the three seed pairs (0.921, 0.968, 0.850). The largest individual frequencies differed: seed 0 selected frequency 1 (energy 0.305), seed 1 frequency 2 (0.292), and seed 2 frequency 4 (0.299). With only five non-DC frequencies at p=11, the spectra remain broad (normalized entropy about 0.95); high whole-profile cosine alone is not evidence of the same algorithm.
- Final-residual CKA averaged 0.232 across seed pairs (range 0.208–0.281), indicating modest representational similarity in this pilot.
- Functional head matching accepted 2 of 6 proposed pairwise matches at the configured 0.25 threshold. The match rate is weak and does not establish a shared head circuit.
- Ablation and patching tables were generated, but effects vary by seed and the models perform poorly on held-out examples. They are not interpretable as evidence that a component implements the target algorithm.

## Phase 4 planning

No Phase 4 training sweep was started. Dry-runs were completed for all five presets using the pilot config and three seeds: core 33 runs, architecture 15, optimization 21, task 9, and data-split 15. The core estimate was about 3.6 minutes and 0.025 GiB, based on measured pilot timings and checkpoint sizes. Because the conditions include larger primes and widths than this pilot, those runtime and storage estimates are likely underestimates for a real sweep.

## Reproducibility and limits

The configured p=113 baseline was started separately and intentionally interrupted at step 976 when the user redirected execution to a laptop-sized pilot. It is not included in the pilot aggregates. The main research question at p=113 remains unanswered.

Configuration: [`configs/laptop_pilot.yaml`](../../configs/laptop_pilot.yaml)

Machine-readable results: Phase 1 [`summary.json`](phase1/seed_42/summary.json), Phase 2 [`aggregate_summary.json`](phase2/aggregate_summary.json) and [`seed_summary.csv`](phase2/seed_summary.csv), Phase 3 [`analysis_summary.json`](phase3/analysis_summary.json), [`fourier_similarity.csv`](phase3/tables/fourier_similarity.csv), and [`circuit_similarity.csv`](phase3/tables/circuit_similarity.csv).
