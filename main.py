"""Command line entry point for single-seed and multi-seed experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from training.config import load_config, validate
from training.trainer import Trainer
from training.utils import device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="YAML experiment configuration")
    parser.add_argument("--seed", type=int, help="Override model seed (and data split seed unless --data-seed is set)")
    parser.add_argument("--data-seed", type=int, help="Override only the deterministic dataset split seed")
    parser.add_argument("--resume", type=Path, help="Resume from a saved .pt checkpoint")
    parser.add_argument("--multi-seed", action="store_true", help="Run the Phase 2 sequential multi-seed study")
    parser.add_argument("--phase4-preset", choices=["core", "architecture", "optimization", "task", "data", "all"], help="Run or plan a controlled Phase 4 preset")
    parser.add_argument("--phase4-factor", choices=["train_fraction", "prime", "width", "depth", "heads", "weight_decay", "learning_rate", "operation", "data_split"], help="Select one Phase 4 factor instead of a preset")
    parser.add_argument("--factor-value", help="Run only this value within --phase4-factor")
    parser.add_argument("--dry-run", action="store_true", help="Print a Phase 4 run and compute plan without training")
    parser.add_argument("--confirm-large-run", action="store_true", help="Confirm a Phase 4 plan above its configured run limit")
    parser.add_argument("--analysis", choices=["all", "fourier", "activations", "ablation", "cross-seed"], help="Run a Phase 3 analysis on saved checkpoints")
    parser.add_argument("--checkpoint", choices=["all", "initial", "memorization", "grokking", "final"], default="all", help="Restrict Phase 3 to one saved training stage")
    parser.add_argument("--model-seed", type=int, help="Run only this model seed (Phase 2)")
    parser.add_argument("--model-seeds", type=int, nargs="+", help="Override the Phase 2 model-seed list")
    parser.add_argument("--results-dir", type=Path, help="Override this run's result directory")
    parser.add_argument("--checkpoints-dir", type=Path, help="Override this run's checkpoint directory")
    parser.add_argument("--results-root", type=Path, help="Override the Phase 2 aggregate/results directory")
    parser.add_argument("--checkpoints-root", type=Path, help="Override the Phase 2 checkpoint root")
    parser.add_argument("--max-steps", type=int, help="Override the training horizon")
    parser.add_argument("--force", action="store_true", help="Allow a fresh run to replace artifacts in its output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.phase4_preset is not None or args.phase4_factor is not None or args.dry_run or args.confirm_large_run or args.factor_value is not None:
        if args.phase4_preset is not None and args.phase4_factor is not None:
            raise SystemExit("Choose either --phase4-preset or --phase4-factor.")
        if args.factor_value is not None and args.phase4_factor is None:
            raise SystemExit("--factor-value requires --phase4-factor.")
        if args.analysis is not None or args.multi_seed or args.seed is not None:
            raise SystemExit("Phase 4 cannot be combined with Phase 1–3 analysis, multi-seed, or --seed flags.")
        if args.model_seed is not None and args.model_seeds is not None:
            raise SystemExit("Choose either --model-seed or --model-seeds for Phase 4.")
        phase4_seeds = [args.model_seed] if args.model_seed is not None else args.model_seeds
        if phase4_seeds is not None and any(seed < 0 for seed in phase4_seeds):
            raise SystemExit("Phase 4 model seeds must be nonnegative.")
        if args.results_dir or args.checkpoints_dir or args.results_root or args.checkpoints_root:
            raise SystemExit("Phase 4 output locations are isolated by configs/phase4.yaml and cannot use Phase 1–3 output overrides.")
        if args.checkpoint != "all":
            raise SystemExit("--checkpoint is available only with --analysis.")
        from experiments.phase4 import run_phase4
        preset = args.phase4_preset or args.phase4_factor or "core"
        try:
            result = run_phase4(config, preset=preset, factor_value=args.factor_value,
                                seeds=phase4_seeds,
                                dry_run=args.dry_run, confirm_large_run=args.confirm_large_run,
                                force=args.force, max_steps=args.max_steps,
                                resume_checkpoint=args.resume)
        except (ValueError, FileExistsError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Phase 4 {result['status']}: {result}", flush=True)
        return

    if args.analysis is not None:
        if args.multi_seed or args.model_seed is not None or args.model_seeds is not None:
            raise SystemExit("Phase 3 uses --analysis and optional --seed; do not combine it with Phase 2 seed flags.")
        if args.resume is not None or args.force or args.max_steps is not None:
            raise SystemExit("Phase 3 reads saved checkpoints; --resume, --force, and --max-steps do not apply.")
        from experiments.phase3 import run_phase3
        selected = [args.seed] if args.seed is not None else config["phase3"]["seeds"]
        run_phase3(config, analysis=args.analysis, seeds=selected, checkpoint=args.checkpoint)
        return
    if args.checkpoint != "all":
        raise SystemExit("--checkpoint is available only with --analysis.")

    if args.multi_seed or args.model_seed is not None:
        if args.resume is not None and args.model_seed is None:
            raise SystemExit("For an explicit Phase 2 --resume checkpoint, also pass --model-seed.")
        if args.seed is not None:
            raise SystemExit("Use --model-seed for Phase 2; --seed is reserved for the single-seed baseline.")
        if args.results_dir or args.checkpoints_dir:
            raise SystemExit("Use --results-root and --checkpoints-root for Phase 2 runs.")
        from experiments.multi_seed import run_multi_seed

        if args.model_seeds is not None:
            config["multi_seed"]["model_seeds"] = args.model_seeds
        if args.max_steps is not None:
            config["training"]["max_steps"] = args.max_steps
        if args.results_root is not None:
            config["experiment"]["output_root"] = str(args.results_root)
        if args.checkpoints_root is not None:
            config["experiment"]["checkpoint_root"] = str(args.checkpoints_root)
        if args.model_seed is not None and args.model_seed not in config["multi_seed"]["model_seeds"]:
            raise SystemExit(f"Model seed {args.model_seed} is not listed in {args.config}'s multi_seed.model_seeds.")
        run_multi_seed(config, only_seed=args.model_seed, resume=args.resume, force=args.force)
        return

    if args.model_seeds is not None or args.results_root or args.checkpoints_root:
        raise SystemExit("--model-seeds/--results-root/--checkpoints-root require --multi-seed or --model-seed.")

    model_seed = config["experiment"]["seed"] if args.seed is None else args.seed
    if model_seed < 0:
        raise SystemExit("--seed must be a nonnegative integer")
    config["experiment"]["seed"] = model_seed
    # In a single-seed baseline the convenient default is to seed both the
    # split and initialization together. An explicit --data-seed decouples them.
    config["data"]["data_seed"] = (
        args.data_seed if args.data_seed is not None
        else model_seed if args.seed is not None
        else config["data"]["data_seed"]
    )
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    validate(config)

    result_dir = args.results_dir or Path(config["experiment"]["output_root"]) / f"seed_{model_seed}"
    checkpoint_dir = args.checkpoints_dir or Path(config["experiment"]["checkpoint_root"]) / f"seed_{model_seed}"
    if not args.resume and not args.force:
        existing = [result_dir / name for name in ("metrics.csv", "summary.json", "config.yaml") if (result_dir / name).exists()]
        existing += [checkpoint_dir / "initial.pt"] if (checkpoint_dir / "initial.pt").exists() else []
        if existing:
            listed = ", ".join(str(path) for path in existing)
            raise SystemExit(f"Run artifacts already exist ({listed}). Use --resume with a checkpoint or choose new output paths.")

    accelerator = device()
    print(f"Device: {accelerator.type.upper()}", flush=True)
    trainer = Trainer(config, model_seed, result_dir, checkpoint_dir)
    summary = trainer.run(resume=args.resume)
    print(f"Run {summary['status']}: {result_dir}", flush=True)
    if summary["grokking_detected"]:
        print(f"Confirmed grokking at step {summary['confirmed_grokking_step']}.", flush=True)
    else:
        print("Grokking was not confirmed within this run's training horizon.", flush=True)


if __name__ == "__main__":
    main()
