"""Sequential, resumable Phase 2 model-seed experiment runner."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from analysis.seed_variance import aggregate
from data.modular_arithmetic import make_dataset
from training.config import config_hash, validate
from training.trainer import Trainer
from training.utils import cleanup, device, environment, json_save


def _manifest(config: dict, dataset, seeds: list[int]) -> dict:
    return {
        "experiment_name": config["experiment"]["name"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_config_hash": config_hash(config),
        "environment": environment(),
        "device": str(device()),
        "data_seed": config["data"]["data_seed"],
        "model_seeds": seeds,
        "dataset_split_hash": dataset.split_hash,
        "train_split_hash": dataset.train_hash,
        "test_split_hash": dataset.test_hash,
        "prime": config["data"]["prime"],
        "operation": config["data"]["operation"],
        "train_fraction": config["data"]["train_fraction"],
        "model_configuration": config["model"],
        "optimizer_configuration": {
            key: config["training"][key]
            for key in ("learning_rate", "weight_decay", "beta1", "beta2")
        },
        "training_configuration": config["training"],
        "memorization_criteria": config["memorization"],
        "grokking_criteria": config["grokking"],
    }


def run_multi_seed(config: dict, only_seed: int | None = None, resume: Path | None = None, force: bool = False) -> dict | None:
    """Run configured seeds one at a time, skipping completed runs and resuming interrupted ones."""
    validate(config)
    seeds = [only_seed] if only_seed is not None else list(config["multi_seed"]["model_seeds"])
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Model seeds must be unique nonnegative integers")

    result_root = Path(config["experiment"]["output_root"])
    checkpoint_root = Path(config["experiment"]["checkpoint_root"])
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(config["data"])
    manifest_path = result_root / "experiment_manifest.json"
    proposed_manifest = _manifest(config, dataset, list(config["multi_seed"]["model_seeds"]))
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("experiment_config_hash", "dataset_split_hash", "train_split_hash", "test_split_hash"):
            if previous.get(key) != proposed_manifest[key]:
                raise ValueError(f"Existing multi-seed manifest differs in {key}; use a separate output root.")
    else:
        json_save(proposed_manifest, manifest_path)

    print(
        f"PHASE 2 MULTI-SEED | seeds={seeds} | fixed data_seed={config['data']['data_seed']} "
        f"| split={dataset.split_hash[:16]} | device={device()} | sequential execution",
        flush=True,
    )

    for seed in seeds:
        result_dir = result_root / f"seed_{seed}"
        checkpoint_dir = checkpoint_root / f"seed_{seed}"
        summary_path = result_dir / "summary.json"
        explicit_resume = resume is not None
        checkpoint_path = Path(resume) if resume is not None else checkpoint_dir / "latest.pt"

        if force and resume is None:
            shutil.rmtree(result_dir, ignore_errors=True)
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        elif summary_path.exists() and not explicit_resume:
            summary = json.loads(summary_path.read_text())
            if summary.get("status") == "completed":
                print(f"Skipping completed model seed {seed}.", flush=True)
                continue

        should_resume = explicit_resume or checkpoint_path.exists()
        if explicit_resume and seed != only_seed:
            raise ValueError("An explicit checkpoint can only be used with its matching --model-seed.")
        if summary_path.exists() and not should_resume and not force:
            raise FileNotFoundError(f"Seed {seed} has a summary but no resumable latest checkpoint: {checkpoint_path}")

        trainer = Trainer(config, seed, result_dir, checkpoint_dir)
        summary = trainer.run(resume=checkpoint_path if should_resume else None)
        if summary["dataset_split_hash"] != dataset.split_hash:
            raise RuntimeError(f"Dataset split changed for model seed {seed}; stopping the study.")
        print(
            f"Seed {seed} finished with status={summary['status']}; "
            f"initial_model_hash={summary['initial_model_hash'][:16]} "
            f"elapsed={summary['total_training_time_seconds']:.2f}s",
            flush=True,
        )
        del trainer
        cleanup()
        if explicit_resume:
            resume = None

    # Aggregate all summaries currently available, preserving any incomplete runs.
    result = aggregate(result_root)
    if result is not None:
        completed_summaries = [
            json.loads(path.read_text())
            for path in sorted(result_root.glob("seed_*/summary.json"))
        ]
        initial_hashes = {row["initial_model_hash"] for row in completed_summaries}
        if len(completed_summaries) > 1 and len(initial_hashes) != len(completed_summaries):
            raise RuntimeError("Different model seeds produced identical initial model hashes.")
        manifest = json.loads(manifest_path.read_text())
        manifest["initial_model_hashes"] = {
            str(row["model_seed"]): row["initial_model_hash"] for row in completed_summaries
        }
        manifest["initialization_hashes_unique"] = len(initial_hashes) == len(completed_summaries)
        json_save(manifest, manifest_path)
        print(f"\nSeed summary: {result_root / 'seed_summary.csv'}", flush=True)
        print((result_root / "seed_summary.csv").read_text(), flush=True)
        print(
            f"Aggregate: {result['number_grokked']}/{result['number_completed']} completed seeds grokked; "
            f"fixed split hash={result['dataset_split_hash'][:16]}",
            flush=True,
        )
    return result
