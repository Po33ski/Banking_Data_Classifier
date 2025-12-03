from __future__ import annotations

import shutil
from sklearn.utils import resample
import pandas as pd
import typer

from .config import ProjectConfig
from .data_ingest import load_raw_dataset
from .data_clean import clean_dataframe
from .quality import run_cleanlab
from .splits import split_dataframe
from .train_hf import train_hf
from .evaluate import evaluate_model
from .behavioral import run_giskard_scan
from .explain import explain_samples






# ensure_dirs: ensure the directories exist for the artifacts (splits, models, etc.)
def ensure_dirs(cfg: ProjectConfig) -> None:
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

# run_load: load the raw dataset and save it to a parquet files for train and test
def run_load(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    df_raw_train, df_raw_test = load_raw_dataset(cfg.dataset)
    if cfg.dataset.concatenate_all_samples:
        df_raw_train = pd.concat([df_raw_train, df_raw_test])
        out_raw_train = cfg.paths.artifacts_dir / "raw_preview_train.parquet" # it is called train for consistency with the rest of the code
        df_raw_train.to_parquet(out_raw_train, index=False)
        typer.echo(f"[load] Preview saved to: {out_raw_train}")
    else:
        out_raw_train = cfg.paths.artifacts_dir / "raw_preview_train.parquet"
        df_raw_train.to_parquet(out_raw_train, index=False)
        typer.echo(f"[load] Preview saved to: {out_raw_train}")
        out_raw_test = cfg.paths.artifacts_dir / "raw_preview_test.parquet"
        df_raw_test.to_parquet(out_raw_test, index=False)
        typer.echo(f"[load] Preview saved to: {out_raw_test}")

# run_clean: clean the raw dataset and save it to a parquet file
def run_clean(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    if cfg.cleaning.clean_train:
        in_raw_train = cfg.paths.artifacts_dir / "raw_preview_train.parquet"
        if not in_raw_train.exists():
            typer.echo("[clean] 'raw_preview_train.parquet' not found. Run 'uv run load' first or check the config of load step.")
            raise typer.Exit(code=1)
        df_raw_train = pd.read_parquet(in_raw_train)
        df_clean_train = clean_dataframe(df_raw_train, cfg.cleaning)
        out_clean_train = cfg.paths.artifacts_dir / "clean_train.parquet"
        df_clean_train.to_parquet(out_clean_train, index=False)
        typer.echo(f"[clean] Cleaned data saved to: {out_clean_train}")
    if cfg.cleaning.clean_test:
        in_raw_test = cfg.paths.artifacts_dir / "raw_preview_test.parquet"
        if not in_raw_test.exists():
            typer.echo("[clean] 'raw_preview_test.parquet' not found. Run 'uv run load' first or check the config of load step.")
            raise typer.Exit(code=1)
        df_raw_test = pd.read_parquet(in_raw_test)
        df_clean_test = clean_dataframe(df_raw_test, cfg.cleaning)
        out_clean_test = cfg.paths.artifacts_dir / "clean_test.parquet"
        df_clean_test.to_parquet(out_clean_test, index=False)
        typer.echo(f"[clean] Cleaned data saved to: {out_clean_test}")

# run_quality: quality and fix the train and test dataframes
def run_quality(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    if cfg.quality.quality_train:   # quality the train dataframe
        in_path_train = cfg.paths.artifacts_dir / "clean_train.parquet"
        if not in_path_train.exists():
            typer.echo("[quality] 'clean_train.parquet' not found. Run 'uv run clean' first or check the config of load step.")
            raise typer.Exit(code=1)
        df_clean_train = pd.read_parquet(in_path_train)
        typer.echo("[quality] Running CleanLab for the train dataframe")
        for i in range(cfg.quality.quality_iterations):
            typer.echo(f"[quality] Running CleanLab iteration {i+1} for the train dataframe")
            df_clean_train = run_cleanlab(df_clean_train, cfg.quality)
        out_path_train = cfg.paths.artifacts_dir / f"quality_fixed_train.parquet"
        df_clean_train.to_parquet(out_path_train, index=False)
        typer.echo(f"[quality] Quality output saved to: {out_path_train}")
    if cfg.quality.quality_test:   # quality the test dataframe
        in_path_test = cfg.paths.artifacts_dir / "clean_test.parquet"
        if not in_path_test.exists():
            typer.echo("[quality] 'clean_test.parquet' not found. Run 'uv run clean' first or check the config of load step.")
            raise typer.Exit(code=1)
        df_clean_test = pd.read_parquet(in_path_test)
        typer.echo("[quality] Running CleanLab for the test dataframe")
        for i in range(cfg.quality.quality_iterations):
            typer.echo(f"[quality] Running CleanLab iteration {i+1} for the test dataframe")
            df_clean_test = run_cleanlab(df_clean_test, cfg.quality)
        out_path_test = cfg.paths.artifacts_dir / f"quality_fixed_test.parquet"
        df_clean_test.to_parquet(out_path_test, index=False)
        typer.echo(f"[quality] Quality output saved to: {out_path_test}")


def run_split(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    # check if the input data is clean or quality for the train dataframe
    if cfg.cleaning.clean_train and not cfg.quality.quality_train:
        in_path = cfg.paths.artifacts_dir / "clean_train.parquet"
    elif cfg.quality.quality_train:
        in_path = cfg.paths.artifacts_dir / "quality_fixed_train.parquet"
    else:
        in_path = cfg.paths.artifacts_dir / "raw_preview_train.parquet"
    if not in_path.exists():
        typer.echo("[split] No input data. Run 'uv run clean' (and optionally 'uv run quality') first or check the config of clean or quality step.")
        raise typer.Exit(code=1)
    df_fixed = pd.read_parquet(in_path)
    # split the dataframe into train, validation and test sets
    if cfg.dataset.concatenate_all_samples:
        # 3-way split produced from the (potentially concatenated) df_fixed
        train_df, valid_df, test_df = split_dataframe(df_fixed, cfg.split, cfg.dataset.concatenate_all_samples)
    else:
        # 2-way split (train/valid); test is provided separately from the original dataset
        train_df, valid_df, _ = split_dataframe(df_fixed, cfg.split, cfg.dataset.concatenate_all_samples)
        # if the test dataframe is not from the concatenated all samples dataframe then we need to check if the input data was processed in the previous steps
        if cfg.cleaning.clean_test and not cfg.quality.quality_test:
            in_path_test = cfg.paths.artifacts_dir / "clean_test.parquet"
        elif cfg.quality.quality_test:
            in_path_test = cfg.paths.artifacts_dir / "quality_fixed_test.parquet"
        else:
            in_path_test = cfg.paths.artifacts_dir / "raw_preview_test.parquet"
        if not in_path_test.exists():
            typer.echo("[split] 'clean_test.parquet' or 'quality_fixed_test.parquet' or 'raw_preview_test.parquet' not found. Run 'uv run clean' or 'uv run quality' or 'uv run load' first or check the config of clean, quality or load step.")
            raise typer.Exit(code=1)
        test_df = pd.read_parquet(in_path_test)
    # save the splits to the artifacts directory
    (cfg.paths.artifacts_dir / "splits").mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(cfg.paths.artifacts_dir / "splits/train.parquet", index=False)
    valid_df.to_parquet(cfg.paths.artifacts_dir / "splits/valid.parquet", index=False)
    test_df.to_parquet(cfg.paths.artifacts_dir / "splits/test.parquet", index=False)
    typer.echo("[split] Saved splits to artifacts/splits")


def run_train(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    splits_dir = cfg.paths.artifacts_dir / "splits"
    if not (splits_dir / "train.parquet").exists():
        typer.echo("[train] Splits not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    valid_df = pd.read_parquet(splits_dir / "valid.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    model_dir, _ = train_hf(train_df, valid_df, test_df, cfg.train, cfg.paths)
    typer.echo(f"[train] Model artifact at: {model_dir}")


def run_evaluate(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    splits_dir = cfg.paths.artifacts_dir / "splits"
    if not (splits_dir / "test.parquet").exists():
        typer.echo("[evaluate] Test split not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    tokenized_dir = cfg.paths.artifacts_dir / "tokenized"
    metrics = evaluate_model(model_dir, tokenized_dir, test_df)
    out = cfg.paths.artifacts_dir / "metrics.json"
    import json as _json

    out.write_text(_json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"[evaluate] Metrics saved to: {out}")


def run_explain(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    splits_dir = cfg.paths.artifacts_dir / "splits"
    if not (splits_dir / "test.parquet").exists():
        typer.echo("[explain] Test split not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    out = cfg.paths.artifacts_dir / "explain"
    explain_samples(model_dir, test_df, out)
    typer.echo(f"[explain] Saved attribution TSVs to: {out}")


def run_scan(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    test_path = cfg.paths.artifacts_dir / "splits/test.parquet"
    if not test_path.exists():
        typer.echo("[scan] Test split not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    test_df = pd.read_parquet(test_path)
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    out = cfg.paths.artifacts_dir / "giskard"
    run_giskard_scan(model_dir, test_df, device=getattr(cfg.quality, "device", "cpu"), out_dir=out)
    typer.echo(f"[scan] Giskard scan saved to: {out}")


def run_purge(cfg: ProjectConfig) -> None:
    artifacts_dir = cfg.paths.artifacts_dir
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[purge] Cleaned artifacts directory: {artifacts_dir}")