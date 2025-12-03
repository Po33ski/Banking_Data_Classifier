from __future__ import annotations

from typing import Tuple

import pandas as pd
from datasets import load_dataset

from .config import DatasetConfig


def load_raw_dataset(cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw text classification dataset from HuggingFace and return a dataframe
    with unified columns: text, label.

    This is adapted to multi-class datasets like PolyAI/banking77 where the raw
    dataset already contains a single text column and an integer label column.
    """
    print(f"[ingest] Loading dataset '{cfg.dataset_name}'")
    hf_ds = load_dataset(cfg.dataset_name)
    df_raw_train = hf_ds["train"].to_pandas()
    df_raw_test = hf_ds["test"].to_pandas()

    # Check if the columns exist
    if cfg.text_col not in df_raw_train.columns:
        raise KeyError(f"[ingest] text_col='{cfg.text_col}' not found in dataset columns: {list(df_raw_train.columns)}")
    if cfg.label_col not in df_raw_train.columns:
        raise KeyError(f"[ingest] label_col='{cfg.label_col}' not found in dataset columns: {list(df_raw_train.columns)}")
    if cfg.text_col not in df_raw_test.columns:
        raise KeyError(f"[ingest] text_col='{cfg.text_col}' not found in dataset columns: {list(df_raw_test.columns)}")
    if cfg.label_col not in df_raw_test.columns:
        raise KeyError(f"[ingest] label_col='{cfg.label_col}' not found in dataset columns: {list(df_raw_test.columns)}")

    df_train = df_raw_train[[cfg.text_col, cfg.label_col]].copy()
    df_train[cfg.text_col] = df_train[cfg.text_col].astype(str).str.strip()
    df_test = df_raw_test[[cfg.text_col, cfg.label_col]].copy()
    df_test[cfg.text_col] = df_test[cfg.text_col].astype(str).str.strip()

    # Normalize column names expected by the rest of the pipeline
    df_train = df_train.rename(columns={cfg.text_col: "text", cfg.label_col: "label"})
    df_test = df_test.rename(columns={cfg.text_col: "text", cfg.label_col: "label"})
    print(f"[ingest] Loaded rows: {len(df_train)} (columns: text, label)")
    return df_train, df_test


 