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
    if "text" not in df_raw_train.columns:
        raise KeyError(f"[ingest] text not found in dataset columns: {list(df_raw_train.columns)}")
    if "label" not in df_raw_train.columns:
        raise KeyError(f"[ingest] label not found in dataset columns: {list(df_raw_train.columns)}")
    if "text" not in df_raw_test.columns:
        raise KeyError(f"[ingest] text not found in dataset columns: {list(df_raw_test.columns)}")
    if "label" not in df_raw_test.columns:
        raise KeyError(f"[ingest] label not found in dataset columns: {list(df_raw_test.columns)}")

    df_train = df_raw_train[["text", "label"]].copy()
    df_train["text"] = df_train["text"].astype(str).str.strip()
    df_test = df_raw_test[["text", "label"]].copy()
    df_test["text"] = df_test["text"].astype(str).str.strip()

    print(f"[ingest] Loaded rows: {len(df_train)} (columns: text, label)")
    return df_train, df_test


 