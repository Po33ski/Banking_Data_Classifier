from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

from .config import CleaningConfig


def clean_dataframe(df_raw: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """
    Apply simple cleaning rules to a dataframe and return a cleaned dataframe.
    """
    df = df_raw.copy()
    df["text"] = df["text"].astype(str).str.strip()
    # Drop very short texts
    if cfg.min_text_len > 0:
        df = df[df["text"].str.len() >= cfg.min_text_len]
        df = df.reset_index(drop=True)
    
    # Optional lowercasing
    if cfg.lowercase_text:
        df["text"] = df["text"].str.lower()
        df = df.reset_index(drop=True)

    # Case-insensitive deduplication if enabled
    if cfg.drop_duplicates:
        df = df.drop_duplicates(subset="text")
        df = df.reset_index(drop=True)
    df = df.reset_index(drop=True)

    print(f"[clean] Final rows: {len(df)}")
    return df


