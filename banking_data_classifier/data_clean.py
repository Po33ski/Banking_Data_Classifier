from __future__ import annotations

import re
from typing import Tuple

import pandas as pd
import typer
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

    # typer.echo("[clean] Label distribution:")
    # counts = df["label"].value_counts()
    # counts = counts.sort_values(ascending=False).head(77)
    # for i, c in counts.items(): typer.echo(f"{i}: {c}")

    # counts_to_cut_off = counts[counts < cfg.cut_off_labels_threshold]
    # if cfg.cut_off_labels:
    #     df = df[~df["label"].isin(counts_to_cut_off.index)]
    #     df = df.reset_index(drop=True)
    #     typer.echo(f"[clean] Cut off labels: {counts_to_cut_off.index}")
    #     typer.echo(f"[clean] Final rows: {len(df)}")

    # typer.echo("[clean] Label distribution after cutting off labels:")
    # counts = df["label"].value_counts()
    # counts = counts.sort_values(ascending=False).head(77)
    # for i, c in counts.items(): typer.echo(f"{i}: {c}")

    print(f"[clean] Final rows: {len(df)}")
    return df


