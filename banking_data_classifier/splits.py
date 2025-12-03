from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SplitConfig


def split_dataframe(
    df: pd.DataFrame,
    cfg: SplitConfig,
    concatenate_all_samples: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Deterministic split into train/valid/test according to fractions in cfg.

    - If concatenate_all_samples=True  -> split df into 3 parts: train, valid, test
      using (train_frac, valid_frac, test_frac).
    - If concatenate_all_samples=False -> split df only into 2 parts: train, valid,
      ignoring test_frac; the test return value is None.
    """
    cfg.validate_sum()
    stratify_labels = df["label"] if cfg.stratify else None

    if concatenate_all_samples:
        # 3-way split: train/valid/test
        test_size = cfg.test_frac
        train_valid, test = train_test_split(
            df,
            test_size=test_size,
            random_state=cfg.random_state,
            stratify=stratify_labels,
        )
        # recompute valid frac relative to remaining (train + valid)
        valid_rel = cfg.valid_frac / (cfg.train_frac + cfg.valid_frac)
        stratify_labels_tv = train_valid["label"] if cfg.stratify else None
        train, valid = train_test_split(
            train_valid,
            test_size=valid_rel,
            random_state=cfg.random_state,
            stratify=stratify_labels_tv,
        )
    else:
        # 2-way split: train/valid only, no separate test split
        tv_total = cfg.train_frac + cfg.valid_frac
        valid_rel = cfg.valid_frac / tv_total
        train, valid = train_test_split(
            df,
            test_size=valid_rel,
            random_state=cfg.random_state,
            stratify=stratify_labels,
        )
        test = None

    test_len = len(test) if test is not None else "test from downloaded dataset" # if test is None then it is the test from the downloaded dataset
    print(f"[split] train={len(train)} valid={len(valid)} test={test_len}")
    return train.reset_index(drop=True), valid.reset_index(drop=True), None if test is None else test.reset_index(drop=True)


