from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

# DatasetConfig: Configuration for the dataset
class DatasetConfig(BaseModel):
    # Hugging Face dataset identifier, e.g. "PolyAI/banking77"
    dataset_name: str = Field(default="PolyAI/banking77")
    random_state: int = Field(default=0)
    # concatenate all samples into a single dataframe
    concatenate_all_samples: bool = Field(default=False) #if here is True then the clean_test and quality_test in CleaningConfig and QualityConfig must be False

# CleaningConfig: Configuration for the cleaning like removing terms, casefolding, deduplication, etc.
class CleaningConfig(BaseModel):
    # clean_train: clean the train dataframe
    clean_train: bool = Field(default=True)
    # clean_test: clean the test dataframe
    clean_test: bool = Field(default=False)
    # drop_duplicates: drop duplicates
    drop_duplicates: bool = Field(default=True)
    # min_text_len: minimum text length
    min_text_len: int = Field(default=2, ge=0)
    # lowercase_text: lowercase the text
    lowercase_text: bool = Field(default=True)

# SplitConfig: Configuration for the split
class SplitConfig(BaseModel):
    # # are we using the concatenated all samples dataframe? it must have the same value like concatenate_all_samples in DatasetConfig
    # use_concatenated_all_samples: bool = Field(default=False) #if here is True then the clean_test and quality_test in CleaningConfig and QualityConfig must be False
    train_frac: float = Field(default=0.8, ge=0.0, le=1.0)
    valid_frac: float = Field(default=0.2, ge=0.0, le=1.0)
    test_frac: float = Field(default=0.0, ge=0.0, le=1.0) # set more than 0 only if concatenate_all_samples is True
    random_state: int = Field(default=0)
    stratify: bool = Field(default=True)
    # validate_sum: validate the sum of the split fractions
    def validate_sum(self) -> None:
        total = self.train_frac + self.valid_frac + self.test_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train_frac + valid_frac + test_frac must equal 1.0")

# TrainConfig: Configuration for the training like model name, learning rate, epochs, seed, use_cpu, etc.
class TrainConfig(BaseModel):
    # model_name: the name of the model
    model_name: str = Field(default="distilbert/distilbert-base-uncased")
    # learning_rate: the learning rate
    learning_rate: float = Field(default=1e-4, gt=0.0)
    # epochs: the number of epochs
    epochs: int = Field(default=5, ge=1)
    # seed: the seed for the random number generator
    seed: int = Field(default=0)
    # device: the device to use for the training
    device: str = Field(default="cuda") # "cuda" or "cpu"

# QualityConfig: Configuration for the quality like embedding model name, cv folds, logistic C, etc.
class QualityConfig(BaseModel):
    # quality_train: quality the train dataframe
    quality_train: bool = Field(default=True)
    # quality_test: quality the test dataframe
    quality_test: bool = Field(default=False)
    # embedding_model_name: the name of the embedding model
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    # quality_iterations: the number of iterations to run the quality analysis
    quality_iterations: int = Field(default=3, ge=1)
    # cv_folds: the number of cross-validation folds
    cv_folds: int = Field(default=5, ge=2)
    # regularization_c: the regularization parameter
    regularization_c: float = Field(default=0.1, gt=0.0)
    # device: the device to use for the embedding model
    embedding_device: str = Field(default="cuda")
    # label_issue_threshold: the threshold for the label issue
    label_issue_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # max_label_fixes: the maximum number of label fixes
    max_label_fixes: Optional[int] = Field(default=None, ge=1)
    
# PathsConfig: Configuration for the paths like artifacts dir, etc.
class PathsConfig(BaseModel):
    artifacts_dir: Path = Field(default=Path("artifacts"))

# ProjectConfig: Configuration for the project like dataset, cleaning, split, train, quality, paths, etc.
class ProjectConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    # load: load the configuration from a JSON file or return the default configuration
    @staticmethod
    def load(config_path: Optional[str]) -> "ProjectConfig":
        if config_path is None:
            cfg = ProjectConfig()
            cfg.split.validate_sum()
            return cfg
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file must be JSON; parse error: {e}") from e
        cfg = ProjectConfig.model_validate(data)
        cfg.split.validate_sum()
        return cfg


