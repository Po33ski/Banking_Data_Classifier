from __future__ import annotations

from typing import Any, Dict, Mapping

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments


def evaluate_model(model_dir: str, tokenized_dir: str, len_test_df: int, label_lookup: Mapping[int, str] | None = None) -> Dict[str, Any]:
    """
    Evaluate saved HF model on test dataframe and compute metrics.
    """
    print(f"[eval] Evaluating model at '{model_dir}' on {len_test_df} rows")

    # Load the model and the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    datasets_tokenized = DatasetDict.load_from_disk(tokenized_dir)
    # Load the training arguments
    training_args = TrainingArguments(output_dir=model_dir, do_train=False, do_eval=True, per_device_eval_batch_size=32)
    # Create the trainer
    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer, eval_dataset=datasets_tokenized["test"])

    # Predict the test set
    pred_output = trainer.predict(datasets_tokenized["test"])
    # Get the logits and the predictions
    logits = torch.from_numpy(pred_output.predictions)
    probs = softmax(logits, dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    y_test = np.array(datasets_tokenized["test"]["label"])

    # Compute metrics for multi-class classification: accuracy, macro F1 and macro AUROC (one-vs-one)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "auroc_ovo_macro": float(roc_auc_score(y_test, probs, multi_class="ovo", average="macro")),
        # store per-class F1 as a dict {class_id: score}
        "f1_per_class": {int(i): float(s) for i, s in enumerate(f1_per_class)},
    }
    print(
        "[eval] "
        + ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if k != "f1_per_class"
        )
    )

    # Plot F1-score for each class and save as PNG next to the model directory
    out_dir = Path(model_dir).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    class_ids = np.arange(len(f1_per_class))
    ax.bar(class_ids, f1_per_class)
    ax.set_xlabel("Class id")
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score per class")
    ax.set_xticks(class_ids)
    ax.set_xticklabels(class_ids, rotation=90, fontsize=6)
    plt.tight_layout()
    fig_path = out_dir / "f1_per_class.png"
    fig.savefig(fig_path)
    plt.close(fig)

    if label_lookup:
        # Save a simple text cheat sheet mapping class ids to human-readable names
        cheat_lines = ["Class id to name mapping (from real_label):"]
        for cid in sorted(label_lookup):
            cheat_lines.append(f"{cid}: {label_lookup[cid]}")
        cheat_text = "\n".join(cheat_lines)
        (out_dir / "class_id_to_name.txt").write_text(cheat_text, encoding="utf-8")

    return metrics


