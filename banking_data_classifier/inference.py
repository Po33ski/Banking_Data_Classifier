from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def inference(
    texts: List[str],
    model_dir: str | Path,
    device: str = "cuda",
    label_lookup: Mapping[int, str] | None = None,
) -> List[Tuple[int, str, float]]:
    """
    Run inference on a list of input texts using a saved HF classification model.

    Returns a list of (class_id, class_name, probability) tuples for each text.
    The class_name is resolved via the provided label_lookup (built from the
    dataframe's real_label column); if unavailable, the numeric class_id is used.
    """
    if not texts:
        return []

    model_dir = Path(model_dir)
    device_resolved = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device_resolved)
    model.eval()

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device_resolved) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

    results: List[Tuple[int, str, float]] = []
    for row in probs:
        class_id = int(np.argmax(row))
        prob = float(row[class_id])
        class_name = str(class_id)
        if label_lookup is not None:
            class_name = label_lookup.get(class_id, class_name)
        results.append((class_id, class_name, prob))

    return results


