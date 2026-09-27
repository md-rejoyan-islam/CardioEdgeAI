"""Shared evaluation: classification report, confusion matrix, results log.

Every model evaluated on DS2 writes its metrics to docs/results.json so
the thesis comparison tables stay reproducible.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (classification_report,  # noqa: E402
                             confusion_matrix, f1_score, accuracy_score)

sys.path.append(str(Path(__file__).resolve().parents[2]))

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
FIG_DIR = DOCS_DIR / "figures"
RESULTS_JSON = DOCS_DIR / "results.json"


def evaluate(model_name: str, y_true, y_pred, classes=("N", "S", "V")) -> dict:
    """Print report, save confusion-matrix figure, append to results.json."""
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    print(f"\n===== {model_name} (DS2, inter-patient) =====")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(model_name)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_name.lower().replace(" ", "_").replace("/", "-")
    fig.savefig(FIG_DIR / f"{slug}_confusion.png", dpi=150)
    plt.close(fig)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "per_class": classification_report(y_true, y_pred,
                                           output_dict=True,
                                           zero_division=0),
        "n_test": int(len(y_true)),
    }
    results = {}
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    results[model_name] = metrics
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return metrics


def load_processed_dataset():
    """Return the Phase 2 dataset split into DS1 / DS2."""
    data = np.load(
        Path(__file__).resolve().parents[2] / "data" / "processed"
        / "mitbih_nsv.npz", allow_pickle=True)
    return (data["X"], data["y"], data["r"], data["record"],
            data["train_mask"], data["test_mask"])
