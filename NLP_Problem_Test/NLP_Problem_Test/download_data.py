"""
Downloads and preprocesses the Japanese honorifics dataset.

Dataset:
https://huggingface.co/datasets/ronantakizawa/japanese-honorifics

Each original entry contains:
  * base_sentence       -- casual/dictionary form (基本形)
  * teineigo            -- polite form (丁寧語)
  * sonkeigo            -- respectful form (尊敬語)
  * kenjogo             -- humble form (謙譲語)
  * english_translation -- English translation

The dataset is transformed into a multiclass classification dataset:

  base_sentence -> plain
  teineigo      -> teineigo
  sonkeigo      -> sonkeigo
  kenjogo       -> kenjogo

The original dataset is split into train/validation/test BEFORE
flattening to prevent data leakage between splits.

Output:
  data/
  ├── train/
  │   └── train.csv
  ├── val/
  │   └── val.csv
  └── test/
      └── test.csv
"""

import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_REPO_ID = "ronantakizawa/japanese-honorifics"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

RANDOM_STATE = 42

# Split ratios
TEST_SIZE = 0.15
VAL_SIZE = 0.15


# Mapping from dataset columns to classification labels
SENTENCE_TYPES = {
    "base_sentence": "plain",
    "teineigo": "teineigo",
    "sonkeigo": "sonkeigo",
    "kenjogo": "kenjogo",
}


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def flatten_dataset(dataset):
    """
    Converts the original honorific dataset into a multiclass
    classification DataFrame.

    Each original example produces four rows:

        base_sentence -> plain
        teineigo      -> teineigo
        sonkeigo      -> sonkeigo
        kenjogo       -> kenjogo

    Args:
        dataset: Hugging Face Dataset.

    Returns:
        pandas.DataFrame with columns:
            sentence
            sentence_politeness
    """

    rows = []

    for row in dataset:
        for column, label in SENTENCE_TYPES.items():
            rows.append(
                {
                    "sentence": row[column],
                    "sentence_politeness": label,
                }
            )

    return pd.DataFrame(rows)


def split_dataset(dataset):
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_dataset = dataset.select(train_idx)
    val_dataset = dataset.select(val_idx)
    test_dataset = dataset.select(test_idx)

    return train_dataset, val_dataset, test_dataset


def save_split(dataset, output_dir, filename):
    """
    Flattens a dataset split and saves it as CSV.

    Args:
        dataset: Hugging Face Dataset.
        output_dir: Directory where the CSV will be saved.
        filename: CSV filename.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    df = flatten_dataset(dataset)

    output_path = output_dir / filename

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"[INFO] Saved {len(df)} rows to {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """
    Downloads the dataset, splits it, flattens it into a multiclass
    classification dataset, and saves each split as CSV.
    """

    # Load HF_TOKEN from .env if available
    load_dotenv(PROJECT_DIR / ".env")

    token = os.getenv("HF_TOKEN") or None

    # -----------------------------------------------------------------------
    # 1. Download dataset
    # -----------------------------------------------------------------------

    print(
        f"[INFO] Downloading dataset "
        f"{DATASET_REPO_ID} from Hugging Face..."
    )

    dataset = load_dataset(
        DATASET_REPO_ID,
        token=token,
    )

    original_dataset = dataset["train"]

    print(
        f"[INFO] Original dataset contains "
        f"{len(original_dataset)} examples."
    )

    # -----------------------------------------------------------------------
    # 2. Split original dataset
    # -----------------------------------------------------------------------

    train_dataset, val_dataset, test_dataset = split_dataset(
        original_dataset
    )

    print(
        f"[INFO] Original split sizes:"
        f"\n       Train: {len(train_dataset)}"
        f"\n       Val:   {len(val_dataset)}"
        f"\n       Test:  {len(test_dataset)}"
    )

    # -----------------------------------------------------------------------
    # 3. Flatten and save as CSV
    # -----------------------------------------------------------------------

    save_split(
        train_dataset,
        DATA_DIR / "train",
        "train.csv",
    )

    save_split(
        val_dataset,
        DATA_DIR / "val",
        "val.csv",
    )

    save_split(
        test_dataset,
        DATA_DIR / "test",
        "test.csv",
    )

    print("[INFO] Dataset preprocessing completed.")


if __name__ == "__main__":
    main()