"""
Contains functionality for creating PyTorch DataLoaders for
NLP politeness classification data.

Each CSV has two columns:
    sentence               -- Japanese sentence
    sentence_politeness    -- one of {plain, teineigo, sonkeigo, kenjogo}

Data is tokenized with a Hugging Face BERT tokenizer (padding + truncation)
inside a custom torch Dataset, then wrapped into DataLoaders.
"""
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

NUM_WORKERS = 0


class PolitenessDataset(Dataset):
    """Turns a list of sentences + labels into a tokenized torch Dataset.

    Args:
        texts: List of raw sentence strings.
        labels: List of integer class indices aligned with `texts`.
        tokenizer: Hugging Face tokenizer used to tokenize each sentence.
        max_length: Maximum number of tokens per sequence.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int,
    ) -> None:
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        return {
            "input_ids": torch.tensor(
                encoding["input_ids"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def read_csv(csv_path: Union[str, Path]) -> Tuple[List[str], List[int], List[str]]:
    """
    Reads a politeness CSV and returns its sentences, integer labels and
    the label -> class name mapping.

    Args:
        csv_path: Path to a CSV with `sentence` and `sentence_politeness` columns.

    Returns:
        A tuple (texts, labels, class_names).
    """
    df = pd.read_csv(csv_path)

    texts = df["sentence"].astype(str).tolist()
    class_names = sorted(df["sentence_politeness"].unique().tolist())
    label_to_id = {label: i for i, label in enumerate(class_names)}
    labels = df["sentence_politeness"].map(label_to_id).tolist()

    return texts, labels, class_names


def create_dataloaders(
    train_csv: Union[str, Path],
    val_csv: Union[str, Path],
    test_csv: Union[str, Path],
    tokenizer,
    max_length: int,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Creates training, validation and testing DataLoaders.

    Args:
        train_csv: Path to the training CSV.
        val_csv: Path to the validation CSV.
        test_csv: Path to the testing CSV.
        tokenizer: A Hugging Face tokenizer used to tokenize the text.
        max_length: Maximum sequence length in tokens for the BERT tokenizer.
        batch_size: Number of samples per batch in each DataLoader.
        num_workers: Number of worker processes per DataLoader.

    Returns:
        A tuple of (train_dataloader, val_dataloader, test_dataloader,
        class_names).
        class_names is a list of the target classes.
    """
    # Build a single label -> id mapping from the train split so that all
    # splits agree on the same integer encoding.
    train_texts, _, train_class_names = read_csv(train_csv)
    train_class_names = sorted(train_class_names)
    label_to_id = {label: i for i, label in enumerate(train_class_names)}

    def to_dataset(csv_path: Union[str, Path]) -> PolitenessDataset:
        df = pd.read_csv(csv_path)
        texts = df["sentence"].astype(str).tolist()
        labels = df["sentence_politeness"].map(label_to_id).tolist()

        return PolitenessDataset(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    train_data = to_dataset(train_csv)
    val_data = to_dataset(val_csv)
    test_data = to_dataset(test_csv)

    # Sanity check: every split only contains known classes.
    for split_name, split_data in (
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ):
        assert all(
            0 <= label < len(train_class_names) for label in split_data.labels
        ), f"[ERROR] {split_name} split contains an unknown class label."

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, test_dataloader, train_class_names