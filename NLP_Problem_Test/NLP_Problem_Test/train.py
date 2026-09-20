"""
Trains a BERT-based multilingual politeness classifier using
device-agnostic PyTorch code.

Pipeline:
    1. Create BERT tokenizer + DataLoaders      (data_setup.py)
    2. Build BERT + classification head          (model_builder.py)
    3. train_step / test_step per epoch          (engine.py)
    4. Save the trained model                    (utils.py)
"""
from pathlib import Path

import torch

from transformers import AutoTokenizer

import data_setup
import engine
import model_builder
import utils

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128

# Fraction of BERT encoder layers to freeze from the bottom (0.8 -> tune
# only the LAST 20% of layers + classifier head).
# NOTE: switch to 0.0 for FULL BERT fine-tuning.
FREEZE_RATIO = 0.8

# Model to use for tokenization + backbone
MODEL_NAME = "google-bert/bert-base-multilingual-cased"

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "Japanese-Politeness"
MODEL_SAVE_DIR = PROJECT_DIR / "models"

# ---------------------------------------------------------------------------
# Device-agnostic setup
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ---------------------------------------------------------------------------
# 1. Tokenizer + DataLoaders
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataloader, val_dataloader, test_dataloader, class_names = \
    data_setup.create_dataloaders(
        train_csv=DATA_DIR / "train" / "train.csv",
        val_csv=DATA_DIR / "val" / "val.csv",
        test_csv=DATA_DIR / "test" / "test.csv",
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )

print(
    f"[INFO] Classes: {class_names}\n"
    f"[INFO] Train batches: {len(train_dataloader)} | "
    f"Val batches: {len(val_dataloader)} | "
    f"Test batches: {len(test_dataloader)}"
)

# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
model = model_builder.BERTPolitenessClassifier(
    model_name=MODEL_NAME,
    num_labels=len(class_names),
    freeze_ratio=FREEZE_RATIO,
).to(device)

# ---------------------------------------------------------------------------
# 3. Loss + Optimizer
# ---------------------------------------------------------------------------
loss_fn = torch.nn.CrossEntropyLoss()

# Only parameters that require gradients get updated (frozen BERT layers
# have requires_grad=False) but passing the full parameter list is harmless.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ---------------------------------------------------------------------------
# 4. Train & validate
# ---------------------------------------------------------------------------
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=val_dataloader,  # validation split for epoch metrics
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
)

# ---------------------------------------------------------------------------
# 5. Final evaluation on the held-out test split
# ---------------------------------------------------------------------------
test_loss, test_acc = engine.test_step(
    model=model,
    dataloader=test_dataloader,
    loss_fn=loss_fn,
    device=device,
)
print(f"[INFO] Final test loss: {test_loss:.4f} | test acc: {test_acc:.4f}")

# ---------------------------------------------------------------------------
# 6. Save the model
# ---------------------------------------------------------------------------
utils.save_model(
    model=model,
    target_dir=str(MODEL_SAVE_DIR),
    model_name="bert_multilingual_politeness_classifier.pth",
)