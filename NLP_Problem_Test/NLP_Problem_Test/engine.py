"""
Contains functions for training and testing a PyTorch BERT
politeness classifier.
"""
from typing import Dict, List, Tuple

import torch

from tqdm.auto import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance the model trains on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of (train_loss, train_accuracy).
    """
    model.train()

    train_loss, train_acc = 0, 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 1. Forward pass
        y_pred_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred_logits, labels)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = y_pred_logits.argmax(dim=1)
        train_acc += (y_pred_class == labels).sum().item() / len(labels)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance the model is tested on.
        loss_fn: A PyTorch loss function to calculate loss on the data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of (test_loss, test_accuracy).
    """
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 1. Forward pass
            test_pred_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, labels)
            test_loss += loss.item()

            # 3. Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == labels).sum().item() / len(labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch model through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on
            (here: the validation split).
        optimizer: A PyTorch optimizer to help minimize the loss.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing (validation) loss as well as
        accuracy metrics, one value per epoch. In the form:
            {train_loss: [...], train_acc: [...],
             test_loss: [...], test_acc: [...]}
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {test_loss:.4f} | "
            f"val_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results