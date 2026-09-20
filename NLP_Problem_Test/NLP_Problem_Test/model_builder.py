"""
Contains PyTorch model code to instantiate a BERT-based multilingual
politeness classifier.

Backbone : Google Multilingual BERT (bert-base-multilingual-cased)
Task     : 4-class politeness classification
           {plain, teineigo, sonkeigo, kenjogo}
"""
import torch
from torch import nn

from transformers import AutoModel


class BERTPolitenessClassifier(nn.Module):
    """Multilingual BERT backbone + classification head.

    By default only the LAST `1 - freeze_ratio` of BERT encoder layers and
    the classification head are fine-tuned; embeddings and the bottom layers
    stay frozen to speed up training and reduce overfitting on small data.

    Args:
        model_name: Hugging Face model identifier (e.g.
            "google-bert/bert-base-multilingual-cased").
        num_labels: Number of output classes.
        freeze_ratio: Fraction of BERT encoder layers to freeze from the
            bottom (0.0 = full fine-tuning, 0.8 = tune last 20%, default).
        dropout: Dropout probability applied before the classification head.

    Example usage:
        model = BERTPolitenessClassifier(
            model_name="google-bert/bert-base-multilingual-cased",
            num_labels=4,
            freeze_ratio=0.8,
        ).to(device)
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        freeze_ratio: float = 0.8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=self.bert.config.hidden_size,
                out_features=num_labels,
            ),
        )

        if freeze_ratio > 0.0:
            self._freeze_bottom_layers(freeze_ratio)

    def _freeze_bottom_layers(self, freeze_ratio: float) -> None:
        """Freezes embeddings + the bottom `freeze_ratio` of encoder layers.

        Keeps the last `1 - freeze_ratio` of encoder layers and the
        classification head trainable.

        NOTE: To FULL fine-tune the whole BERT backbone instead, construct
        the model with `freeze_ratio=0.0` (i.e. uncomment and switch to):
            model = BERTPolitenessClassifier(
                model_name=MODEL_NAME,
                num_labels=len(class_names),
                freeze_ratio=0.0,
            )
        """
        num_layers = len(self.bert.encoder.layer)

        # Freeze token/position/segment embeddings.
        self.bert.embeddings.requires_grad_(False)

        # Freeze the bottom (freeze_ratio) encoder layers.
        freeze_upto = int(num_layers * freeze_ratio)
        for layer_idx in range(freeze_upto):
            self.bert.encoder.layer[layer_idx].requires_grad_(False)

        print(
            f"[INFO] Freezing embeddings + bottom {freeze_upto}/{num_layers} "
            f"BERT encoder layers; tuning last {num_layers - freeze_upto} "
            f"layers + classifier head."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes raw class logits.

        Args:
            input_ids: Token ids, shape (batch_size, max_length).
            attention_mask: 1 for real tokens, 0 for padding,
                shape (batch_size, max_length).

        Returns:
            Logits of shape (batch_size, num_labels).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Mask-aware mean pooling over the last hidden states so padding
        # tokens do not contribute to the sentence representation.
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = last_hidden * mask
        pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return self.classifier(pooled)