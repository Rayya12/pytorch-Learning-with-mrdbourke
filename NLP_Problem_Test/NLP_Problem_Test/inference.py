"""
Contains functions for evaluating Japanese text input and
classifying it into a politeness level using a fine-tuned
BERT multilingual classifier.
"""

import logging
import pathlib

import torch
from transformers import AutoTokenizer

import model_builder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODEL_PATH / "bert_multilingual_politeness_classifier.pth"
MODEL_NAME = "google-bert/bert-base-multilingual-cased"
MAX_LENGTH = 128

# Must match the label order used during training, which is the
# sorted order of the unique labels in the training split.
ID2LABEL = {
    0: "kenjogo",
    1: "plain",
    2: "sonkeigo",
    3: "teineigo",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loaded once and cached, instead of reloading on every call
_tokenizer: AutoTokenizer | None = None
_model: model_builder.BERTPolitenessClassifier | None = None


def _get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        logger.info("Loading tokenizer '%s'...", MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def _get_model() -> model_builder.BERTPolitenessClassifier:
    global _model
    if _model is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")

        logger.info("Loading model from %s...", MODEL_FILE)
        state_dict = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)

        model = model_builder.BERTPolitenessClassifier(
            model_name=MODEL_NAME,
            num_labels=len(ID2LABEL),
            freeze_ratio=0.0,
        )
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        _model = model
        logger.info("Model loaded successfully on device: %s", DEVICE)
    return _model


def inference_sentence(text: str) -> str | None:
    """
    Classifies the politeness level of a Japanese sentence.

    Args:
        text: Input Japanese sentence.

    Returns:
        Predicted politeness label ("plain", "teineigo", "sonkeigo",
        "kenjogo"), or None if inference failed.
    """
    if not text or not text.strip():
        logger.warning("Empty input text provided.")
        return None

    try:
        tokenizer = _get_tokenizer()
        model = _get_model()
    except Exception as e:
        logger.error(e)
        return None

    try:
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = model(
                input_ids=encoding["input_ids"].to(DEVICE),
                attention_mask=encoding["attention_mask"].to(DEVICE),
            )

        prediction_id = torch.argmax(outputs, dim=1).item()
        label = ID2LABEL.get(prediction_id, "unknown")

        logger.info("Prediction: %s (id=%d)", label, prediction_id)
        return label

    except Exception as e:
        logger.exception("Inference failed: %s", e)
        return None


def main() -> None:
    text = input("文章を入力してください：")
    result = inference_sentence(text=text)

    if result is not None:
        print(f"Predicted politeness level: {result}")
    else:
        print("Could not classify the sentence.")


if __name__ == "__main__":
    main()