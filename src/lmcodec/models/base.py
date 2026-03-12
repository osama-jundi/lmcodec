from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class BaseProbabilityModel(ABC):

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._tokenizer = None
        self._vocab_size: int = 0
        self._max_context_length: int = 0

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def get_probabilities(self, token_ids: list[int]) -> np.ndarray:
        pass

    @abstractmethod
    def encode_text(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode_tokens(self, token_ids: list[int]) -> str:
        pass

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "vocab_size": self._vocab_size,
            "max_context_length": self._max_context_length,
            "device": self.device,
        }
