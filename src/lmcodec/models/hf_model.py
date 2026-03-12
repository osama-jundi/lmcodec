import logging
import warnings
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmcodec.models.base import BaseProbabilityModel

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*sequence length.*")


class HuggingFaceCausalModel(BaseProbabilityModel):

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self._torch_dtype = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._cache_dir = cache_dir

        self._kv_cache = None
        self._cached_token_ids: list[int] = []

    def load(self) -> None:
        print(f"[MODEL] Загрузка модели: {self.model_name}")
        print(f"[MODEL] Устройство: {self.device}")
        if self._cache_dir:
            print(f"[MODEL] Кеш: {self._cache_dir}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self._trust_remote_code,
            cache_dir=self._cache_dir,
        )

        dtype = self._torch_dtype
        if dtype is None:
            if self.device == "cuda" and torch.cuda.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=self._trust_remote_code,
            cache_dir=self._cache_dir,
        ).to(self.device)

        self._model.eval()

        self._vocab_size = self._model.config.vocab_size
        self._max_context_length = self._get_max_context_length()

        print(f"[MODEL] Словарь: {self._vocab_size} токенов")
        print(f"[MODEL] Контекстное окно: {self._max_context_length} токенов")
        print(f"[MODEL] KV-cache: включён")
        print(f"[MODEL] Загрузка завершена")

    def _get_max_context_length(self) -> int:
        config = self._model.config
        for attr in [
            "max_position_embeddings",
            "n_positions",
            "seq_length",
            "max_sequence_length",
        ]:
            if hasattr(config, attr):
                return getattr(config, attr)
        return 1024

    def reset_cache(self) -> None:
        self._kv_cache = None
        self._cached_token_ids = []

    def _can_extend_cache(self, token_ids: list[int]) -> bool:
        if self._kv_cache is None:
            return False
        if len(token_ids) < len(self._cached_token_ids):
            return False
        cached_len = len(self._cached_token_ids)
        if token_ids[:cached_len] != self._cached_token_ids:
            return False
        return True

    @torch.no_grad()
    def get_probabilities(self, token_ids: list[int]) -> np.ndarray:
        self._ensure_loaded()

        max_len = self._max_context_length
        if len(token_ids) > max_len:
            token_ids = token_ids[-max_len:]
            self.reset_cache()

        if self._can_extend_cache(token_ids):
            new_tokens = token_ids[len(self._cached_token_ids):]
            input_ids = torch.tensor(
                [new_tokens], dtype=torch.long, device=self.device
            )

            outputs = self._model(
                input_ids,
                past_key_values=self._kv_cache,
                use_cache=True,
            )
        else:
            self.reset_cache()
            input_ids = torch.tensor(
                [token_ids], dtype=torch.long, device=self.device
            )

            outputs = self._model(
                input_ids,
                use_cache=True,
            )

        self._kv_cache = outputs.past_key_values
        self._cached_token_ids = list(token_ids)

        logits = outputs.logits[0, -1, :]
        probabilities = torch.softmax(logits, dim=-1)

        return probabilities.cpu().numpy().astype(np.float64)

    def encode_text(self, text: str) -> list[int]:
        self._ensure_loaded()
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, token_ids: list[int]) -> str:
        self._ensure_loaded()
        return self._tokenizer.decode(token_ids)

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info["dtype"] = str(self._torch_dtype or "auto")
        info["kv_cache"] = True
        info["cache_dir"] = self._cache_dir
        return info