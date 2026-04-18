import time
from typing import Optional

import numpy as np

from lmcodec.arithmetic import ArithmeticEncoder
from lmcodec.context import ContextManager, ContextStrategy
from lmcodec.file_format import FileMetadata, compute_hash, save_compressed
from lmcodec.models import BaseProbabilityModel
from lmcodec.models import create_model
from lmcodec.models.registry import MODEL_CONFIGS
from lmcodec.preprocessing import Preprocessor
from lmcodec.utils import stabilize_probabilities


class Compressor:

    def __init__(
        self,
        model_key: str = "gpt2",
        device: Optional[str] = None,
        context_strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW,
        encoding: str = "utf-8",
    ):
        # Bug 4 fix: validate model_key early instead of failing deep in the pipeline
        if model_key not in MODEL_CONFIGS:
            available = ", ".join(sorted(MODEL_CONFIGS.keys()))
            raise ValueError(
                f"Неизвестная модель: '{model_key}'. Доступные: {available}"
            )

        self.model_key = model_key
        self.device = device
        self.context_strategy = context_strategy
        self.encoding = encoding

        self._model: Optional[BaseProbabilityModel] = None
        self._preprocessor = Preprocessor(encoding=encoding)
        self._context_manager: Optional[ContextManager] = None

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            self._model = create_model(self.model_key, self.device)
            self._model.load()
            self._context_manager = ContextManager(
                max_context_length=self._model.max_context_length,
                strategy=self.context_strategy,
            )

    def compress(
        self,
        input_path: str,
        output_path: str,
        verbose: bool = True,
    ) -> dict:
        self._ensure_model_loaded()
        start_time = time.time()

        if verbose:
            print(f"\n[COMPRESS] Входной файл: {input_path}")

        raw_text = self._preprocessor.read_file(input_path)
        text, preproc_info = self._preprocessor.preprocess(raw_text)
        original_bytes = raw_text.encode(self.encoding)
        original_size = len(original_bytes)
        original_hash = compute_hash(original_bytes)

        if verbose:
            print(f"[COMPRESS] Размер: {original_size:,} байт")

        token_ids = self._model.encode_text(text)
        num_tokens = len(token_ids)

        if verbose:
            print(f"[COMPRESS] Токенов: {num_tokens}")
            print(f"[COMPRESS] Модель: {self.model_key}")
            print(f"[COMPRESS] Фаза 1: вычисление вероятностей (с KV-cache)...")

        if hasattr(self._model, 'reset_cache'):
            self._model.reset_cache()

        all_probs = []
        for i in range(num_tokens):
            context = self._context_manager.get_context(token_ids, i)

            if len(context) == 0:
                probs = np.ones(self._model.vocab_size, dtype=np.float64)
                probs /= probs.sum()
            else:
                probs = self._model.get_probabilities(context)

            probs = stabilize_probabilities(probs, self._model.vocab_size)
            all_probs.append(probs)

            if verbose and (i + 1) % 100 == 0:
                pct = (i + 1) / num_tokens * 100
                print(f"\r[COMPRESS] Фаза 1: {i+1}/{num_tokens} ({pct:.1f}%)",
                      end="", flush=True)

        if verbose:
            print(f"\r[COMPRESS] Фаза 1: {num_tokens}/{num_tokens} (100.0%)")
            print(f"[COMPRESS] Фаза 2: арифметическое кодирование...")

        encoder = ArithmeticEncoder()
        encoder.create_encoder()

        for i in range(num_tokens - 1, -1, -1):
            encoder.encode_symbol(token_ids[i], all_probs[i])

        compressed_data = encoder.finish()

        del all_probs

        metadata = FileMetadata(
            model_key=self.model_key,
            model_name=self._model.model_name,
            vocab_size=self._model.vocab_size,
            max_context_length=self._model.max_context_length,
            context_strategy=self.context_strategy.value,
            encoding=self.encoding,
            num_tokens=num_tokens,
            original_size=original_size,
            original_hash=original_hash,
            preproc_info=preproc_info,
        )

        compressed_size = save_compressed(
            output_path, metadata, compressed_data,
        )

        elapsed = time.time() - start_time

        ratio = compressed_size / original_size if original_size > 0 else 0
        bpc = (compressed_size * 8) / original_size if original_size > 0 else 0

        results = {
            "input_path": input_path,
            "output_path": output_path,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": ratio,
            "bpc": bpc,
            "num_tokens": num_tokens,
            "model": self.model_key,
            "time_seconds": elapsed,
        }

        if verbose:
            print(f"[COMPRESS] Сжатый размер: {compressed_size:,} байт")
            print(f"[COMPRESS] Коэффициент: {ratio:.4f} ({ratio*100:.2f}%)")
            print(f"[COMPRESS] BPC: {bpc:.4f}")
            print(f"[COMPRESS] Время: {elapsed:.2f} с")

        return results
