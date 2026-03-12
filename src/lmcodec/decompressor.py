import time
from typing import Optional

import numpy as np

from lmcodec.arithmetic import ArithmeticDecoder
from lmcodec.context import ContextManager, ContextStrategy
from lmcodec.file_format import  compute_hash, load_compressed
from lmcodec.models import BaseProbabilityModel
from lmcodec.models import create_model
from lmcodec.preprocessing import Preprocessor


class Decompressor:

    def __init__(
        self,
        device: Optional[str] = None,
    ):
        self.device = device
        self._models_cache: dict[str, BaseProbabilityModel] = {}

    def _get_model(self, model_key: str) -> BaseProbabilityModel:
        if model_key not in self._models_cache:
            model = create_model(model_key, self.device)
            model.load()
            self._models_cache[model_key] = model
        return self._models_cache[model_key]

    def decompress(
        self,
        input_path: str,
        output_path: str,
        verbose: bool = True,
    ) -> dict:
        start_time = time.time()

        if verbose:
            print(f"\n[DECOMPRESS] Файл: {input_path}")

        metadata, compressed_data = load_compressed(input_path)

        if verbose:
            print(f"[DECOMPRESS] Модель: {metadata.model_key}")
            print(f"[DECOMPRESS] Токенов: {metadata.num_tokens}")
            print(f"[DECOMPRESS] Исходный размер: {metadata.original_size:,} байт")

        model = self._get_model(metadata.model_key)

        if hasattr(model, 'reset_cache'):
            model.reset_cache()

        context_strategy = ContextStrategy(metadata.context_strategy)
        context_manager = ContextManager(
            max_context_length=metadata.max_context_length,
            strategy=context_strategy,
        )

        decoder = ArithmeticDecoder(compressed_data)
        num_tokens = metadata.num_tokens
        decoded_tokens: list[int] = []

        for i in range(num_tokens):
            context = context_manager.get_context(decoded_tokens, i)

            if len(context) == 0:
                probs = np.ones(model.vocab_size, dtype=np.float64)
                probs /= probs.sum()
            else:
                probs = model.get_probabilities(context)

            probs = _stabilize_probabilities(probs, model.vocab_size)

            token_id = decoder.decode_symbol(probs)
            decoded_tokens.append(token_id)

            if verbose and (i + 1) % 100 == 0:
                pct = (i + 1) / num_tokens * 100
                print(f"\r[DECOMPRESS] Прогресс: {i+1}/{num_tokens} ({pct:.1f}%)", end="", flush=True)

        if verbose:
            print(f"\r[DECOMPRESS] Прогресс: {num_tokens}/{num_tokens} (100.0%)")

        text = model.decode_tokens(decoded_tokens)

        preprocessor = Preprocessor(encoding=metadata.encoding)
        restored_text = preprocessor.reverse_preprocess(text)

        restored_bytes = restored_text.encode(metadata.encoding)
        restored_hash = compute_hash(restored_bytes)

        is_lossless = (restored_hash == metadata.original_hash)

        if not is_lossless:
            print(f"[DECOMPRESS] ПРЕДУПРЕЖДЕНИЕ: хеши не совпадают!")
            print(f"  Ожидаемый: {metadata.original_hash}")
            print(f"  Полученный: {restored_hash}")

        preprocessor.write_file(output_path, restored_text)

        elapsed = time.time() - start_time

        results = {
            "input_path": input_path,
            "output_path": output_path,
            "restored_size": len(restored_bytes),
            "original_size": metadata.original_size,
            "is_lossless": is_lossless,
            "model": metadata.model_key,
            "time_seconds": elapsed,
        }

        if verbose:
            status = "✓ ДА" if is_lossless else "✗ НЕТ"
            print(f"[DECOMPRESS] Без потерь: {status}")
            print(f"[DECOMPRESS] Время: {elapsed:.2f} с")
            print(f"[DECOMPRESS] Файл: {output_path}")

        return results


def _stabilize_probabilities(probs: np.ndarray, vocab_size: int) -> np.ndarray:
    min_prob = 1e-10
    probs = np.maximum(probs, min_prob)

    if len(probs) > vocab_size:
        probs = probs[:vocab_size]
    elif len(probs) < vocab_size:
        padding = np.full(vocab_size - len(probs), min_prob)
        probs = np.concatenate([probs, padding])

    probs /= probs.sum()
    return probs
