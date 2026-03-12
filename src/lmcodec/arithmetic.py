from typing import Optional

import numpy as np
import constriction


class ArithmeticEncoder:

    def __init__(self):
        self._encoder = None
        self._encoded_data: Optional[np.ndarray] = None

    def create_encoder(self) -> None:
        self._encoder = constriction.stream.stack.AnsCoder()

    def encode_symbol(
        self,
        symbol: int,
        probabilities: np.ndarray,
    ) -> None:
        probs = probabilities.astype(np.float32)

        entropy_model = constriction.stream.model.Categorical(probs)

        self._encoder.encode_reverse(
            np.array([symbol], dtype=np.int32),
            entropy_model,
        )

    def finish(self) -> np.ndarray:
        return self._encoder.get_compressed()

    def get_compressed_size_bits(self) -> int:
        compressed = self._encoder.get_compressed()
        return len(compressed) * 32


class ArithmeticDecoder:

    def __init__(self, compressed_data: np.ndarray):
        self._decoder = constriction.stream.stack.AnsCoder(compressed_data)

    def decode_symbol(self, probabilities: np.ndarray) -> int:
        probs = probabilities.astype(np.float32)
        entropy_model = constriction.stream.model.Categorical(probs)

        decoded = self._decoder.decode(entropy_model, 1)
        return int(decoded[0])
