from enum import Enum
from typing import Optional


class ContextStrategy(Enum):
    SLIDING_WINDOW = "sliding_window"
    BLOCK = "block"


class ContextManager:

    def __init__(
        self,
        max_context_length: int,
        strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW,
    ):
        self.max_context_length = max_context_length
        self.strategy = strategy

    def get_context(
        self,
        all_tokens: list[int],
        current_position: int,
    ) -> list[int]:
        if self.strategy == ContextStrategy.SLIDING_WINDOW:
            return self._sliding_window(all_tokens, current_position)
        elif self.strategy == ContextStrategy.BLOCK:
            return self._block_context(all_tokens, current_position)
        else:
            raise ValueError(f"Неизвестная стратегия: {self.strategy}")

    def _sliding_window(
        self,
        all_tokens: list[int],
        current_position: int,
    ) -> list[int]:
        context = all_tokens[:current_position]

        if len(context) > self.max_context_length:
            context = context[-self.max_context_length:]

        return context

    def _block_context(
        self,
        all_tokens: list[int],
        current_position: int,
    ) -> list[int]:
        block_size = self.max_context_length
        block_start = (current_position // block_size) * block_size
        context = all_tokens[block_start:current_position]
        return context

    def get_config(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "max_context_length": self.max_context_length,
        }
