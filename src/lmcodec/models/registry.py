
from typing import Optional
from pathlib import Path

import torch

from lmcodec.models.base import BaseProbabilityModel
from lmcodec.models.hf_model import HuggingFaceCausalModel


# Автоопределение папки кеша моделей внутри проекта.
# Ищем models_cache/ поднимаясь от текущего файла:
#   registry.py → models/ → lmcodec/ → src/ → PROJECT_ROOT/models_cache
def _find_project_cache() -> Optional[str]:
    current = Path(__file__).resolve()
    # Поднимаемся до корня проекта (ищем pyproject.toml или models_cache/)
    for parent in [current.parent.parent.parent.parent,  # src/lmcodec/models/ → project root
                   current.parent.parent.parent,          # если структура другая
                   Path.cwd()]:                           # или текущая директория
        cache_dir = parent / "models_cache"
        if cache_dir.exists() and cache_dir.is_dir():
            return str(cache_dir)
    return None


MODEL_CONFIGS = {
    "gpt2": {
        "hf_name": "gpt2",
        "display_name": "GPT-2 (124M)",
        "family": "GPT-2",
        "parameters": "124M",
        "torch_dtype": None,
        "trust_remote_code": False,
    },
    "gpt2-medium": {
        "hf_name": "gpt2-medium",
        "display_name": "GPT-2 Medium (355M)",
        "family": "GPT-2",
        "parameters": "355M",
        "torch_dtype": None,
        "trust_remote_code": False,
    },
    "opt-1.3b": {
        "hf_name": "facebook/opt-1.3b",
        "display_name": "OPT-1.3B (Meta)",
        "family": "OPT",
        "parameters": "1.3B",
        "torch_dtype": torch.float16,
        "trust_remote_code": False,
    },
    "phi-2": {
        "hf_name": "microsoft/phi-2",
        "display_name": "Phi-2 (2.7B, Microsoft)",
        "family": "Phi",
        "parameters": "2.7B",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    },
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "display_name": "Mistral-7B (Mistral AI)",
        "family": "Mistral",
        "parameters": "7B",
        "torch_dtype": torch.float16,
        "trust_remote_code": False,
    },
}


def create_model(
    model_key: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> BaseProbabilityModel:
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise ValueError(
            f"Неизвестная модель: '{model_key}'. "
            f"Доступные модели: {available}"
        )

    config = MODEL_CONFIGS[model_key]

    # Приоритет: аргумент → models_cache/ в проекте → HuggingFace default
    if cache_dir is None:
        cache_dir = _find_project_cache()

    model = HuggingFaceCausalModel(
        model_name=config["hf_name"],
        device=device,
        torch_dtype=config.get("torch_dtype"),
        trust_remote_code=config.get("trust_remote_code", False),
        cache_dir=cache_dir,
    )

    return model


def list_available_models() -> list[dict]:
    models = []
    for key, config in MODEL_CONFIGS.items():
        models.append({
            "key": key,
            "display_name": config["display_name"],
            "hf_name": config["hf_name"],
            "family": config["family"],
            "parameters": config["parameters"],
        })
    return models


def get_model_display_name(model_key: str) -> str:
    if model_key in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_key]["display_name"]
    return model_key