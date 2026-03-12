#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CACHE = str(PROJECT_ROOT / "models_cache")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lmcodec.models.registry import MODEL_CONFIGS


def setup_cache(cache_dir: str) -> str:

    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    return cache_dir


def download_model(key: str, config: dict, cache_dir: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_name = config["hf_name"]
    print(f"\n{'='*60}")
    print(f"Загрузка: {config['display_name']} ({hf_name})")
    print(f"Кеш: {cache_dir}")
    print(f"{'='*60}")

    print(f"  Токенизатор...", end=" ", flush=True)
    AutoTokenizer.from_pretrained(
        hf_name,
        cache_dir=cache_dir,
        trust_remote_code=config.get("trust_remote_code", False),
    )
    print("OK")

    print(f"  Веса модели...", end=" ", flush=True)
    AutoModelForCausalLM.from_pretrained(
        hf_name,
        cache_dir=cache_dir,
        trust_remote_code=config.get("trust_remote_code", False),
    )
    print("OK")

    print(f"  {config['display_name']} — загружена.")


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка моделей LLM в папку проекта"
    )
    parser.add_argument(
        "models", nargs="*", default=None,
        help="Ключи моделей (по умолчанию: все 5)"
    )
    parser.add_argument(
        "--cache-dir", default=DEFAULT_CACHE,
        help=f"Папка для моделей (по умолчанию: models_cache/)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Показать список доступных моделей"
    )
    args = parser.parse_args()

    if args.list:
        print("\nДоступные модели:")
        for key, cfg in MODEL_CONFIGS.items():
            print(f"  {key:15s}  {cfg['display_name']:30s}  {cfg['parameters']}")
        return

    cache_dir = setup_cache(args.cache_dir)
    print(f"Кеш моделей: {cache_dir}")

    keys = args.models if args.models else list(MODEL_CONFIGS.keys())

    for key in keys:
        if key not in MODEL_CONFIGS:
            print(f"ОШИБКА: модель '{key}' не найдена.")
            print(f"Доступные: {', '.join(MODEL_CONFIGS.keys())}")
            sys.exit(1)

    print(f"Моделей к загрузке: {len(keys)}")
    print(f"Модели: {', '.join(keys)}")

    for key in keys:
        download_model(key, MODEL_CONFIGS[key], cache_dir)

    total = sum(f.stat().st_size for f in Path(cache_dir).rglob("*") if f.is_file())
    print(f"\n{'='*60}")
    print(f"Все {len(keys)} моделей загружены.")
    print(f"Папка: {cache_dir}")
    print(f"Размер: {total / 1024 / 1024 / 1024:.1f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
