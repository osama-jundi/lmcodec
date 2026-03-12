#!/usr/bin/env python3

import argparse
import bz2
import gzip
import json
import lzma
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmcodec import Compressor, Decompressor, list_available_models


def benchmark_traditional(input_path: str) -> list[dict]:
    results = []

    with open(input_path, "rb") as f:
        data = f.read()

    original_size = len(data)

    start = time.time()
    gzip_data = gzip.compress(data)
    gzip_time = time.time() - start
    gzip_size = len(gzip_data)

    results.append({
        "method": "GZIP",
        "family": "Traditional",
        "original_size": original_size,
        "compressed_size": gzip_size,
        "ratio": gzip_size / original_size,
        "bpc": (gzip_size * 8) / original_size,
        "time_seconds": gzip_time,
        "speed_kbps": original_size / 1024 / gzip_time if gzip_time > 0 else 0,
    })

    start = time.time()
    bz2_data = bz2.compress(data)
    bz2_time = time.time() - start
    bz2_size = len(bz2_data)

    results.append({
        "method": "BZIP2",
        "family": "Traditional",
        "original_size": original_size,
        "compressed_size": bz2_size,
        "ratio": bz2_size / original_size,
        "bpc": (bz2_size * 8) / original_size,
        "time_seconds": bz2_time,
        "speed_kbps": original_size / 1024 / bz2_time if bz2_time > 0 else 0,
    })

    start = time.time()
    lzma_data = lzma.compress(data)
    lzma_time = time.time() - start
    lzma_size = len(lzma_data)

    results.append({
        "method": "LZMA",
        "family": "Traditional",
        "original_size": original_size,
        "compressed_size": lzma_size,
        "ratio": lzma_size / original_size,
        "bpc": (lzma_size * 8) / original_size,
        "time_seconds": lzma_time,
        "speed_kbps": original_size / 1024 / lzma_time if lzma_time > 0 else 0,
    })

    return results


def benchmark_llm(input_path: str, model_key: str, device: str = None) -> dict:
    original_size = os.path.getsize(input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        compressed_path = os.path.join(tmpdir, "compressed.llmz")
        restored_path = os.path.join(tmpdir, "restored.txt")

        compressor = Compressor(model_key=model_key, device=device)
        comp_results = compressor.compress(input_path, compressed_path, verbose=True)

        compressed_size = os.path.getsize(compressed_path)

        decompressor = Decompressor(device=device)
        decomp_results = decompressor.decompress(compressed_path, restored_path, verbose=True)

        return {
            "method": model_key,
            "family": "LLM",
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": comp_results["ratio"],
            "bpc": comp_results["bpc"],
            "time_seconds": comp_results["time_seconds"],
            "speed_kbps": original_size / 1024 / comp_results["time_seconds"] if comp_results["time_seconds"] > 0 else 0,
            "is_lossless": decomp_results["is_lossless"],
            "decompress_time": decomp_results["time_seconds"],
        }


def print_results(results: list[dict]):
    print("\n" + "=" * 90)
    print("РЕЗУЛЬТАТЫ СРАВНИТЕЛЬНОГО АНАЛИЗА")
    print("=" * 90)

    if not results:
        print("Нет результатов.")
        return

    original_size = results[0]["original_size"]
    print(f"\nИсходный размер: {original_size:,} байт")

    print(f"\n{'Метод':15s} | {'Семейство':12s} | {'Размер':>10s} | "
          f"{'Ratio':>7s} | {'BPC':>7s} | {'Время':>8s} | {'Скорость':>10s} | {'Lossless':>8s}")
    print("-" * 90)

    for r in results:
        lossless = r.get("is_lossless", "—")
        if isinstance(lossless, bool):
            lossless = "✓" if lossless else "✗"

        print(f"{r['method']:15s} | {r['family']:12s} | "
              f"{r['compressed_size']:>10,} | "
              f"{r['ratio']:>6.4f} | "
              f"{r['bpc']:>6.4f} | "
              f"{r['time_seconds']:>7.2f}s | "
              f"{r['speed_kbps']:>8.1f}kB/s | "
              f"{lossless:>8s}")


def save_results(results: list[dict], output_path: str):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nРезультаты сохранены: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Бенчмарк LLMZip — сравнение с традиционными методами"
    )
    parser.add_argument("--input", required=True, help="Входной файл")
    parser.add_argument("--models", nargs="+", default=["gpt2"],
                        help="Модели для тестирования")
    parser.add_argument("--all-models", action="store_true",
                        help="Тестировать все 5 моделей")
    parser.add_argument("--traditional-only", action="store_true",
                        help="Только традиционные методы")
    parser.add_argument("--device", default=None, help="cuda или cpu")
    parser.add_argument("--output", default=None,
                        help="Файл для сохранения результатов (JSON)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Файл не найден: {args.input}")
        sys.exit(1)

    all_results = []

    print("\n" + "=" * 60)
    print("БЕНЧМАРК ТРАДИЦИОННЫХ МЕТОДОВ")
    print("=" * 60)

    trad_results = benchmark_traditional(args.input)
    all_results.extend(trad_results)

    for r in trad_results:
        print(f"  {r['method']:6s}: {r['compressed_size']:>10,} байт "
              f"(ratio={r['ratio']:.4f}, bpc={r['bpc']:.4f}, "
              f"time={r['time_seconds']:.4f}s)")

    if not args.traditional_only:
        if args.all_models:
            model_keys = [m["key"] for m in list_available_models()]
        else:
            model_keys = args.models

        for model_key in model_keys:
            print(f"\n" + "=" * 60)
            print(f"БЕНЧМАРК LLM: {model_key}")
            print("=" * 60)

            try:
                result = benchmark_llm(args.input, model_key, args.device)
                all_results.append(result)
            except Exception as e:
                print(f"  ОШИБКА: {e}")
                all_results.append({
                    "method": model_key,
                    "family": "LLM",
                    "error": str(e),
                })

    print_results([r for r in all_results if "error" not in r])

    if args.output:
        save_results(all_results, args.output)


if __name__ == "__main__":
    main()
