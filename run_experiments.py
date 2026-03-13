#!/usr/bin/env python3
import argparse
import bz2
import csv
import gzip
import json
import lzma
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODELS_CACHE = str(PROJECT_ROOT / "models_cache")

if os.path.exists(MODELS_CACHE):
    os.environ["HF_HOME"] = MODELS_CACHE
    os.environ["TRANSFORMERS_CACHE"] = MODELS_CACHE
    os.environ["HUGGINGFACE_HUB_CACHE"] = MODELS_CACHE

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lmcodec import Compressor, Decompressor, ContextStrategy

def compress_traditional(data: bytes) -> list[dict]:
    results = []
    original = len(data)

    for name, func in [("GZIP", gzip.compress), ("BZIP2", bz2.compress), ("LZMA", lzma.compress)]:
        t0 = time.time()
        compressed = func(data)
        elapsed = time.time() - t0
        size = len(compressed)
        results.append({
            "method": name,
            "family": "traditional",
            "original_size": original,
            "compressed_size": size,
            "ratio": size / original,
            "bpc": (size * 8) / original,
            "time_seconds": elapsed,
            "is_lossless": True,
        })
    return results

def compress_llm(
    input_path: str,
    model_key: str,
    context_strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW,
    device: str = None,
) -> dict:
    original_size = os.path.getsize(input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        compressed_path = os.path.join(tmpdir, "out.lmc")
        restored_path = os.path.join(tmpdir, "restored.txt")

        compressor = Compressor(
            model_key=model_key,
            device=device,
            context_strategy=context_strategy,
        )
        comp = compressor.compress(input_path, compressed_path, verbose=True)

        decompressor = Decompressor(device=device)
        decomp = decompressor.decompress(compressed_path, restored_path, verbose=True)

        return {
            "method": model_key,
            "family": "LLM",
            "original_size": original_size,
            "compressed_size": comp["compressed_size"],
            "ratio": comp["ratio"],
            "bpc": comp["bpc"],
            "time_seconds": comp["time_seconds"],
            "decompress_time": decomp["time_seconds"],
            "is_lossless": decomp["is_lossless"],
            "context_strategy": context_strategy.value,
            "num_tokens": comp["num_tokens"],
        }

def test_h1(input_path: str, model_keys: list[str], device: str = None) -> list[dict]:
    print("\n" + "=" * 70)
    print("ГИПОТЕЗА H1: Увеличение размера модели → снижение BPC")
    print("=" * 70)

    results = []
    for key in model_keys:
        print(f"\n--- Модель: {key} ---")
        try:
            r = compress_llm(input_path, key, device=device)
            r["hypothesis"] = "H1"
            results.append(r)
            print(f"  BPC = {r['bpc']:.4f}, время = {r['time_seconds']:.1f}с, lossless = {r['is_lossless']}")
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            results.append({"method": key, "hypothesis": "H1", "error": str(e)})

    valid = [r for r in results if "error" not in r]
    if len(valid) >= 2:
        bpcs = [r["bpc"] for r in valid]
        trend = all(bpcs[i] >= bpcs[i + 1] for i in range(len(bpcs) - 1))
        print(f"\n  H1 результат: BPC = {[f'{b:.4f}' for b in bpcs]}")
        print(f"  H1 {'ПОДТВЕРЖДЕНА ✓' if trend else 'НЕ ПОДТВЕРЖДЕНА (немонотонно)'}")

    return results


def test_h2(input_path: str, device: str = None) -> list[dict]:
    print("\n" + "=" * 70)
    print("ГИПОТЕЗА H2: GPT-2 (124M) < GZIP/BZIP2/LZMA по BPC")
    print("=" * 70)

    with open(input_path, "rb") as f:
        data = f.read()

    trad_results = compress_traditional(data)
    for r in trad_results:
        r["hypothesis"] = "H2"
        print(f"  {r['method']:6s}: BPC = {r['bpc']:.4f}, время = {r['time_seconds']:.4f}с")

    print(f"\n--- Модель: gpt2 ---")
    llm_result = compress_llm(input_path, "gpt2", device=device)
    llm_result["hypothesis"] = "H2"
    print(f"  GPT-2:  BPC = {llm_result['bpc']:.4f}, время = {llm_result['time_seconds']:.1f}с")

    results = trad_results + [llm_result]

    gpt2_bpc = llm_result["bpc"]
    beaten = [r["method"] for r in trad_results if gpt2_bpc < r["bpc"]]
    not_beaten = [r["method"] for r in trad_results if gpt2_bpc >= r["bpc"]]

    print(f"\n  GPT-2 BPC ({gpt2_bpc:.4f}) лучше: {beaten if beaten else 'никого'}")
    if not_beaten:
        print(f"  GPT-2 BPC ({gpt2_bpc:.4f}) НЕ лучше: {not_beaten}")
    print(f"  H2 {'ПОДТВЕРЖДЕНА ✓' if len(beaten) == 3 else 'ЧАСТИЧНО' if beaten else 'НЕ ПОДТВЕРЖДЕНА'}")

    return results


def test_h3(input_path: str, model_keys: list[str], device: str = None) -> list[dict]:
    print("\n" + "=" * 70)
    print("ГИПОТЕЗА H3: LLM на порядки медленнее традиционных")
    print("=" * 70)

    with open(input_path, "rb") as f:
        data = f.read()

    trad_results = compress_traditional(data)
    for r in trad_results:
        r["hypothesis"] = "H3"

    avg_trad_time = sum(r["time_seconds"] for r in trad_results) / len(trad_results)
    print(f"  Среднее время традиционных: {avg_trad_time:.4f}с")

    results = list(trad_results)

    for key in model_keys:
        print(f"\n--- Модель: {key} ---")
        try:
            r = compress_llm(input_path, key, device=device)
            r["hypothesis"] = "H3"
            results.append(r)
            slowdown = r["time_seconds"] / avg_trad_time if avg_trad_time > 0 else float("inf")
            print(f"  Время: {r['time_seconds']:.1f}с (в {slowdown:.0f}× медленнее)")
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            results.append({"method": key, "hypothesis": "H3", "error": str(e)})

    valid_llm = [r for r in results if r.get("family") == "LLM" and "error" not in r]
    if valid_llm:
        slowdowns = [r["time_seconds"] / avg_trad_time for r in valid_llm]
        print(f"\n  Замедление: {[f'{s:.0f}×' for s in slowdowns]}")
        orders = all(s >= 10 for s in slowdowns)
        print(f"  H3 {'ПОДТВЕРЖДЕНА ✓' if orders else 'ЧАСТИЧНО (не все на порядки)'}")

    return results


def test_h4(input_path: str, model_key: str = "gpt2", device: str = None) -> list[dict]:
    print("\n" + "=" * 70)
    print("ГИПОТЕЗА H4: Скользящее окно > блочное сжатие")
    print(f"  Модель: {model_key}")
    print("=" * 70)

    results = []

    for strategy in [ContextStrategy.SLIDING_WINDOW, ContextStrategy.BLOCK]:
        print(f"\n--- Стратегия: {strategy.value} ---")
        try:
            r = compress_llm(input_path, model_key, context_strategy=strategy, device=device)
            r["hypothesis"] = "H4"
            results.append(r)
            print(f"  BPC = {r['bpc']:.4f}, время = {r['time_seconds']:.1f}с")
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            results.append({"method": model_key, "context_strategy": strategy.value,
                            "hypothesis": "H4", "error": str(e)})

    valid = [r for r in results if "error" not in r]
    if len(valid) == 2:
        sw = next(r for r in valid if r["context_strategy"] == "sliding_window")
        bl = next(r for r in valid if r["context_strategy"] == "block")
        print(f"\n  Скользящее окно: BPC = {sw['bpc']:.4f}")
        print(f"  Блочное сжатие:  BPC = {bl['bpc']:.4f}")
        print(f"  Разница: {bl['bpc'] - sw['bpc']:.4f} bpc")
        print(f"  H4 {'ПОДТВЕРЖДЕНА ✓' if sw['bpc'] < bl['bpc'] else 'НЕ ПОДТВЕРЖДЕНА'}")

    return results

def save_results(results: list[dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON: {json_path}")

    csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
    valid = [r for r in results if "error" not in r]
    if valid:
        all_keys = set()
        for r in valid:
            all_keys.update(r.keys())
        keys = sorted(all_keys)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(valid)
        print(f"CSV:  {csv_path}")

    return json_path, csv_path


def print_summary(results: list[dict]):
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("Нет результатов.")
        return

    print("\n" + "=" * 95)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 95)
    print(f"{'Гипотеза':>8s} | {'Метод':>15s} | {'Стратегия':>15s} | "
          f"{'Размер':>8s} | {'BPC':>8s} | {'Время':>10s} | {'Lossless':>8s}")
    print("-" * 95)

    for r in valid:
        hyp = r.get("hypothesis", "—")
        method = r.get("method", "?")
        strat = r.get("context_strategy", "—")
        size = r.get("compressed_size", 0)
        bpc = r.get("bpc", 0)
        t = r.get("time_seconds", 0)
        ll = r.get("is_lossless", "—")
        if isinstance(ll, bool):
            ll = "✓" if ll else "✗"
        print(f"{hyp:>8s} | {method:>15s} | {strat:>15s} | "
              f"{size:>8,} | {bpc:>8.4f} | {t:>9.2f}s | {ll:>8s}")


def main():
    parser = argparse.ArgumentParser(
        description="Экспериментальная проверка гипотез H1–H4 (Глава 4)"
    )
    parser.add_argument("--data-dir", default="data",
                        help="Директория с тестовыми файлами")
    parser.add_argument("--size", default="100kb",
                        choices=["1kb", "5kb", "10kb", "50kb", "100kb"],
                        help="Размер тестового файла (по умолчанию: 100kb)")
    parser.add_argument("--hypothesis", nargs="+", default=["H1", "H2", "H3", "H4"],
                        choices=["H1", "H2", "H3", "H4"],
                        help="Какие гипотезы проверять")
    parser.add_argument("--models", nargs="+",
                        default=["gpt2", "gpt2-medium", "opt-1.3b", "phi-2", "mistral-7b"],
                        help="Модели для H1/H3")
    parser.add_argument("--h4-model", default="gpt2",
                        help="Модель для H4 (по умолчанию: gpt2)")
    parser.add_argument("--device", default=None, help="cuda или cpu")
    parser.add_argument("--output-dir", default="results",
                        help="Директория для результатов")

    args = parser.parse_args()

    input_path = os.path.join(args.data_dir, f"text8_{args.size}.txt")
    if not os.path.exists(input_path):
        print(f"ОШИБКА: файл не найден: {input_path}")
        print(f"Сначала запустите: python download_test_data.py --output-dir {args.data_dir}")
        sys.exit(1)

    file_size = os.path.getsize(input_path)
    cache_status = "✓ найден" if os.path.exists(MODELS_CACHE) else "✗ не найден (будет загружать из ~/.cache)"

    print(f"{'='*70}")
    print(f"ЭКСПЕРИМЕНТАЛЬНАЯ ПРОВЕРКА ГИПОТЕЗ (Глава 4)")
    print(f"{'='*70}")
    print(f"Файл:      {input_path} ({file_size:,} байт)")
    print(f"Гипотезы:  {', '.join(args.hypothesis)}")
    print(f"Модели:    {', '.join(args.models)}")
    print(f"Устройство: {args.device or 'auto'}")
    print(f"Кеш моделей: {MODELS_CACHE} ({cache_status})")
    print(f"{'='*70}")

    all_results = []

    if "H1" in args.hypothesis:
        all_results.extend(test_h1(input_path, args.models, args.device))

    if "H2" in args.hypothesis:
        all_results.extend(test_h2(input_path, args.device))

    if "H3" in args.hypothesis:
        all_results.extend(test_h3(input_path, args.models, args.device))

    if "H4" in args.hypothesis:
        all_results.extend(test_h4(input_path, args.h4_model, args.device))

    print_summary(all_results)
    save_results(all_results, args.output_dir)

    print(f"\nЗавершено. Результаты в {args.output_dir}/")

if __name__ == "__main__":
    main()
