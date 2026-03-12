#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
TEXT8_MIRROR = "https://data.deepai.org/text8.zip"

SIZES = {
    "1kb": 1024,
    "5kb": 5 * 1024,
    "10kb": 10 * 1024,
    "50kb": 50 * 1024,
    "100kb": 100 * 1024,
}


def download_text8(output_dir: str) -> str:

    os.makedirs(output_dir, exist_ok=True)
    text8_path = os.path.join(output_dir, "text8")

    if os.path.exists(text8_path):
        size = os.path.getsize(text8_path)
        print(f"text8 уже существует: {text8_path} ({size:,} байт)")
        return text8_path

    zip_path = os.path.join(output_dir, "text8.zip")

    for url in [TEXT8_URL, TEXT8_MIRROR]:
        try:
            print(f"Загрузка text8 из {url}...")
            urllib.request.urlretrieve(url, zip_path, _progress_hook)
            print()
            break
        except Exception as e:
            print(f"  Ошибка: {e}")
            continue
    else:
        print("ОШИБКА: не удалось скачать text8.")
        print("Скачайте вручную:")
        print(f"  wget {TEXT8_URL} -O {zip_path}")
        sys.exit(1)

    print("Распаковка...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    os.remove(zip_path)

    size = os.path.getsize(text8_path)
    print(f"text8: {size:,} байт ({size / 1024 / 1024:.1f} MB)")
    return text8_path


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = downloaded / total_size * 100
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)


def cut_files(text8_path: str, output_dir: str) -> dict:

    with open(text8_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    print(f"\ntext8 загружен: {len(full_text):,} символов")

    files = {}
    for name, size_bytes in SIZES.items():
        chunk = full_text[:size_bytes]


        last_space = chunk.rfind(" ")
        if last_space > size_bytes * 0.9:
            chunk = chunk[:last_space]

        filepath = os.path.join(output_dir, f"text8_{name}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chunk)

        actual_size = os.path.getsize(filepath)
        files[name] = filepath
        print(f"  {name:>6s}: {filepath} ({actual_size:,} байт)")

    return files


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка и подготовка тестовых данных text8"
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Директория для данных (по умолчанию: data/)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    text8_path = download_text8(output_dir)
    files = cut_files(text8_path, output_dir)

    print(f"\nГотово. Создано {len(files)} тестовых файлов в {output_dir}/")
    print("\nДля запуска экспериментов:")
    print(f"  python run_experiments.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()
