import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmcodec import Compressor, Decompressor, ContextStrategy, list_available_models


def cmd_compress(args):
    strategy = ContextStrategy(args.context_strategy)

    compressor = Compressor(
        model_key=args.model,
        device=args.device,
        context_strategy=strategy,
    )

    results = compressor.compress(args.input, args.output, verbose=True)

    print(f"\nИтог: {results['original_size']:,} → {results['compressed_size']:,} "
          f"байт ({results['ratio']*100:.2f}%), {results['bpc']:.4f} bpc")


def cmd_decompress(args):
    decompressor = Decompressor(device=args.device)
    results = decompressor.decompress(args.input, args.output, verbose=True)

    if results["is_lossless"]:
        print("\n✓ Данные восстановлены без потерь.")
    else:
        print("\n✗ ОШИБКА: данные повреждены!")
        sys.exit(1)


def cmd_models(args):
    print("\nДоступные модели:")
    print(f"{'Ключ':15s} | {'Название':30s} | {'Семейство':10s} | {'Параметры':10s}")
    print("-" * 75)

    for m in list_available_models():
        print(f"{m['key']:15s} | {m['display_name']:30s} | {m['family']:10s} | {m['parameters']:10s}")


def main():
    parser = argparse.ArgumentParser(
        description="LMCodec — сжатие данных без потерь на основе LLM"
    )
    subparsers = parser.add_subparsers(dest="command")

    p_comp = subparsers.add_parser("compress", help="Сжатие файла")
    p_comp.add_argument("input", help="Входной файл")
    p_comp.add_argument("output", help="Выходной файл .lmc")
    p_comp.add_argument("--model", default="gpt2",
                        help="Модель (по умолчанию: gpt2)")
    p_comp.add_argument("--device", default=None,
                        help="Устройство (cuda/cpu)")
    p_comp.add_argument("--context-strategy", default="sliding_window",
                        choices=["sliding_window", "block"],
                        help="Стратегия контекста")
    p_comp.set_defaults(func=cmd_compress)

    p_dec = subparsers.add_parser("decompress", help="Декомпрессия файла")
    p_dec.add_argument("input", help="Файл .lmc")
    p_dec.add_argument("output", help="Выходной файл")
    p_dec.add_argument("--device", default=None, help="Устройство")
    p_dec.set_defaults(func=cmd_decompress)

    p_models = subparsers.add_parser("models", help="Список моделей")
    p_models.set_defaults(func=cmd_models)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
