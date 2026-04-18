"""
Full compression tests for lmcodec.

The last test (test_compression_lossless) loads GPT-2 (~500 MB) and
requires a working PyTorch install.

Run: python tests/test_compression.py
Or:  pytest tests/test_compression.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Allow running directly without pip install -e .
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lmcodec import Compressor, Decompressor, ContextStrategy, list_available_models
from lmcodec.arithmetic import ArithmeticEncoder, ArithmeticDecoder
from lmcodec.context import ContextManager
from lmcodec.file_format import FileMetadata, save_compressed, load_compressed
from lmcodec.preprocessing import Preprocessor


def test_list_models():
    print("\n" + "=" * 60)
    print("ТЕСТ: Список доступных моделей")
    print("=" * 60)

    models = list_available_models()
    assert len(models) == 5, f"Ожидалось 5 моделей, получено {len(models)}"

    for m in models:
        print(f"  {m['key']:15s} | {m['display_name']:30s} | {m['family']}")

    print("✓ Пройден")


def test_arithmetic_coding():
    print("\n" + "=" * 60)
    print("ТЕСТ: Арифметическое кодирование")
    print("=" * 60)

    vocab_size = 100
    num_symbols = 50
    np.random.seed(42)

    symbols = np.random.randint(0, vocab_size, size=num_symbols)

    all_probs = []
    for _ in range(num_symbols):
        p = np.random.dirichlet(np.ones(vocab_size))
        all_probs.append(p.astype(np.float64))

    encoder = ArithmeticEncoder()
    encoder.create_encoder()

    for i in range(num_symbols - 1, -1, -1):
        probs = all_probs[i].astype(np.float32)
        probs /= probs.sum()
        encoder.encode_symbol(int(symbols[i]), probs)

    compressed = encoder.finish()
    print(f"  Символов: {num_symbols}")
    print(f"  Сжатый размер: {len(compressed) * 4} байт")

    decoder = ArithmeticDecoder(compressed)
    decoded = []
    for i in range(num_symbols):
        probs = all_probs[i].astype(np.float32)
        probs /= probs.sum()
        sym = decoder.decode_symbol(probs)
        decoded.append(sym)

    assert list(symbols) == decoded, "Декодированные символы не совпадают!"
    print("✓ Пройден")


def test_preprocessing():
    """
    Bug 2 fix verification: preprocessing must be byte-exactly reversible
    for any combination of line endings (\\r\\n, \\r, \\n, mixed).
    """
    print("\n" + "=" * 60)
    print("ТЕСТ: Предобработка (побитовая обратимость)")
    print("=" * 60)

    preprocessor = Preprocessor()

    test_cases = [
        "Hello, World!",                       # no newlines
        "Line1\nLine2\nLine3",                 # LF only
        "Line1\r\nLine2\r\nLine3",             # CRLF only (Windows)
        "Line1\rLine2\rLine3",                 # CR only (old Mac)
        "Mixed\r\nline\rendings\n",            # all three, mixed
        "Привет, мир! Unicode тест 🎉\n",       # Unicode + LF
        "",                                    # empty
        "\n\n\n",                              # only newlines
        "No trailing newline",                 # no final EOL
    ]

    for text in test_cases:
        # New API: preprocess returns (normalized_text, info_dict)
        processed, info = preprocessor.preprocess(text)
        # All newline variants must be \n in the normalized text
        if "\r" in text or "\n" in text:
            assert "\r" not in processed, \
                f"Нормализованный текст содержит \\r: {processed!r}"

        # reverse_preprocess must reproduce the ORIGINAL bytes exactly
        restored = preprocessor.reverse_preprocess(processed, info)
        assert restored == text, (
            f"Побитовая обратимость нарушена.\n"
            f"  Оригинал:    {text!r}\n"
            f"  Восстанов.:  {restored!r}"
        )
        preview = text[:30].replace("\r", "\\r").replace("\n", "\\n")
        print(f"  '{preview}' → OK")

    print("✓ Пройден")


def test_context_manager():
    print("\n" + "=" * 60)
    print("ТЕСТ: Контекстное окно")
    print("=" * 60)

    tokens = list(range(20))

    cm = ContextManager(
        max_context_length=5, strategy=ContextStrategy.SLIDING_WINDOW
    )

    ctx = cm.get_context(tokens, 0)
    assert ctx == [], f"Ожидался [], получено {ctx}"

    ctx = cm.get_context(tokens, 3)
    assert ctx == [0, 1, 2], f"Ожидалось [0,1,2], получено {ctx}"

    ctx = cm.get_context(tokens, 10)
    assert ctx == [5, 6, 7, 8, 9], f"Ожидалось [5,6,7,8,9], получено {ctx}"

    print("  Скользящее окно: OK")

    cm_block = ContextManager(
        max_context_length=5, strategy=ContextStrategy.BLOCK
    )

    ctx = cm_block.get_context(tokens, 7)
    assert ctx == [5, 6], f"Ожидалось [5,6], получено {ctx}"

    print("  Блочная стратегия: OK")
    print("✓ Пройден")


def test_file_format():
    """Verify round-trip of metadata including the new preproc_info field."""
    print("\n" + "=" * 60)
    print("ТЕСТ: Формат файла .lmc (v4 с preproc_info)")
    print("=" * 60)

    metadata = FileMetadata(
        model_key="gpt2",
        model_name="gpt2",
        vocab_size=50257,
        max_context_length=1024,
        context_strategy="sliding_window",
        encoding="utf-8",
        num_tokens=100,
        original_size=500,
        original_hash="abc123",
        preproc_info={"mode": "mixed", "sequence": ["rn", "n", "r"]},
    )

    compressed_data = np.array([1, 2, 3, 4, 5], dtype=np.uint32)

    temp_path = os.path.join(tempfile.gettempdir(), "test_lmcodec.lmc")

    try:
        save_compressed(temp_path, metadata, compressed_data)
        print(f"  Записано: {Path(temp_path).stat().st_size} байт")

        loaded_meta, loaded_data = load_compressed(temp_path)

        assert loaded_meta.model_key == "gpt2"
        assert loaded_meta.vocab_size == 50257
        assert loaded_meta.original_hash == "abc123"
        assert list(loaded_data) == [1, 2, 3, 4, 5]
        # New field must round-trip intact
        assert loaded_meta.preproc_info == {
            "mode": "mixed",
            "sequence": ["rn", "n", "r"],
        }

        print("  Чтение/запись: OK")
        print("  preproc_info сохранён и восстановлен корректно")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    print("✓ Пройден")


def test_model_key_validation():
    """Bug 4 fix: bad model_key must raise in __init__, not during load."""
    print("\n" + "=" * 60)
    print("ТЕСТ: Ранняя валидация model_key")
    print("=" * 60)

    try:
        Compressor(model_key="nonexistent-model")
        raise AssertionError("Должна была сработать валидация")
    except ValueError as e:
        print(f"  Получено ожидаемое исключение: {str(e)[:80]}...")

    print("✓ Пройден")


def test_compression_lossless(model_key: str = "gpt2"):
    """End-to-end lossless round-trip using a real language model."""
    print("\n" + "=" * 60)
    print(f"ТЕСТ: Сжатие без потерь (модель: {model_key})")
    print("=" * 60)

    test_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In information theory, entropy measures the average uncertainty "
        "of a random variable. Shannon showed that the entropy of a source "
        "determines the theoretical limit of lossless compression. "
        "Large language models can serve as powerful probability estimators "
        "for arithmetic coding, potentially achieving compression ratios "
        "that surpass traditional methods like GZIP and BZIP2. "
        "This is a test of the LMCodec compression system. "
    ) * 3

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.txt")
        compressed_path = os.path.join(tmpdir, "output.lmc")
        restored_path = os.path.join(tmpdir, "restored.txt")

        # Write in binary mode to guarantee exact byte content — avoids
        # the OS adding platform-specific line endings on Windows.
        with open(input_path, "wb") as f:
            f.write(test_text.encode("utf-8"))

        original_size = os.path.getsize(input_path)
        print(f"  Исходный размер: {original_size} байт")

        compressor = Compressor(model_key=model_key)
        comp_results = compressor.compress(
            input_path, compressed_path, verbose=True
        )

        compressed_size = os.path.getsize(compressed_path)
        print(f"  Сжатый размер: {compressed_size} байт")
        print(f"  Коэффициент: {comp_results['ratio']:.4f}")
        print(f"  BPC: {comp_results['bpc']:.4f}")

        decompressor = Decompressor()
        decomp_results = decompressor.decompress(
            compressed_path, restored_path, verbose=True
        )

        # Compare raw bytes, not decoded strings — this is what "lossless" means
        with open(input_path, "rb") as f:
            original_bytes = f.read()
        with open(restored_path, "rb") as f:
            restored_bytes = f.read()

        assert original_bytes == restored_bytes, (
            f"Восстановленные байты НЕ совпадают!\n"
            f"  Оригинал: {len(original_bytes)} байт\n"
            f"  Восстан.: {len(restored_bytes)} байт"
        )
        assert decomp_results["is_lossless"], "SHA-256 хеши не совпадают!"

        print("\n  ✓ LOSSLESS ПОДТВЕРЖДЕНО (байт-в-байт)")

    print("✓ Пройден")


def test_compression_crlf_lossless(model_key: str = "gpt2"):
    """
    Bug 2 regression test: a file with \\r\\n line endings must
    round-trip byte-exactly. In the old code this silently corrupted
    to \\n and SHA-256 verification failed.
    """
    print("\n" + "=" * 60)
    print(f"ТЕСТ: Lossless для CRLF-файла (модель: {model_key})")
    print("=" * 60)

    # Windows-style line endings
    test_text = (
        "First line with CRLF.\r\n"
        "Second line here.\r\n"
        "Mixed line\rwith CR only.\r\n"
        "And a final LF\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.txt")
        compressed_path = os.path.join(tmpdir, "output.lmc")
        restored_path = os.path.join(tmpdir, "restored.txt")

        with open(input_path, "wb") as f:
            f.write(test_text.encode("utf-8"))

        compressor = Compressor(model_key=model_key)
        compressor.compress(input_path, compressed_path, verbose=False)

        decompressor = Decompressor()
        result = decompressor.decompress(
            compressed_path, restored_path, verbose=False
        )

        with open(input_path, "rb") as f:
            original_bytes = f.read()
        with open(restored_path, "rb") as f:
            restored_bytes = f.read()

        assert original_bytes == restored_bytes, (
            "CRLF не восстановлен побитово! Bug 2 регрессия."
        )
        assert result["is_lossless"]
        print("  ✓ \\r\\n, \\r и \\n восстановлены побитово")

    print("✓ Пройден")


def run_all_tests():
    print("=" * 60)
    print("LMCodec — Полный набор тестов")
    print("=" * 60)

    test_list_models()
    test_arithmetic_coding()
    test_preprocessing()
    test_context_manager()
    test_file_format()
    test_model_key_validation()

    print("\nСледующие тесты загружают модель GPT-2 (~500 MB).")
    print("Для пропуска нажмите Ctrl+C.\n")

    try:
        test_compression_lossless("gpt2")
        test_compression_crlf_lossless("gpt2")
    except KeyboardInterrupt:
        print("\n  Пропущено по запросу пользователя.")

    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
