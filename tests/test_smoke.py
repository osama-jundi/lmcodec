import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    print("1. Проверка импортов...")
    from lmcodec import Compressor, Decompressor, ContextStrategy, list_available_models
    from lmcodec.models import create_model, BaseProbabilityModel
    from lmcodec.arithmetic import ArithmeticEncoder, ArithmeticDecoder
    from lmcodec.preprocessing import Preprocessor
    from lmcodec.context import ContextManager
    from lmcodec.file_format import FileMetadata
    print("   ✓ Все модули импортированы успешно")


def test_models_registry():
    print("2. Проверка реестра моделей...")
    from lmcodec import list_available_models
    models = list_available_models()
    assert len(models) == 5, f"Ожидалось 5 моделей, получено {len(models)}"
    for m in models:
        print(f"   ✓ {m['key']:15s} → {m['display_name']}")
    print(f"   ✓ Все 5 моделей зарегистрированы")


def test_arithmetic():
    print("3. Проверка арифметического кодирования...")
    import numpy as np
    from src.lmcodec.arithmetic import ArithmeticEncoder, ArithmeticDecoder

    np.random.seed(42)
    vocab = 50
    symbols = [3, 7, 1, 9, 4, 2, 8, 0, 5, 6]

    probs_list = [np.random.dirichlet(np.ones(vocab)).astype(np.float64) for _ in symbols]

    enc = ArithmeticEncoder()
    enc.create_encoder()
    for i in range(len(symbols) - 1, -1, -1):
        p = probs_list[i].astype(np.float32)
        p /= p.sum()
        enc.encode_symbol(symbols[i], p)
    compressed = enc.finish()

    dec = ArithmeticDecoder(compressed)
    decoded = []
    for i in range(len(symbols)):
        p = probs_list[i].astype(np.float32)
        p /= p.sum()
        decoded.append(dec.decode_symbol(p))

    assert symbols == decoded, f"Mismatch: {symbols} != {decoded}"
    print(f"   ✓ Кодирование/декодирование: {len(symbols)} символов OK")


def test_file_format():
    print("4. Проверка формата файла .llmz...")
    import os
    import tempfile
    import numpy as np
    from src.lmcodec.file_format import FileMetadata, save_compressed, load_compressed

    meta = FileMetadata(
        model_key="gpt2", model_name="gpt2",
        vocab_size=50257, max_context_length=1024,
        num_tokens=10, original_size=50, original_hash="test",
    )
    data = np.array([1, 2, 3], dtype=np.uint32)

    temp_path = os.path.join(tempfile.gettempdir(), "test_llmzip.llmz")
    try:
        save_compressed(temp_path, meta, data)
        m2, d2 = load_compressed(temp_path)
        assert m2.model_key == "gpt2"
        assert list(d2) == [1, 2, 3]
        print("   ✓ Запись/чтение .llmz OK")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    print("=" * 50)
    print("LLMZip v2.0 — Smoke Test")
    print("=" * 50)

    test_imports()
    test_models_registry()
    test_arithmetic()
    test_file_format()

    print("\n" + "=" * 50)
    print("✓ ВСЕ БАЗОВЫЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("=" * 50)

    print("\nДля полного теста (с загрузкой модели) запустите:")
    print("  python tests/test_compression.py")


if __name__ == "__main__":
    main()
