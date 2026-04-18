"""
Smoke tests for lmcodec — quick sanity checks that do NOT load a language model.
Run: python tests/test_smoke.py
Or:  pytest tests/test_smoke.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Allow running this file directly from repo root without pip install -e .
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_imports():
    print("1. Проверка импортов...")
    from lmcodec import Compressor, Decompressor, ContextStrategy, list_available_models
    from lmcodec.models import create_model, BaseProbabilityModel
    from lmcodec.arithmetic import ArithmeticEncoder, ArithmeticDecoder
    from lmcodec.preprocessing import Preprocessor
    from lmcodec.context import ContextManager
    from lmcodec.file_format import FileMetadata
    from lmcodec.utils import stabilize_probabilities
    print("   ✓ Все модули импортированы успешно (включая utils.py)")


def test_models_registry():
    print("2. Проверка реестра моделей...")
    from lmcodec import list_available_models
    models = list_available_models()
    assert len(models) == 5, f"Ожидалось 5 моделей, получено {len(models)}"
    for m in models:
        print(f"   ✓ {m['key']:15s} → {m['display_name']}")
    print("   ✓ Все 5 моделей зарегистрированы")


def test_arithmetic():
    print("3. Проверка арифметического кодирования...")
    from lmcodec.arithmetic import ArithmeticEncoder, ArithmeticDecoder

    np.random.seed(42)
    vocab = 50
    symbols = [3, 7, 1, 9, 4, 2, 8, 0, 5, 6]

    probs_list = [
        np.random.dirichlet(np.ones(vocab)).astype(np.float64)
        for _ in symbols
    ]

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
    print("4. Проверка формата файла .lmc (v4, с preproc_info)...")
    from lmcodec.file_format import FileMetadata, save_compressed, load_compressed

    # Build metadata with the new preproc_info field (Bug 2 fix)
    meta = FileMetadata(
        model_key="gpt2",
        model_name="gpt2",
        vocab_size=50257,
        max_context_length=1024,
        num_tokens=10,
        original_size=50,
        original_hash="test",
        preproc_info={"mode": "uniform", "ending": "n"},
    )
    data = np.array([1, 2, 3], dtype=np.uint32)

    temp_path = os.path.join(tempfile.gettempdir(), "test_lmcodec.lmc")
    try:
        save_compressed(temp_path, meta, data)
        m2, d2 = load_compressed(temp_path)
        assert m2.model_key == "gpt2"
        assert list(d2) == [1, 2, 3]
        # Confirm the new field round-trips through JSON
        assert m2.preproc_info == {"mode": "uniform", "ending": "n"}
        print("   ✓ Запись/чтение .lmc OK (preproc_info сохранён)")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_stabilize_probabilities():
    print("5. Проверка функции stabilize_probabilities (utils.py)...")
    from lmcodec.utils import stabilize_probabilities

    # Case 1: zero probabilities get a minimum floor
    probs = np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float64)
    result = stabilize_probabilities(probs, vocab_size=4)
    assert np.all(result > 0), "Нулевые вероятности не заменены"
    assert abs(result.sum() - 1.0) < 1e-9, f"Сумма не 1: {result.sum()}"

    # Case 2: array shorter than vocab_size gets padded
    probs = np.array([0.5, 0.5], dtype=np.float64)
    result = stabilize_probabilities(probs, vocab_size=5)
    assert len(result) == 5, f"Длина не 5: {len(result)}"
    assert abs(result.sum() - 1.0) < 1e-9

    # Case 3: array longer than vocab_size gets truncated
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
    result = stabilize_probabilities(probs, vocab_size=4)
    assert len(result) == 4, f"Длина не 4: {len(result)}"
    assert abs(result.sum() - 1.0) < 1e-9

    print("   ✓ Стабилизация работает для всех трёх случаев")


def test_model_key_validation():
    print("6. Проверка ранней валидации model_key...")
    from lmcodec import Compressor

    # Bug 4 fix: unknown key should raise immediately in __init__,
    # not later during model loading.
    try:
        Compressor(model_key="gtp2")  # deliberate typo
        assert False, "Должна была сработать валидация"
    except ValueError as e:
        assert "Неизвестная модель" in str(e) or "gtp2" in str(e)
        print("   ✓ Некорректный model_key отклонён на этапе __init__")


def main():
    print("=" * 60)
    print("LMCodec — Smoke Test (без загрузки моделей)")
    print("=" * 60)

    test_imports()
    test_models_registry()
    test_arithmetic()
    test_file_format()
    test_stabilize_probabilities()
    test_model_key_validation()

    print("\n" + "=" * 60)
    print("ВСЕ БАЗОВЫЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("=" * 60)
    print("\nДля полного теста (с загрузкой модели) запустите:")
    print("  python tests/test_compression.py")


if __name__ == "__main__":
    main()
