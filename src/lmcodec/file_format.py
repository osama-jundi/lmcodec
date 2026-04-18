import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

MAGIC_BYTES = b"LMCO"
# Format version bumped to 4 because metadata now carries preproc_info.
# Older (v3) files remain readable — preproc_info defaults to None.
FORMAT_VERSION = 4


@dataclass
class FileMetadata:
    model_key: str = ""
    model_name: str = ""
    vocab_size: int = 0
    max_context_length: int = 0
    context_strategy: str = "sliding_window"
    encoding: str = "utf-8"
    num_tokens: int = 0
    original_size: int = 0
    original_hash: str = ""
    # New in v4 — information needed to exactly reverse preprocessing.
    # For files produced by v3 this field is absent in JSON and stays None.
    preproc_info: Optional[dict] = None


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_compressed(
    filepath: str,
    metadata: FileMetadata,
    compressed_data: np.ndarray,
) -> int:
    meta_json = json.dumps(metadata.__dict__, ensure_ascii=False).encode("utf-8")
    # Bug 7 note: compressed_data is uint32 native byte order. We write it as-is.
    # All mainstream platforms are little-endian, but we document this assumption.
    compressed_bytes = compressed_data.astype("<u4").tobytes()

    with open(filepath, "wb") as f:
        f.write(MAGIC_BYTES)
        f.write(struct.pack("<H", FORMAT_VERSION))

        f.write(struct.pack("<I", len(meta_json)))
        f.write(meta_json)

        f.write(struct.pack("<I", len(compressed_bytes)))
        f.write(compressed_bytes)

    return Path(filepath).stat().st_size


def load_compressed(filepath: str) -> tuple[FileMetadata, np.ndarray]:
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC_BYTES:
            raise ValueError(
                f"Неверный формат файла: ожидалось {MAGIC_BYTES!r}, "
                f"получено {magic!r}"
            )

        version = struct.unpack("<H", f.read(2))[0]
        if version > FORMAT_VERSION:
            raise ValueError(
                f"Неподдерживаемая версия формата: {version} "
                f"(максимальная: {FORMAT_VERSION})"
            )

        meta_len = struct.unpack("<I", f.read(4))[0]
        meta_json = f.read(meta_len).decode("utf-8")
        meta_dict = json.loads(meta_json)
        # Tolerate fields that the current class does not know about.
        known_fields = set(FileMetadata.__dataclass_fields__.keys())
        meta_dict = {k: v for k, v in meta_dict.items() if k in known_fields}
        metadata = FileMetadata(**meta_dict)

        compressed_len = struct.unpack("<I", f.read(4))[0]
        compressed_bytes = f.read(compressed_len)
        # Explicit little-endian read so compressed data is portable across
        # architectures (fixes the theoretical endian issue).
        compressed_data = np.frombuffer(compressed_bytes, dtype="<u4").astype(np.uint32).copy()

    return metadata, compressed_data
