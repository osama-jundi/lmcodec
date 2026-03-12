import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC_BYTES = b"LLMZ"
FORMAT_VERSION = 3


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


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_compressed(
    filepath: str,
    metadata: FileMetadata,
    compressed_data: np.ndarray,
) -> int:
    meta_json = json.dumps(metadata.__dict__, ensure_ascii=False).encode("utf-8")
    compressed_bytes = compressed_data.tobytes()

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
        metadata = FileMetadata(**meta_dict)

        compressed_len = struct.unpack("<I", f.read(4))[0]
        compressed_bytes = f.read(compressed_len)
        compressed_data = np.frombuffer(compressed_bytes, dtype=np.uint32).copy()

    return metadata, compressed_data
