from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class ChunkError(ValueError):
    """Raised for invalid chunk configuration or input."""


@dataclass(frozen=True)
class FileChunk:
    index: int
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)


def iter_file_chunks(path: Path, chunk_size: int) -> Iterator[FileChunk]:
    if chunk_size <= 0:
        raise ChunkError("chunk_size must be greater than zero")
    if not path.is_file():
        raise ChunkError(f"input file does not exist: {path}")
    with path.open("rb") as source:
        index = 0
        while data := source.read(chunk_size):
            yield FileChunk(index=index, data=data)
            index += 1


def count_file_chunks(path: Path, chunk_size: int) -> int:
    return sum(1 for _ in iter_file_chunks(path, chunk_size))
