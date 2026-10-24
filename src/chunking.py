"""Chunking logic for Python source code and Markdown documentation."""
import re
from typing import List, Tuple

# Chunk format: (file_path, first_char_index, last_char_index, text)
Chunk = Tuple[str, int, int, str]


def read_file(file_path: str) -> str:
    """Read file content safely as UTF-8.

    Args:
        file_path: Path to the target file.

    Returns:
        The text content of the file.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _get_split_points(text: str, is_python: bool) -> List[int]:
    """Find logical boundary points in the file.

    Args:
        text: File string content.
        is_python: True if file is Python, False otherwise.

    Returns:
        Sorted list of boundary character indices.
    """
    if is_python:
        # Split at function, class, and decorator lines
        pattern = re.compile(r"^(?:def |class |async def |@)", re.MULTILINE)
    else:
        # Split at Markdown headings (# Header)
        pattern = re.compile(r"^(?=#+\s)", re.MULTILINE)

    points = [m.start() for m in pattern.finditer(text)]
    if not points or points[0] != 0:
        points.insert(0, 0)
    points.append(len(text))
    return sorted(list(set(points)))


def _slice_large_block(
    file_path: str, text: str, start: int, end: int, max_size: int
) -> List[Chunk]:
    """Break a block larger than max_size along newline boundaries.

    Args:
        file_path: Path of the source file.
        text: Entire text content.
        start: Start character index.
        end: End character index.
        max_size: Maximum allowed chunk size.

    Returns:
        List of subdivided Chunk tuples.
    """
    chunks: List[Chunk] = []
    current_start = start

    while current_start < end:
        current_end = min(current_start + max_size, end)
        # Try to break at a newline so words or lines are not cut in half
        if current_end < end:
            last_nl = text.rfind("\n", current_start, current_end)
            if last_nl > current_start:
                current_end = last_nl + 1

        chunk_text = text[current_start:current_end]
        chunks.append((file_path, current_start, current_end, chunk_text))
        current_start = current_end

    return chunks


def chunk_file(file_path: str, text: str, max_size: int = 2000) -> List[Chunk]:
    """Split a file into chunks smaller than max_size characters.

    Args:
        file_path: Path of the source file.
        text: Content to chunk.
        max_size: Upper limit of chunk character length.

    Returns:
        List of generated chunks.
    """
    if not text.strip():
        return []

    is_python = file_path.endswith(".py")
    points = _get_split_points(text, is_python)

    chunks: List[Chunk] = []
    group_start = 0

    for i in range(len(points) - 1):
        block_start = points[i]
        block_end = points[i + 1]
        block_len = block_end - block_start

        # If a single block is already bigger than max_size, slice it
        if block_len > max_size:
            # Flush current accumulated group first
            if block_start > group_start:
                chunks.append(
                    (file_path, group_start, block_start, text[group_start:block_start])
                )
            chunks.extend(
                _slice_large_block(file_path, text, block_start, block_end, max_size)
            )
            group_start = block_end
        elif (block_end - group_start) > max_size:
            # Current group is full, close it and start a new one
            if block_start > group_start:
                chunks.append(
                    (file_path, group_start, block_start, text[group_start:block_start])
                )
            group_start = block_start

    # Flush remaining text
    if group_start < len(text):
        chunks.append((file_path, group_start, len(text), text[group_start:]))

    return chunks
