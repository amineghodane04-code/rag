import re
from typing import List, Dict, Any


class DocChunker:
    """
    Markdown and documentation chunker.
    Splits documents into coherent sections using Markdown heading hierarchy (#, ##, ###),
    paragraph boundaries, and tables.
    Preserves exact character offsets [first_character_index, last_character_index],
    strictly guaranteeing chunk length <= max_chunk_len (< 2000 chars).
    """

    def __init__(
        self,
        max_chunk_len: int = 1800,
        min_chunk_len: int = 40,
        prefix_path: str = "data/raw/vllm-0.10.1",
    ):
        self.max_chunk_len = max_chunk_len
        self.min_chunk_len = min_chunk_len
        self.prefix_path = prefix_path

    def chunk_file(self, content: str, rel_path: str) -> List[Dict[str, Any]]:
        if rel_path.endswith(".md"):
            return self._chunk_markdown(content, rel_path)
        else:
            return self._chunk_text(content, rel_path)

    def _chunk_markdown(
        self, content: str, rel_path: str
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        full_file_path = f"{self.prefix_path}/{rel_path}"
        header_pattern = re.compile(r"^(#{1,4}\s+.+)$", re.MULTILINE)
        matches = list(header_pattern.finditer(content))

        if not matches:
            return self._chunk_text(content, rel_path)

        doc_title = matches[0].group(1).lstrip("#").strip()
        sections = []

        if matches[0].start() > 0:
            prefix = content[: matches[0].start()]
            if len(prefix.strip()) >= self.min_chunk_len:
                sections.append((0, matches[0].start(), "Overview"))

        for i, m in enumerate(matches):
            start = m.start()
            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(content)
            )
            title = m.group(1).lstrip("#").strip()
            sections.append((start, end, title))

        for start, end, title in sections:
            sec_text = content[start:end]
            if len(sec_text.strip()) < self.min_chunk_len:
                continue

            if len(sec_text) <= self.max_chunk_len:
                enriched = (
                    f"File: {rel_path} | Document: {doc_title} | Section: {title}\n"
                    f"{sec_text}"
                )
                chunks.append(
                    {
                        "file_path": full_file_path,
                        "first_character_index": start,
                        "last_character_index": end,
                        "raw_content": sec_text,
                        "enriched_text": enriched,
                        "section_title": title,
                        "doc_title": doc_title,
                    }
                )
            else:
                # Split large section along paragraph boundaries (\n\n)
                step = 1200
                overlap = 200
                pos = start
                while pos < end:
                    sub_end = min(pos + step, end)
                    p_break = content.rfind("\n\n", pos, sub_end)
                    if p_break > pos + 300 and sub_end < end:
                        sub_end = p_break
                    sub_raw = content[pos:sub_end]
                    if len(sub_raw.strip()) >= self.min_chunk_len:
                        enriched = (
                            f"File: {rel_path} | Document: {doc_title} | Section: {title}\n"
                            f"{sub_raw}"
                        )
                        chunks.append(
                            {
                                "file_path": full_file_path,
                                "first_character_index": pos,
                                "last_character_index": sub_end,
                                "raw_content": sub_raw,
                                "enriched_text": enriched,
                                "section_title": title,
                                "doc_title": doc_title,
                            }
                        )
                    if sub_end >= end:
                        break
                    pos = max(pos + 1, sub_end - overlap)

        return chunks

    def _chunk_text(self, content: str, rel_path: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        full_file_path = f"{self.prefix_path}/{rel_path}"
        step = 1200
        overlap = 200
        pos = 0

        while pos < len(content):
            end = min(pos + step, len(content))
            nl = content.rfind("\n", pos, end)
            if nl > pos + 300 and end < len(content):
                end = nl
            raw = content[pos:end]
            if len(raw.strip()) >= self.min_chunk_len:
                enriched = f"File: {rel_path}\n{raw}"
                chunks.append(
                    {
                        "file_path": full_file_path,
                        "first_character_index": pos,
                        "last_character_index": end,
                        "raw_content": raw,
                        "enriched_text": enriched,
                        "section_title": "",
                        "doc_title": rel_path,
                    }
                )
            if end >= len(content):
                break
            pos = max(pos + 1, end - overlap)

        return chunks
