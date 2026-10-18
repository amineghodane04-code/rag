import ast
from typing import List, Dict, Any, Tuple


class CodeChunker:
    """
    AST-based Python code chunker.
    Chunks code by function, method, class header, and module-level constants.
    Preserves exact character offsets [first_character_index, last_character_index]
    in the source file, strictly guaranteeing chunk length <= max_chunk_len (< 2000 chars).
    """

    def __init__(
        self,
        max_chunk_len: int = 1800,
        min_chunk_len: int = 50,
        prefix_path: str = "data/raw/vllm-0.10.1",
    ):
        self.max_chunk_len = max_chunk_len
        self.min_chunk_len = min_chunk_len
        self.prefix_path = prefix_path

    def chunk_file(self, content: str, rel_path: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        # Precompute line start character positions
        line_starts = [0]
        for i, ch in enumerate(content):
            if ch == "\n":
                line_starts.append(i + 1)

        def node_to_char_range(node: Any) -> Tuple[int, int]:
            # Account for decorators that precede the function/class line
            if (
                getattr(node, "decorator_list", None)
                and len(node.decorator_list) > 0
            ):
                first_dec = node.decorator_list[0]
                start_line = first_dec.lineno
                line_start = line_starts[start_line - 1]
                line_end = (
                    line_starts[start_line]
                    if start_line < len(line_starts)
                    else len(content)
                )
                line_str = content[line_start:line_end]
                at_pos = line_str.find("@")
                start = line_start + (
                    at_pos if at_pos != -1 else first_dec.col_offset
                )
            else:
                start = line_starts[node.lineno - 1] + node.col_offset

            end = line_starts[node.end_lineno - 1] + node.end_col_offset
            return start, end

        try:
            tree = ast.parse(content)
        except Exception:
            return self._fallback_chunking(content, rel_path)

        def add_chunk(
            start: int,
            end: int,
            class_name: str = "",
            func_name: str = "",
            context_type: str = "code",
        ) -> None:
            raw_text = content[start:end]
            if len(raw_text.strip()) < self.min_chunk_len:
                return

            full_file_path = f"{self.prefix_path}/{rel_path}"

            # If chunk exceeds max_chunk_len, split it safely with sliding
            # window
            if len(raw_text) > self.max_chunk_len:
                step = 1000
                overlap = 200
                pos = start
                while pos < end:
                    sub_end = min(pos + step, end)
                    # Align to newline when feasible
                    nl = content.rfind("\n", pos, sub_end)
                    if nl > pos + 300 and sub_end < end:
                        sub_end = nl
                    sub_raw = content[pos:sub_end]
                    if len(sub_raw.strip()) >= 30:
                        header = (
                            f"File: {rel_path} | Class: {class_name} | "
                            f"Function: {func_name} | Scope: {context_type}\n"
                        )
                        enriched = f"{header}{sub_raw}"
                        chunks.append(
                            {
                                "file_path": full_file_path,
                                "first_character_index": pos,
                                "last_character_index": sub_end,
                                "raw_content": sub_raw,
                                "enriched_text": enriched,
                                "class_name": class_name,
                                "func_name": func_name,
                                "context_type": context_type,
                            }
                        )
                    if sub_end >= end:
                        break
                    pos = max(pos + 1, sub_end - overlap)
            else:
                header = (
                    f"File: {rel_path} | Class: {class_name} | "
                    f"Function: {func_name} | Scope: {context_type}\n"
                )
                enriched = f"{header}{raw_text}"
                chunks.append(
                    {
                        "file_path": full_file_path,
                        "first_character_index": start,
                        "last_character_index": end,
                        "raw_content": raw_text,
                        "enriched_text": enriched,
                        "class_name": class_name,
                        "func_name": func_name,
                        "context_type": context_type,
                    }
                )

        prev_end = 0
        for node in tree.body:
            node_start, node_end = node_to_char_range(node)

            # Capture module-level comments/constants/imports in gaps
            if node_start > prev_end:
                gap = content[prev_end:node_start].strip()
                if len(gap) >= self.min_chunk_len:
                    add_chunk(prev_end, node_start, "", "", "module_global")

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_chunk(node_start, node_end, "", node.name, "function")
            elif isinstance(node, ast.ClassDef):
                # Separate class definition header from its methods
                first_method_start = None
                for item in node.body:
                    if isinstance(
                        item, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        m_start, _ = node_to_char_range(item)
                        first_method_start = m_start
                        break

                if first_method_start:
                    add_chunk(
                        node_start,
                        first_method_start,
                        node.name,
                        "__class_def__",
                        "class_header",
                    )
                else:
                    add_chunk(
                        node_start,
                        node_end,
                        node.name,
                        "__class_def__",
                        "class_def",
                    )

                # Methods inside class
                for item in node.body:
                    if isinstance(
                        item, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        m_start, m_end = node_to_char_range(item)
                        add_chunk(
                            m_start, m_end, node.name, item.name, "method"
                        )
            else:
                # Top level statements (assignments, constants, dataclasses)
                add_chunk(node_start, node_end, "", "", "statement")

            prev_end = node_end

        if prev_end < len(content):
            rem = content[prev_end:].strip()
            if len(rem) >= self.min_chunk_len:
                add_chunk(prev_end, len(content), "", "", "module_footer")

        return chunks

    def _fallback_chunking(
        self, content: str, rel_path: str
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        step = 1200
        overlap = 200
        pos = 0
        full_file_path = f"{self.prefix_path}/{rel_path}"
        while pos < len(content):
            end = min(pos + step, len(content))
            nl = content.rfind("\n", pos, end)
            if nl > pos + 300 and end < len(content):
                end = nl
            raw = content[pos:end]
            if raw.strip():
                chunks.append(
                    {
                        "file_path": full_file_path,
                        "first_character_index": pos,
                        "last_character_index": end,
                        "raw_content": raw,
                        "enriched_text": f"File: {rel_path}\n{raw}",
                        "class_name": "",
                        "func_name": "",
                        "context_type": "fallback",
                    }
                )
            if end >= len(content):
                break
            pos = max(pos + 1, end - overlap)
        return chunks
