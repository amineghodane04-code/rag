import os
import sys
import ast
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

def split_identifier(identifier: str) -> List[str]:
    """Split identifier into constituent tokens while preserving the original."""
    tokens = [identifier]
    # Split camelCase / PascalCase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', identifier)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    parts = [p for p in s2.split('_') if len(p) > 1]
    tokens.extend(parts)
    return list(dict.fromkeys(tokens))

def extract_identifiers(text: str) -> List[str]:
    raw_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{2,}', text)
    result = []
    for tok in raw_tokens:
        result.extend(split_identifier(tok))
    return list(dict.fromkeys(result))

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize query or document for BM25 search."""
    # Extract words and identifiers
    words = re.findall(r'[a-zA-Z0-9_]+', text.lower())
    extra = []
    for w in words:
        if '_' in w:
            extra.extend([p for p in w.split('_') if len(p) > 1])
    return words + extra

class CodeChunker:
    def __init__(self, max_chunk_len: int = 1800, min_chunk_len: int = 100):
        self.max_chunk_len = max_chunk_len
        self.min_chunk_len = min_chunk_len

    def chunk_file(self, content: str, rel_path: str) -> List[Dict[str, Any]]:
        chunks = []
        line_starts = [0]
        for i, ch in enumerate(content):
            if ch == '\n':
                line_starts.append(i + 1)

        def node_to_char_range(node) -> Tuple[int, int]:
            if getattr(node, 'decorator_list', None):
                first_dec = node.decorator_list[0]
                start_line = first_dec.lineno
                line_start = line_starts[start_line - 1]
                line_end = line_starts[start_line] if start_line < len(line_starts) else len(content)
                line_str = content[line_start:line_end]
                at_pos = line_str.find('@')
                start = line_start + (at_pos if at_pos != -1 else first_dec.col_offset)
            else:
                start = line_starts[node.lineno - 1] + node.col_offset
            end = line_starts[node.end_lineno - 1] + node.end_col_offset
            return start, end

        try:
            tree = ast.parse(content)
        except Exception:
            # Fallback for unparseable python files
            return self._fallback_chunking(content, rel_path)

        def add_chunk(start: int, end: int, class_name: str = "", func_name: str = "", context_type: str = "code"):
            raw_text = content[start:end]
            if not raw_text.strip():
                return
            # If chunk is larger than max_chunk_len, split it
            if len(raw_text) > self.max_chunk_len:
                step = 1000
                overlap = 200
                pos = start
                while pos < end:
                    sub_end = min(pos + step, end)
                    # Try to align sub_end to newline if possible
                    nl = content.rfind('\n', pos, sub_end)
                    if nl > pos + 300 and sub_end < end:
                        sub_end = nl
                    sub_raw = content[pos:sub_end]
                    if sub_raw.strip():
                        enriched = f"File: {rel_path} | Class: {class_name} | Function: {func_name} | Type: {context_type}\n{sub_raw}"
                        chunks.append({
                            "file_path": f"data/raw/vllm-0.10.1/{rel_path}",
                            "first_character_index": pos,
                            "last_character_index": sub_end,
                            "raw_content": sub_raw,
                            "enriched_text": enriched,
                            "class_name": class_name,
                            "func_name": func_name
                        })
                    if sub_end >= end:
                        break
                    pos = max(pos + 1, sub_end - overlap)
            else:
                enriched = f"File: {rel_path} | Class: {class_name} | Function: {func_name} | Type: {context_type}\n{raw_text}"
                chunks.append({
                    "file_path": f"data/raw/vllm-0.10.1/{rel_path}",
                    "first_character_index": start,
                    "last_character_index": end,
                    "raw_content": raw_text,
                    "enriched_text": enriched,
                    "class_name": class_name,
                    "func_name": func_name
                })

        # Process AST body
        prev_end = 0
        for node in tree.body:
            node_start, node_end = node_to_char_range(node)
            
            # Module level comments/constants before this node
            if node_start > prev_end:
                gap = content[prev_end:node_start].strip()
                if len(gap) > 50:
                    add_chunk(prev_end, node_start, "", "", "module_global")

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_chunk(node_start, node_end, "", node.name, "function")
            elif isinstance(node, ast.ClassDef):
                # Header of class (docstring + attributes before methods)
                first_method_start = None
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start, _ = node_to_char_range(item)
                        first_method_start = m_start
                        break
                
                if first_method_start:
                    add_chunk(node_start, first_method_start, node.name, "__class_def__", "class_header")
                else:
                    add_chunk(node_start, node_end, node.name, "__class_def__", "class_def")

                # Methods inside class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start, m_end = node_to_char_range(item)
                        add_chunk(m_start, m_end, node.name, item.name, "method")
            else:
                add_chunk(node_start, node_end, "", "", "statement")
            
            prev_end = node_end

        if prev_end < len(content):
            rem = content[prev_end:].strip()
            if len(rem) > 50:
                add_chunk(prev_end, len(content), "", "", "module_footer")

        return chunks

    def _fallback_chunking(self, content: str, rel_path: str) -> List[Dict[str, Any]]:
        chunks = []
        step = 1200
        overlap = 200
        pos = 0
        while pos < len(content):
            end = min(pos + step, len(content))
            nl = content.rfind('\n', pos, end)
            if nl > pos + 300 and end < len(content):
                end = nl
            raw = content[pos:end]
            if raw.strip():
                chunks.append({
                    "file_path": f"data/raw/vllm-0.10.1/{rel_path}",
                    "first_character_index": pos,
                    "last_character_index": end,
                    "raw_content": raw,
                    "enriched_text": f"File: {rel_path}\n{raw}"
                })
            if end >= len(content):
                break
            pos = max(pos + 1, end - overlap)
        return chunks

class DocChunker:
    def __init__(self, max_chunk_len: int = 1800):
        self.max_chunk_len = max_chunk_len

    def chunk_file(self, content: str, rel_path: str) -> List[Dict[str, Any]]:
        chunks = []
        # Find markdown headers
        header_pattern = re.compile(r'^(#{1,4}\s+.+)$', re.MULTILINE)
        matches = list(header_pattern.finditer(content))

        if not matches:
            # Fallback sliding window
            step = 1200
            overlap = 200
            pos = 0
            while pos < len(content):
                end = min(pos + step, len(content))
                nl = content.rfind('\n', pos, end)
                if nl > pos + 300 and end < len(content):
                    end = nl
                raw = content[pos:end]
                if raw.strip():
                    chunks.append({
                        "file_path": f"data/raw/vllm-0.10.1/{rel_path}",
                        "first_character_index": pos,
                        "last_character_index": end,
                        "raw_content": raw,
                        "enriched_text": f"File: {rel_path}\n{raw}"
                    })
                if end >= len(content):
                    break
                pos = max(pos + 1, end - overlap)
            return chunks

        # Extract title from first header
        doc_title = matches[0].group(1).lstrip('#').strip()

        sections = []
        # Content before first header
        if matches[0].start() > 0:
            prefix = content[:matches[0].start()].strip()
            if len(prefix) > 30:
                sections.append((0, matches[0].start(), "Introduction"))

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            title = m.group(1).lstrip('#').strip()
            sections.append((start, end, title))

        for start, end, title in sections:
            sec_text = content[start:end]
            if not sec_text.strip():
                continue
            if len(sec_text) <= self.max_chunk_len:
                enriched = f"File: {rel_path} | Doc: {doc_title} | Section: {title}\n{sec_text}"
                chunks.append({
                    "file_path": f"data/raw/vllm-0.10.1/{rel_path}",
                    "first_character_index": start,
                    "last_character_index": end,
                    "raw_content": sec_text,
                    "enriched_text": enriched
                })
            else:
                # Split large section by paragraphs
                step = 1200
                overlap = 200
                pos = start
                while pos < end:
                    sub_end = min(pos + step, end)
                    p_break = content.rfind('\n\n', pos, sub_end)
                    if p_break > pos + 300 and sub_end < end:
                        sub_end = p_break
                    sub_raw = content[pos:sub_end]
                    if sub_raw.strip():
                        enriched = f"File: {rel_path} | Doc: {doc_title} | Section: {title}\n{sub_raw}"
                        chunks.append({
                            "file_path": f"data/raw/vllm-0.10.1/{rel_path}",
                            "first_character_index": pos,
                            "last_character_index": sub_end,
                            "raw_content": sub_raw,
                            "enriched_text": enriched
                        })
                    if sub_end >= end:
                        break
                    pos = max(pos + 1, sub_end - overlap)

        return chunks

print("Chunker classes defined.")
