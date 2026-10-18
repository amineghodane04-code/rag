"""
Answer generation module using Qwen/Qwen3-0.6B.

Grounded answer generation using retrieved context snippets, adhering to
the token budget and pydantic data models defined in Chapter VI.3 and VI.4.
"""

import os
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class AnswerGenerator:
    """
    Generates grounded answers to questions using retrieved context snippets
    and a local language model (default: Qwen/Qwen3-0.6B).
    """

    DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

    def __init__(self, model_name: str = DEFAULT_MODEL, max_tokens: int = 150):
        """
        Initialize the generator. Model loading is deferred until first call.

        Args:
            model_name: Hugging Face model identifier.
            max_tokens: Maximum number of new tokens to generate.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        """Lazy load the model and tokenizer into memory."""
        if self.model is None or self.tokenizer is None:
            print(f"Loading generation model '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            # Use CPU by default, float32 or bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )
            self.model.eval()

    def load_snippet_text(
        self, source: Dict[str, Any], raw_root: str = "."
    ) -> str:
        """
        Extract the raw snippet text from file based on character indices.

        Args:
            source: Dict containing file_path, first_character_index, and last_character_index.
            raw_root: Project root to resolve file_path.

        Returns:
            str: The extracted text snippet.
        """
        fp = source.get("file_path", "")
        # Resolve path relative to project root
        actual_path = os.path.join(raw_root, fp)
        # Also try direct relative path if data/raw prefix is mapped
        if not os.path.exists(actual_path):
            actual_path = fp

        if not os.path.exists(actual_path):
            return ""

        try:
            with open(
                actual_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                content = f.read()
            first = source.get("first_character_index", 0)
            last = source.get("last_character_index", len(content))
            return content[first:last].strip()
        except Exception:
            return ""

    def generate_answer(
        self, question: str, sources: List[Dict[str, Any]], raw_root: str = "."
    ) -> str:
        """
        Generate a concise grounded answer given a question and retrieved sources.

        Args:
            question: The user or dataset question.
            sources: List of retrieved source dicts.
            raw_root: Project root for resolving source file paths.

        Returns:
            str: Generated natural language answer.
        """
        if not question or not question.strip():
            return "No question provided."

        self._ensure_loaded()
        assert self.model is not None and self.tokenizer is not None

        # Build context from top snippets (limit total context to ~3000 chars)
        context_parts = []
        total_len = 0
        for i, src in enumerate(sources[:3]):
            text = self.load_snippet_text(src, raw_root=raw_root)
            if text:
                part = f"[Source {i+1} ({src.get('file_path', '')})]:\n{text}"
                if total_len + len(part) > 3500:
                    break
                context_parts.append(part)
                total_len += len(part)

        context_str = (
            "\n\n".join(context_parts)
            if context_parts
            else "No relevant context found."
        )

        prompt = (
            f"You are a technical assistant answering questions about the vLLM codebase.\n"
            f"Context information from the repository:\n"
            f"---------------------\n"
            f"{context_str}\n"
            "Based strictly on the context above, provide a concise and factual "
            "answer to the question.\n"
            f"Question: {question.strip()}\n"
            f"Answer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        raw_answer = self.tokenizer.decode(
            gen_tokens, skip_special_tokens=True
        ).strip()
        # Clean answer to first paragraph
        lines = [
            line.strip() for line in raw_answer.splitlines() if line.strip()
        ]
        answer = lines[0] if lines else "No answer generated."
        return answer
