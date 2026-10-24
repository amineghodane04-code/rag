"""Answer generation module using the local Qwen/Qwen3-0.6B model."""
import json
import os
import re
from typing import Any, List, Optional, Tuple

from tqdm import tqdm

from src.chunking import read_file
from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARS = 4000
MAX_NEW_TOKENS = 256

_TOKENIZER: Optional[Any] = None
_MODEL: Optional[Any] = None


def get_model() -> Tuple[Any, Any]:
    """Load the model and tokenizer only once when needed.

    Returns:
        Tuple of (tokenizer, model).
    """
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading default model {MODEL_NAME}...")
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    _MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")
    print("Model loaded.")
    return _TOKENIZER, _MODEL


def build_context(sources: List[MinimalSource]) -> str:
    """Extract snippet texts from files within character budget.

    Args:
        sources: List of MinimalSource locations.

    Returns:
        Formatted context string.
    """
    snippets: List[str] = []
    total_len = 0

    for s in sources:
        try:
            full_text = read_file(s.file_path)
            snippet = full_text[s.first_character_index:s.last_character_index]
        except Exception:
            continue

        if total_len + len(snippet) > MAX_CONTEXT_CHARS:
            break

        snippets.append(f"Source ({s.file_path}):\n{snippet}")
        total_len += len(snippet)

    return "\n\n".join(snippets)


def generate_answer(question: str, sources: List[MinimalSource]) -> str:
    """Generate a grounded answer for a single question.

    Args:
        question: Question to answer.
        sources: Retrieved source references.

    Returns:
        Answer string synthesized by the LLM.
    """
    if not sources:
        return "No sources found."

    context = build_context(sources)
    if not context.strip():
        return "No sources found."

    tokenizer, model = get_model()

    messages = [
        {
            "role": "system",
            "content": (
                "Answer using ONLY the provided context. "
                "If the answer is not in the context, say you do not know."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    import torch

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    raw_answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Strip any thinking tags if present
    cleaned = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
    return cleaned if cleaned else "No answer generated."


def run_answer_dataset(
    student_search_results_path: str,
    save_directory: str = "data/output/search_results_and_answer",
) -> None:
    """Generate answers for a dataset of search results.

    Args:
        student_search_results_path: Path to StudentSearchResults JSON.
        save_directory: Output directory for answers JSON.
    """
    if not os.path.exists(student_search_results_path):
        print(f"Error: file not found '{student_search_results_path}'")
        return

    try:
        with open(student_search_results_path, "r", encoding="utf-8") as f:
            search_data = StudentSearchResults.model_validate(json.load(f))
    except Exception as e:
        print(f"Error reading search results: {e}")
        return

    print(f"Loaded {len(search_data.search_results)} questions ...")
    answers: List[MinimalAnswer] = []

    for item in tqdm(search_data.search_results, desc="Generating answers"):
        ans = generate_answer(item.question, item.retrieved_sources)
        answers.append(
            MinimalAnswer(
                question_id=item.question_id,
                question=item.question,
                retrieved_sources=item.retrieved_sources,
                answer=ans,
            )
        )

    output = StudentSearchResultsAndAnswer(search_results=answers, k=search_data.k)
    os.makedirs(save_directory, exist_ok=True)
    out_file = os.path.join(save_directory, os.path.basename(student_search_results_path))
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    print(f"Saved student_search_results_and_answer to {out_file}")
