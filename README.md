*This project has been created as part of the 42 curriculum by <login>.*

# RAG Against The Machine

A high-performance, modular Retrieval-Augmented Generation (RAG) system engineered to ingest, index, retrieve, and generate grounded answers for complex codebases and technical documentation (vLLM repository).

---

## Description

Large language models (LLMs) are frozen in time and lack visibility into private codebases, rapid framework iterations, and internal architecture updates. Rather than performing prohibitively expensive fine-tuning or retraining, Retrieval-Augmented Generation (RAG) supplements the model by providing relevant, verifiable ground-truth evidence at query time.

This project implements a complete RAG pipeline from scratch:
1. **Ingestion & Indexing**: Ingests Python source code and Markdown documentation from the target repository, dividing documents into cohesive syntactic units with exact character span tracking.
2. **Retrieval**: Leverages BM25 lexical search with sub-identifier tokenization (snake_case, camelCase, PascalCase splitting) to match precise technical identifiers and natural language queries.
3. **Augmentation & Generation**: Supplies retrieved context snippets to `Qwen/Qwen3-0.6B` within strict context budgets to generate factual, grounded answers.
4. **Evaluation**: Measures retrieval performance with Recall@k using a 1D character-level Intersection-over-Union (IoU $\ge 0.05$) metric.

---

## Instructions

### Prerequisites
- Python 3.10+
- [uv](https://astral.sh/uv) (fast Python package manager)

### Installation
Install project dependencies using `uv`:
```bash
uv sync
```

### Commands

Every command is invoked through Python Fire via `uv run python -m src <command>`:

#### 1. Ingest and Build the Index
Ingests the codebase from `data/raw/` and persists indices under `data/processed/`:
```bash
uv run python -m src index --max_chunk_size 2000
```

#### 2. Search a Single Query
Retrieve the top-$k$ sources for a question:
```bash
uv run python -m src search "How to configure OpenAI server?" --k 5
```

#### 3. Search a Dataset
Run batch retrieval over a question dataset and save structured results:
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

#### 4. Answer a Single Query
Generate a grounded natural language answer using `Qwen/Qwen3-0.6B`:
```bash
uv run python -m src answer "What is the default value of trust_remote_code in vLLM's LLM class constructor?" --k 5
```

#### 5. Generate Answers for a Dataset
Run batch answer generation across an entire search results file:
```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

#### 6. Evaluate Retrieval
Evaluate local search results against ground-truth datasets:
```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

---

## System Architecture

The pipeline consists of modular, decoupled components interacting through strict Pydantic schemas:

```
┌────────────────────────────────────────────────────────┐
│                   Target Repository                    │
│            (data/raw/vllm-0.10.1/: vllm/, docs/)        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Chunking & Indexing Stage                │
│  - CodeChunker: AST-based class/method/function units  │
│  - DocChunker: Markdown header & paragraph hierarchy   │
│  - Sub-identifier Tokenization & Context Injection     │
│  - Persists index state under data/processed/          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                    Retrieval Stage                     │
│  - BM25Okapi scoring over enriched token corpora       │
│  - Extracts exact [first_char_idx : last_char_idx]    │
│  - Outputs StudentSearchResults JSON                   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                   Generation Stage                     │
│  - Qwen/Qwen3-0.6B local language model                │
│  - Context grounding & prompt budgeting                │
│  - Outputs StudentSearchResultsAndAnswer JSON          │
└────────────────────────────────────────────────────────┘
```

---

## Chunking Strategy

Standard naive chunking (fixed token/character windows) fails on codebases because critical signatures, parameters, and assertions become severed across arbitrary boundaries. We implement two distinct domain-specific strategies:

### 1. Python Code Chunking (`CodeChunker`)
- **AST Parsing**: Deconstructs Python source files using the `ast` module into semantic code units:
  - Class definition headers, class attributes, and docstrings
  - Standalone functions and nested async functions
  - Methods within classes (with decorator offset preservation)
  - Module-level constants, configuration variables, and imports
- **Character Offset Precision**: Converts AST `lineno`, `col_offset`, `end_lineno`, and `end_col_offset` directly into 0-indexed character offsets against the raw source file.
- **Ceiling Enforcement**: The Moulinette strictly rejects any chunk exceeding 2000 characters. When a large function (e.g. complex PyTorch `forward()` pass) exceeds 1800 characters, it is smoothly partitioned using a sliding window of 1000 characters with 200-character overlap.
- **Lexical Context Enrichment**: The searchable representation of each chunk prepends breadcrumbs (`File: ... | Class: ... | Function: ... | Scope: ...`) without mutating the raw character coordinates.

### 2. Markdown / Text Chunking (`DocChunker`)
- **Header Hierarchy**: Decomposes documentation along header levels (`#`, `##`, `###`, `####`).
- **Block Integrity**: Keeps Markdown tables and fenced code blocks (` ``` `) intact within their corresponding section.
- **Paragraph Slicing**: Large documentation sections exceeding 1800 characters are partitioned along paragraph boundaries (`\n\n`).
- **Breadcrumb Injection**: Prepends document titles and section hierarchy to the search index text.

---

## Retrieval Method

Lexical search relies on the **BM25Okapi** probabilistic ranking algorithm.

### Identifier Tokenization
Technical queries and codebase files frequently differ in token representation (e.g. `tie_word_embeddings` vs `tie word embeddings`, or `HybridKVCacheCoordinator` vs `Hybrid KV Cache Coordinator`).
Our custom tokenizer (`src/indexing/tokenizer.py`):
1. Preserves the full compound token in lowercase.
2. Deconstructs snake_case, camelCase, PascalCase, and SCREAMING_SNAKE_CASE into individual word stems.
3. Splits alphanumeric transitions (e.g. `gemma2` $\rightarrow$ `gemma`, `2`).

### Ranking & Output
1. The query is tokenized and scored against the BM25 corpus.
2. The top-$k$ scoring documents are mapped back to their original `file_path`, `first_character_index`, and `last_character_index`.
3. Results are validated against the `StudentSearchResults` Pydantic model.

---

## Performance Analysis

Tested against the public evaluation suite (199 questions):

| Dataset | Total Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Minimum Required | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Docs Public** | 100 | 71.0% | 90.0% | **93.0%** | 96.0% | **$\ge$ 80%** | **PASSED ✅** |
| **Code Public** | 99 | 62.6% | 76.8% | **82.8%** | 88.9% | **$\ge$ 50%** | **PASSED ✅** |

- **Indexing Time**: ~10 seconds for the entire repository (well under the 5-minute requirement).
- **Retrieval Throughput**: ~0.25 seconds for 100 docs questions (~400 q/s), ~11 seconds for 99 code questions (well within the 90-second requirement for 200 questions).
- **Context Length**: 100% of generated chunks have `length < 2000` characters, completely satisfying Moulinette validation checks.

---

## Design Decisions

1. **AST-Based Chunking over Fixed Windowing**: Guarantees that method definitions, function signatures, and docstrings remain intact, directly aligning with code-level questions.
2. **Context Enrichment without Coordinate Mutation**: Injects file paths and class/function metadata into the searchable corpus text while preserving exact character indices in the original file.
3. **Separate Domain Indices**: Dedicated indexing for code and documentation accounts for the structural difference between natural language documentation and symbolic source code.
4. **Lazy Model Loading**: The generation LLM (`Qwen/Qwen3-0.6B`) is only loaded into memory when `answer` or `answer_dataset` is invoked, keeping `index`, `search`, and `evaluate` sub-second operations.

---

## Challenges Faced

1. **Character Offset Mapping**: Decorators in Python AST appear before `node.lineno`. Handled by explicitly scanning decorator tokens and leading `@` positions to ensure exact character alignment.
2. **Moulinette IoU Calculation**: The grader calculates $\text{IoU} = \frac{\text{Intersection}}{\text{Union}} > 0.05$. If chunks are too small, the union penalty drops IoU below 0.05; if chunks exceed 2000 characters, the output is rejected. The optimal chunk size was calibrated between 800 and 1800 characters.
3. **Compound Identifier Mismatch**: Queries containing natural phrasing failed to match compound identifiers in code (e.g. `VLLM_RPC_TIMEOUT`). Resolved via sub-identifier tokenization.

---

## Example Usage

```bash
# 1. Build index
uv run python -m src index --max_chunk_size 2000

# 2. Search single query
uv run python -m src search "What hardware platforms does vLLM support?" --k 3

# 3. Batch search dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 4. Generate answer for single query
uv run python -m src answer "What is the default value of trust_remote_code in vLLM's LLM class constructor?" --k 5

# 5. Local recall evaluation
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
```

---

## Resources & AI Disclosure

### References
- [BM25: The Probabilistic Relevance Framework (Robertson & Zaragoza, 2009)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [Qwen Series Models (Alibaba Cloud)](https://huggingface.co/Qwen)
- [Python ast — Abstract Syntax Trees Documentation](https://docs.python.org/3/library/ast.html)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)

### AI Usage Disclosure
AI assistance was utilized for:
- AST parsing architecture design for calculating exact character spans.
- Analyzing ground-truth source distributions across the public question datasets.
- Assisting with regex rules for sub-identifier tokenization (snake_case and camelCase splitting).
- Code formatting and adherence verification for `flake8` and `mypy` strict typing standards.
