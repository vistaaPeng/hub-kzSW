# CHANGELOG

## Unreleased

### Added

- Added `docs/project-retrospective.md`, summarizing the engineering decisions,
  project lessons, retrieval fixes, validation results, and next-step
  recommendations from the recent stabilization work.

### Changed

- Rebuilt README around the current pipeline, diagnostic workflow, retrieval
  improvements, and index rebuild notes.

## v0.3 (2026-07-08) - Retrieval diagnostics and stabilization

### Added

- Added a unified `RAGPipeline` so CLI, Web, and evaluation paths share the same
  retrieval and generation flow.
- Added `scripts/check_index.py` for index integrity validation.
- Added `scripts/diagnose_retrieval.py` for single-query retrieval diagnostics,
  including BM25, vector, and RRF rankings.
- Added runtime configuration support in `src/config.py` and `.env.example`.
- Added unit coverage for pipeline behavior, index checks, configuration,
  glossary cache behavior, diagnostics, BM25 tokenizer changes, and RRF boosts.

### Changed

- Centered parent-context expansion on the hit parent chunk instead of expanding
  from the beginning of a document.
- Cached glossary retrievers to avoid repeated reloads.
- Injected Rust keywords and Glossary terms into BM25 tokenization.
- Added phrase-token support for selected English technical phrases in BM25.
- Added lightweight RRF metadata boosts:
  - heading match boost
  - list-intent boost for `structured` chunks
  - code-intent boost for `code_unit` chunks
  - comparison-intent boost for `narrative` and `mixed` chunks
- Rebuilt indexes after tokenizer changes:
  - 6 sources
  - 6440 total chunks
  - 4589 child chunks indexed
  - 1851 parent chunks
  - 4589 BM25 docs
  - 374 Glossary terms

### Verified

- Full unit test suite passed:

```text
109 passed, 4 warnings
```

- Real-query smoke checks improved or remained stable for:
  - `rust有哪些关键字`
  - `如何写Rust函数示例`
  - `Rust所有权和借用有什么区别`
  - `trait是什么`
  - `生命周期是什么`

### Notes

- `diagnose_retrieval.py` currently reports base RRF breakdown. The final
  `HybridRetriever.search()` ranking also includes metadata boost; a future
  diagnostic update should expose base score, boost components, and final score.
- On Windows, use `PYTHONUTF8=1` when rebuilding indexes to avoid console
  encoding errors from Unicode log output.

## v0.2 (2026-07-07) - Web interface and evaluation workflow

### Added

- Added Streamlit Web interface with query, evaluation, settings, and history
  pages.
- Expanded FastAPI backend endpoints for stats, history, parsing, and
  evaluation.
- Added real-time evaluation progress and persistent evaluation results.
- Added one-command startup script via `scripts/start.ps1`.
- Added query history persistence to `logs/queries.jsonl`.
- Added evaluation question upload/replacement workflow.
- Added prompt and safety handling improvements.

### Changed

- Improved faithfulness and answer relevancy evaluation display.
- Returned all source chunks used by the LLM for better source visibility.
- Strengthened query rewrite fallback behavior.
- Increased available context for generation.
- Added frontend caching for file and API reads.

### Evaluation

Manual v0.2 evaluation snapshot:

| Metric | Score |
|---|---:|
| Context Precision | 0.88 |
| MRR | 1.00 |
| Faithfulness | 0.84 |
| Answer Relevancy | 0.79 |

## v0.1 (2026-07-06) - First complete version

### Added

- Added BM25 retrieval with jieba and rank-bm25.
- Added vector retrieval with `bge-base-zh-v1.5` and FAISS `IndexFlatIP`.
- Added handwritten RRF fusion.
- Added optional Cross-Encoder reranking with `bge-reranker-base`.
- Added parent-child chunking with morphology labels:
  `narrative`, `code_unit`, `specification`, `structured`, `note`, `mixed`.
- Added DeepSeek v4-flash generation.
- Added Glossary term injection into the generation prompt.
- Added LLM query rewriting into multiple standardized queries.
- Added evaluation metrics:
  - Context Precision
  - MRR
  - Faithfulness
  - Answer Relevancy

### Data

- Rust Chinese documentation corpus across multiple sources.
- 6440 chunks:
  - 1851 parent chunks
  - 4589 child chunks

### Evaluation

Initial mixed 60-question evaluation snapshot:

| Metric | Score |
|---|---:|
| Context Precision | 0.81 |
| MRR | 1.00 |
| Faithfulness | 0.83 |
| Answer Relevancy | 0.75 |
