# Evaluation scripts

Every system of the paper on HotpotQA, MuSiQue, G-Medical and Contract QA, with the same data
(`../evaluation_data/`), the same LLM, the same embedding model, the same answer step and the same scorers. The
baselines are vendored from their official code with minimal, documented patches; UDH is `../nkg/` driven by `unh/`.

| Folder | System | Upstream | Ingestion unit (HotpotQA / MuSiQue) |
|---|---|---|---|
| `hipporag2/` | HippoRAG 2 | OSU-NLP-Group/HippoRAG @ 1438aba | one doc per passage (`title\ntext`) |
| `memgraphrag/` | MemGraphRAG | XMUDeepLIT/MemGraphRAG @ cd6fabd | 256/32-token windows over the whole corpus `.txt` (upstream behaviour; passages shuffled) |
| `hygrag/` | HyGRAG (`HKGraphTreeLSH`) | zjunet/HyGRAG @ fa1a57e (DIGIMON fork) | one doc per passage, never chunked across passages |
| `ms_graphrag/` | Microsoft GraphRAG 3.2.0 (stock settings, local search) | PyPI `graphrag==3.2.0`, unmodified | the whole corpus `.txt` as one document, stock 1200/100-token chunks |
| `hypergraphrag/` | HyperGraphRAG | LHRLAB/HyperGraphRAG @ d587cdf | one doc per passage, chunked within the doc |
| `logicrag/` | LogicRAG | chensyCN/LogicRAG @ 72a298c | one doc per passage |
| `unh/` | **UDH** (package `nkg`; result files use the system name `unh`) | this repository | one passage = one chunk |
| `common/` | shared answer step, prompts, metrics, scorers, result collectors, checks | | |

Every line patched in a vendored baseline is tagged `[UDH]`.
G-Medical and Contract QA reach every baseline through one document-collection loader (`--dataset docs`) that reads
an inputs dir of `evaluation_data/` (`documents.json`, `chunks.json`, `questions.json`, ...).

## Protocol

Each system builds its own index and retrieves its own context; then one shared step answers and one scorer scores:

| Dataset | Answer step | Metric | Scorer |
|---|---|---|---|
| HotpotQA, MuSiQue | `common/answer.py`: HippoRAG 2's `rag_qa_musique` one-shot CoT prompt, answer = text after "Answer:", max_tokens 2048 | EM, F1 (gold + aliases), LLM-Acc (LinearRAG's judge prompt) | `common/score.py --llm-judge` |
| G-Medical | `common/answer.py --dataset gmedical`: LinearRAG's zero-shot answer prompt, max_tokens 4096 | LLM-Acc, overall and per question type | `common/score_gmedical.py` |
| Contract QA | `common/contracts_score.py`: contract prompt (reasoning, then "Answer:"; abstain phrase "Not specified in this policy."), max_tokens 4096 | judge score 1 / 0.5 / 0 (CAT5 = unanswerable) | `common/contracts_score.py` |

Only the context differs between systems, and each system decides its own (what its own final answer call reads):

| System | Context passed to the reader |
|---|---|
| HippoRAG 2 | top-5 passages (`HippoRAG.qa()` input) |
| MemGraphRAG | top-5 token windows (`MemGraphRAG.qa()` input, `--qa-top-k 5`) |
| HyGRAG | `HKGraphTreeQuery._retrieve_relevant_contexts` output |
| MS GraphRAG | local-search `build_context` text (what graphrag's own answer prompt would contain) |
| HyperGraphRAG | hybrid-mode context (`aquery(only_need_context=True)`) |
| LogicRAG | its final `info_summary` (`LogicRAG.generate_answer()` input) |
| UDH | evidence paths under a 2,000-token budget; UDH+5p: the same paths plus the five passages with the highest PPR scores |

The judges see the question (Contract QA), the gold answer and the predicted answer, never the context.
**Ctx** (mean reader-context tokens per question) is measured with `cl100k_base` on the context the reader received.

## Models

- **LLM:** DeepSeek V4.1 Flash (`LLM_MODEL`, default `deepseek-flash`) for index construction, every baseline's
  query-time calls, the answer step and the judges. Thinking is disabled in every client; no client sends a
  temperature (DeepSeek's default applies). Settings: `common/env.sh`.
- **Embeddings:** `voyageai/voyage-4-nano`. LogicRAG and UDH load it in-process (SentenceTransformer); the others call
  an OpenAI-compatible `/v1/embeddings` endpoint, `EMBEDDING_BASE_URL` (default `http://localhost:8001/v1`; the vLLM
  command is in `common/env.sh`, `common/st_embedding_server.py` is a slower fallback).
- **UDH cross-encoder:** `cross-encoder/ms-marco-MiniLM-L6-v2`.

## Run

```bash
export DEEPSEEK_API_KEY=...                      # or DEEPSEEK_API_KEY=... in <repo>/.env (git-ignored)
python evaluation_scripts/common/verify_data.py  # byte-check HotpotQA / MuSiQue
bash evaluation_scripts/launch_all.sh            # HotpotQA + MuSiQue, every system in parallel (logs in run_logs/)
bash evaluation_scripts/launch_gmedical.sh       # G-Medical
bash evaluation_scripts/launch_contracts.sh      # Contract QA
bash evaluation_scripts/stop_all.sh              # stop everything (LLM calls are cached; a relaunch resumes)
```

or one system at a time, e.g. `bash evaluation_scripts/unh/run.sh hotpotqa`, `bash evaluation_scripts/hipporag2/run_gmedical.sh`,
`bash evaluation_scripts/hygrag/run_contracts.sh`. MS GraphRAG and HyGRAG use their own Python environments
(`ms_graphrag/requirements.txt`, `hygrag/requirements.unh.txt`; `GRAPHRAG_BIN`, `HYGRAG_PYTHON`).

## Results

The committed result files are the paper's runs.

| | Per-question answers (context, raw reply, answer) | Scores (per question + totals) |
|---|---|---|
| HotpotQA / MuSiQue | `<system>/outputs/<ds>/answers_<system>_<ds>.json` (LogicRAG: `logicrag/result/`) | `scores_<system>_<ds>.json` next to it |
| G-Medical | `<system>/outputs/gmedical/answers_<system>_gmedical.json` (HippoRAG 2: `outputs/gmedical_docs/`; LogicRAG: `result/gmedical/`) | `scores_<system>_gmedical.json` |
| Contract QA | `contracts_results/long_qa/<system>/<contract>/scores_qa.json` (context, answer, judge reply, score) | `contracts_results/long_qa/<system>/summary.json` |

UDH (`unh/outputs/<dataset>/`, contracts: `unh/outputs/contracts_long_qa/<contract>/`) also keeps its retrieved
contexts (`contexts.json` = UDH, `contexts_hyb_p5.json` = UDH+5p, `contexts_hypergraph.json` = UDH-H,
`contexts_discourse.json` = UDH-D), the retrieval settings and path statistics (`retrieval_report_*.json`), and the
index construction settings and LLM token usage (`construction_results.json`). The answer and score files of the four
UDH rows use the system names `unh`, `unh_hyb_p5`, `unh_hypergraph`, `unh_discourse`. One file over GitHub's size
limit is stored gzipped (`hypergraphrag/outputs/gmedical/answers_hypergraphrag_gmedical.json.gz`).

```bash
python evaluation_scripts/common/collect_results.py    # HotpotQA / MuSiQue / G-Medical tables -> results_table.csv
python evaluation_scripts/common/collect_contracts.py  # Contract QA tables -> contracts_results/long_qa/tables.md
python evaluation_scripts/common/verify_results.py     # re-derive every table number (and Ctx) from the per-question rows
```
