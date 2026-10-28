# Evaluation data

The four datasets of the paper. Every system reads the same files; the per-system inputs for G-Medical and Contract QA
are built by the two scripts below (deterministic: rebuilding gives byte-identical files and the SHA-256 digests in
each `manifest.json`).

| Dataset | Questions | Corpus | Files |
|---|---|---|---|
| HotpotQA | 1,000 | 9,811 passages | `hotpotqa/` |
| MuSiQue | 1,000 | 11,656 passages | `musique/` |
| G-Medical | 2,062 (4 question types) | 1 document (NCCN patient guidelines, ~175k words) | `gmedical/inputs/` |
| Contract QA | 445 (5 categories) | 5 contracts, one index each (6k-49k tokens) | `contracts/long_qa/`, source in `long_form_contracts/` |

## HotpotQA and MuSiQue (`hotpotqa/`, `musique/`)

The standard 1,000-question multi-hop sets used by HippoRAG 2, MemGraphRAG, LinearRAG and LogicRAG, copied unchanged
from the HippoRAG 2 data release ([osunlp/HippoRAG_2](https://huggingface.co/datasets/osunlp/HippoRAG_2)).
`python evaluation_scripts/common/verify_data.py` re-checks them byte for byte.

| File | Contents |
|---|---|
| `<ds>.json` | 1,000 questions (HotpotQA: dev-distractor records; MuSiQue: answerable dev records incl. `answer_aliases`) |
| `<ds>_corpus.json` | retrieval corpus `[{"title", "text"}]`: every paragraph attached to any of the 1,000 questions (gold + distractors) |
| `<ds>.txt` | the corpus as `title\ntext` blocks joined by a blank line (MemGraphRAG's input format) |
| `<ds>_scrambled.txt` | the same passages in shuffled order (seed 0; `make_scrambled_txt.py`), what MemGraphRAG indexes: MemGraphRAG cuts 256-token windows across passage boundaries, and `<ds>.txt` stores the passages of one question next to each other |

| | HotpotQA | MuSiQue |
|---|---|---|
| questions | 1,000 (811 bridge / 189 comparison) | 1,000 (518 2-hop / 316 3-hop / 166 4-hop) |
| corpus passages | 9,811 | 11,656 (only 9,838 unique titles: never deduplicate by title) |

The HippoRAG 2 release is byte-identical to `OSU-NLP-Group/HippoRAG reproduce/dataset/`, to MemGraphRAG's `dataset/`
(except a wrapper around `musique.json`) and, in content, to LogicRAG's copies.

## G-Medical (`gmedical/inputs/`)

GraphRAG-Bench Medical as MemGraphRAG evaluates it: MemGraphRAG's `dataset/medical/` (`medical.txt`, `question.json`).
The corpus is one document, `medical.txt` verbatim (the official GraphRAG-Bench `medical.json` context); every system
chunks it itself. Built by

```bash
python evaluation_data/build_gmedical_inputs.py --source <MemGraphRAG checkout>/dataset/medical
```

| File | Read by |
|---|---|
| `questions.json` | every runner, `answer.py`, `score_gmedical.py`: `{id, question, answer, question_type, level, source, evidence}` |
| `documents.json` | HyGRAG, HyperGraphRAG, UDH (they chunk it themselves) |
| `chunks.json` | HippoRAG 2 (no chunker): MemGraphRAG's own 256/32-token windows (991 chunks) |
| `chunks_logicrag.json` | LogicRAG (no chunker): the same chunks |
| `memgraphrag_corpus.txt` | MemGraphRAG |
| `ms_graphrag_input/` | MS GraphRAG |

## Contract QA (`long_form_contracts/`, `contracts/long_qa/`)

Five long-form insurance contracts (`long_form_contracts/contracts/<c>.pdf` and the extracted text `<c>.txt`) with
445 question-answer pairs (`long_form_contracts/qa_pairs/qa_<c>.json`: question, answer, evidence, category),
generated with `qa_generation_prompt.txt`; 20% were checked by hand.

| Category | Questions |
|---|---|
| CAT1_ATOMIC_LOOKUP | 131 |
| CAT2_RELATIONAL_LOOKUP | 100 |
| CAT3_CONDITIONAL_REASONING | 98 |
| CAT4_MULTIHOP_REASONING | 82 |
| CAT5_UNANSWERABLE | 34 |

Each contract is its own corpus (one index per contract per system). `python evaluation_data/build_contract_inputs.py`
writes one inputs dir per contract, `contracts/long_qa/<contract>/`, in the same layout as G-Medical's (plus
`contract.txt`, UDH's input); the text is the contract `.txt` with its `<!-- image -->` placeholder lines removed.
`contracts/long_qa/contracts.json` lists the contracts and their sizes.
