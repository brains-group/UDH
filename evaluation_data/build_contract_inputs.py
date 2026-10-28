#!/usr/bin/env python3
"""Contract QA in the input format of every system in evaluation_scripts/.

    python evaluation_data/build_contract_inputs.py      # -> evaluation_data/contracts/long_qa/<contract>/

Source: long_form_contracts/ (5 long-form insurance contracts, 6k-49k tokens each): contracts/<c>.txt (text extracted
from contracts/<c>.pdf) and qa_pairs/qa_<c>.json (445 QA pairs in five categories, CAT1-CAT5; qa_generation_prompt.txt).

Protocol: one index per contract per system; every question is a retrieval query against its own contract's index
only. Each contract gets its own inputs dir, read by every runner's document-collection loader (`--dataset docs`):

  questions.json         every question of the contract: {id, kind: "qa", question, answer (gold), answer_letter: ""
                         (unused), category, evidence, fact_id: null, contract}; ids "<contract>/qa/<n>".
  documents.json         HyGRAG, HyperGraphRAG: the whole contract as one document {article_id, title: "", text}.
  chunks.json            HippoRAG 2 (no chunker): the contract cut with HyGRAG's own chunker (tiktoken gpt-3.5-turbo,
                         1200 tokens, a new window every 1100).
  chunks_logicrag.json   LogicRAG (no chunker): the same chunks, title "".
  memgraphrag_corpus.txt MemGraphRAG: the contract text (it cuts its own 256/32-token windows).
  ms_graphrag_input/     MS GraphRAG: <contract>.txt.
  contract.txt           UDH: the contract text (UDH chunks it itself, 600/50).
  manifest.json          counts and SHA-256 of every file.

The text is the contract .txt with its `<!-- image -->` lines (PDF-conversion placeholders) removed; otherwise
byte-identical for every system. There is no title: a contract is one document, so `title` is "" everywhere.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import tiktoken

HERE = Path(__file__).resolve().parent
SRC = HERE / "long_form_contracts"
SET = "long_qa"
OUT_ROOT = HERE / "contracts" / SET
CHUNK_TOKENS, OVERLAP = 1200, 100  # HyGRAG config/HKGraphTreeLSH.yaml chunk_token_size / chunk_overlap_token_size
ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")  # HyGRAG token_model: gpt-3.5-turbo (cl100k_base)
CATEGORIES = ["CAT1_ATOMIC_LOOKUP", "CAT2_RELATIONAL_LOOKUP", "CAT3_CONDITIONAL_REASONING",
              "CAT4_MULTIHOP_REASONING", "CAT5_UNANSWERABLE"]


def digimon_chunks(text: str) -> list:
    """HyGRAG Core/Chunk/DocChunk.build_chunks + Tokensize.chunking_by_token_size for one document."""
    tokens = ENC.encode(text.strip(), disallowed_special=())
    windows = [tokens[start:start + CHUNK_TOKENS] for start in range(0, len(tokens), CHUNK_TOKENS - OVERLAP)]
    return [chunk.strip() for chunk in ENC.decode_batch(windows)]


def clean(text: str) -> str:
    text = "\n".join(line for line in text.splitlines() if line.strip() != "<!-- image -->")
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_contract(out: Path, c: str, text: str, qa: list):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    queries = []
    for n, q in enumerate(qa):
        assert q["category"] in CATEGORIES, q["category"]
        queries.append({"id": f"{c}/qa/{n:04d}", "kind": "qa", "question": q["question"].strip(),
                        "answer": q["answer"], "answer_letter": "", "category": q["category"],
                        "evidence": q.get("evidence", ""), "fact_id": None, "contract": c})
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids)), f"{c}: duplicate query ids"

    chunks = digimon_chunks(text)

    def dump(name, obj):
        json.dump(obj, open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    dump("questions.json", queries)
    dump("documents.json", [{"article_id": c, "title": "", "text": text}])
    dump("chunks.json", [{"article_id": c, "chunk_index": i, "title": "", "text": t} for i, t in enumerate(chunks)])
    dump("chunks_logicrag.json", [{"article_id": c, "chunk_index": i, "title": "", "text": t} for i, t in enumerate(chunks)])
    (out / "memgraphrag_corpus.txt").write_text(text, encoding="utf-8")
    (out / "contract.txt").write_text(text, encoding="utf-8")
    (out / "ms_graphrag_input").mkdir()
    (out / "ms_graphrag_input" / f"{c}.txt").write_text(text, encoding="utf-8")
    files = ["questions.json", "documents.json", "chunks.json", "chunks_logicrag.json", "memgraphrag_corpus.txt",
             "contract.txt", f"ms_graphrag_input/{c}.txt"]
    manifest = {"contract": c, "source": "long_qa (QA only) from long",
                "tokens_cl100k": len(ENC.encode(text, disallowed_special=())),
                "chars": len(text), "chunks_1200": len(chunks),
                "qa": len(qa), "facts": 0,
                "qa_by_category": {k: sum(q["category"] == k for q in qa) for k in CATEGORIES},
                "sha256": {f: sha256(out / f) for f in files}}
    dump("manifest.json", manifest)
    return manifest


def main():
    rows = []
    for c in sorted(p.stem for p in (SRC / "contracts").glob("*.txt")):
        text = clean((SRC / "contracts" / f"{c}.txt").read_text(encoding="utf-8"))
        qa = json.load(open(SRC / "qa_pairs" / f"qa_{c}.json", encoding="utf-8"))["qa_pairs"]
        rows.append(write_contract(OUT_ROOT / c, c, text, qa))
    json.dump({"set": SET, "contracts": [r["contract"] for r in rows],
               "totals": {k: sum(r[k] for r in rows) for k in ("tokens_cl100k", "qa", "facts", "chunks_1200")},
               "per_contract": [{k: v for k, v in r.items() if k != "sha256"} for r in rows]},
              open(OUT_ROOT / "contracts.json", "w"), indent=1)
    print(f"{SET}: {len(rows)} contracts, {sum(r['qa'] for r in rows)} QA")


if __name__ == "__main__":
    main()
