#!/usr/bin/env python3
"""G-Medical (GraphRAG-Bench Medical, as MemGraphRAG evaluates it) in the input format of every system in
evaluation_scripts/: the same document-collection layout as the contract inputs, read by every runner's loader
(`--dataset docs`); only the answer and score steps differ (open-ended answers, LLM-Acc).

    python evaluation_data/build_gmedical_inputs.py --source <MemGraphRAG>/dataset/medical    # -> gmedical/inputs/

Source = MemGraphRAG's copy (github.com/XMUDeepLIT/MemGraphRAG, dataset/medical/): medical.txt (one NCCN patient
guideline per line, 44 lines, 3 of them exact duplicates: AML, CML, ALL) and question.json ([{"source": "nccn",
"questions": {"type1": [...], ..., "type4": [...]}}], 2,062 questions). MemGraphRAG itself reads medical.txt as one
string and cuts 256/32-token windows across guideline boundaries.

Corpus: ONE document = medical.txt verbatim (all 44 lines, duplicates kept, line breaks kept), exactly the official
GraphRAG-Bench Datasets/Corpus/medical.json `context` string and what MemGraphRAG ingests; every system's own chunker
cuts across the line breaks (the benchmark defines no document boundaries).
What each system gets, written to <out>/:
  questions.json         every runner, answer.py, score_gmedical.py: {id, question, answer, question_type
                         (type1..type4), level (Fact Retrieval / Complex Reasoning / Contextual Summarize /
                         Creative Generation), source, evidence}, in MemGraphRAG's order (type1, type2, type3, type4)
  documents.json         HyGRAG, HyperGraphRAG, UDH: {article_id, title: "", text}; they chunk it themselves
  chunks.json            HippoRAG 2 (no chunker of its own): the document cut exactly as MemGraphRAG's index.py and
                         GraphRAG-Bench's Examples/run_hipporag2.py cut it: 256/32 windows of the embedding model's HF
                         tokenizer, each passage prefixed "<idx>:" -> the same 991 windows MemGraphRAG indexes.
  chunks_logicrag.json   LogicRAG (no chunker either): the same chunks, {title: "", text}
  memgraphrag_corpus.txt MemGraphRAG: the document text
  ms_graphrag_input/     MS GraphRAG: medical.txt
  manifest.json          counts and SHA-256 of every file
"""
import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CHUNK_TOKENS, CHUNK_OVERLAP = 256, 32   # MemGraphRAG run_index.sh / GraphRAG-Bench run_hipporag2.py
TOKENIZER = "voyageai/voyage-4-nano"    # the embedding model's tokenizer, as both use (memgraphrag/run_gmedical.sh)


def split_text(text: str, tokenizer, chunk_size: int, overlap: int) -> list:
    """MemGraphRAG code/index.py split_text, verbatim (GraphRAG-Bench's run_hipporag2.py split_text is the same)."""
    previous_limit = getattr(tokenizer, "model_max_length", None)
    try:
        tokenizer.model_max_length = int(1e9)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    finally:
        if previous_limit is not None:
            tokenizer.model_max_length = previous_limit
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(token_ids), step):
        chunk = token_ids[start:start + chunk_size]
        if not chunk:
            break
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        if start + chunk_size >= len(token_ids):
            break
    return chunks

HERE = Path(__file__).resolve().parent
SOURCE_NAME = "MemGraphRAG dataset/medical (github.com/XMUDeepLIT/MemGraphRAG)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

LEVELS = {"type1": "Fact Retrieval", "type2": "Complex Reasoning", "type3": "Contextual Summarize",
          "type4": "Creative Generation"}


def infer_source_docs(docs: list, questions: list) -> dict:
    """question `source` -> index of the guideline its evidence sentences overlap most (word overlap, >3 letters).
    A heuristic (medical.txt carries no names), recorded in the manifest only; never used as gold."""
    words = lambda t: {w for w in re.sub(r"[^a-z0-9 ]", " ", t.lower()).split() if len(w) > 3}  # noqa: E731
    vocab = [words(d["text"]) for d in docs]
    votes = defaultdict(Counter)
    for q in questions:
        for e in q["evidence"] if isinstance(q["evidence"], list) else [q["evidence"]]:
            w = words(e)
            if w:
                votes[q["source"]][max(range(len(docs)), key=lambda i: len(w & vocab[i]))] += 1
    return {src: c.most_common(1)[0][0] for src, c in votes.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="MemGraphRAG's dataset/medical dir (medical.txt, question.json)")
    ap.add_argument("--out", default=None, help="default: gmedical/inputs")
    args = ap.parse_args()
    SRC = Path(args.source)
    out = Path(args.out) if args.out else HERE / "gmedical" / "inputs"

    raw = (SRC / "medical.txt").read_text(encoding="utf-8")
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    seen, docs = set(), []
    for line in lines:
        h = hashlib.md5(line.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            docs.append({"article_id": f"med_{len(docs):02d}", "title": "", "text": line})

    groups = json.load(open(SRC / "question.json", encoding="utf-8"))[0]["questions"]
    questions = []
    for qtype in ("type1", "type2", "type3", "type4"):
        for q in groups[qtype]:
            questions.append({"id": q["id"], "question": q["question"].strip(), "answer": q["answer"],
                              "question_type": qtype, "level": LEVELS[qtype], "source": q["source"],
                              "evidence": q["evidence"]})
    assert len({q["id"] for q in questions}) == len(questions), "duplicate question ids"
    source_docs = infer_source_docs(docs, questions)
    # ONE document: the corpus verbatim (official medical.json context)
    docs = [{"article_id": "medical", "title": "", "text": raw.strip("\n")}]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    chunks = []
    for d in docs:  # one document: split_text(medical.txt as index.py reads it) -> MemGraphRAG's own windows
        text = d["text"] + "\n" if d["article_id"] == "medical" else d["text"]  # = memgraphrag_corpus.txt
        for i, piece in enumerate(split_text(text, tokenizer, CHUNK_TOKENS, CHUNK_OVERLAP)):
            chunks.append({"article_id": d["article_id"], "chunk_index": i, "title": "",
                           "text": f"{len(chunks)}:{piece}"})  # "<idx>:" as MemGraphRAG / GraphRAG-Bench docs

    if out.exists():
        shutil.rmtree(out)
    (out / "ms_graphrag_input").mkdir(parents=True)
    dump = lambda obj, name: json.dump(obj, open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)  # noqa: E731
    dump(questions, "questions.json")
    dump(docs, "documents.json")
    dump(chunks, "chunks.json")
    dump([{"title": c["title"], "text": c["text"]} for c in chunks], "chunks_logicrag.json")
    (out / "memgraphrag_corpus.txt").write_text("\n".join(d["text"] for d in docs) + "\n", encoding="utf-8")
    for d in docs:
        (out / "ms_graphrag_input" / f"{d['article_id']}.txt").write_text(d["text"], encoding="utf-8")

    files = ["questions.json", "documents.json", "chunks.json", "chunks_logicrag.json", "memgraphrag_corpus.txt"]
    manifest = {"source": SOURCE_NAME, "source_lines": len(lines), "distinct_guidelines": len(seen),
                "layout": "one_document (medical.txt verbatim)", "smoke_sources": None, "documents": len(docs),
                "document_words": sum(len(d["text"].split()) for d in docs), "chunks": len(chunks),
                "chunking (HippoRAG 2, LogicRAG)": f"{CHUNK_TOKENS}/{CHUNK_OVERLAP} {TOKENIZER} tokens, MemGraphRAG split_text", "questions": len(questions),
                "questions_per_type": {t: sum(q["question_type"] == t for q in questions) for t in LEVELS},
                "inferred_source_to_document (heuristic, not gold)": {s: f"med_{i:02d}" for s, i in sorted(source_docs.items())},
                "sha256": {f: sha256(out / f) for f in files}}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256" and not k.startswith("inferred")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
