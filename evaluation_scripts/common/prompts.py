"""Prompts copied from the papers' own code, so every system is read and judged the same way.

READER  — HippoRAG 2 `rag_qa_musique` one-shot CoT prompt. MemGraphRAG uses the same template
          (its code falls back to `rag_qa_musique` for every dataset), and so does HippoRAG 2 for
          both HotpotQA and MuSiQue. Source file copied verbatim: hipporag2_rag_qa_musique.py
          (OSU-NLP-Group/HippoRAG @ 1438aba, src/hipporag/prompts/templates/rag_qa_musique.py).
JUDGE   — LLM-Acc prompt, verbatim from LinearRAG src/evaluate.py (DEEP-PolyU/LinearRAG), the
          "LLM-Acc" metric reported in the MemGraphRAG / LinearRAG tables.
"""
from string import Template
from typing import Dict, List

from hipporag2_rag_qa_musique import prompt_template as _RAG_QA_TEMPLATE


def build_rag_qa_messages(question: str, passages: List[str], top_k: int = 5) -> List[Dict[str, str]]:
    """Exactly what HippoRAG2.qa() / MemGraphRAG.qa() send: every message goes through
    string.Template.substitute (so the one-shot's "$$94 million" renders as "$94 million")."""
    prompt_user = ""
    for passage in passages[:top_k]:
        prompt_user += f"Wikipedia Title: {passage}\n\n"
    prompt_user += "Question: " + question + "\nThought: "
    return [
        {"role": m["role"], "content": Template(m["content"]).substitute(prompt_user=prompt_user)}
        for m in _RAG_QA_TEMPLATE
    ]


# ---- LinearRAG's answer prompt (DEEP-PolyU/LinearRAG src/LinearRAG.py LinearRAG.qa, verbatim; commit bcc94e6), the
# G-Medical answer prompt (answer.py --dataset gmedical default). LinearRAG is from the
# GraphRAG-Bench / MemGraphRAG authors' group and runs it on G-Medical: the same system instruction as HippoRAG 2's
# rag_qa_musique, but zero-shot, without the Wikipedia one-shot example or the "Wikipedia Title:" labels. ----
LINEARRAG_QA_SYSTEM_PROMPT = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions '
    'meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, '
    'illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, '
    'devoid of additional elaborations.'
)


def build_linearrag_messages(question: str, passages: List[str]) -> List[Dict[str, str]]:
    """LinearRAG.qa(): every passage followed by a newline, then the question and " Thought: " (its spacing kept).
    The answer is the text after "Answer:" (extract_answer, as LinearRAG parses it)."""
    prompt_user = ""
    for passage in passages:
        prompt_user += f"{passage}\n"
    prompt_user += f"Question: {question}\n Thought: "
    return [{"role": "system", "content": LINEARRAG_QA_SYSTEM_PROMPT}, {"role": "user", "content": prompt_user}]


def extract_answer(response: str) -> str:
    """HippoRAG2.qa() / MemGraphRAG.qa(): text after the first 'Answer:', else the whole response."""
    try:
        return response.split("Answer:")[1].strip()
    except Exception:
        return response


# ---- LinearRAG src/evaluate.py :: Evaluator.calculate_llm_accuracy (verbatim strings) ----
# (escaped so the trailing space after "evaluator." survives editors that strip whitespace)
JUDGE_SYSTEM_PROMPT = "You are an expert evaluator. \n        "


def judge_user_prompt(pre_answer: str, gold_ans: str) -> str:
    return f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
        Generated answer: {pre_answer}
        Gold answer: {gold_ans}

        The generated answer should be considered correct if it:
        1. Contains the key information from the gold answer
        2. Is factually accurate and consistent with the gold answer
        3. Does not contain any contradicting information

        Respond with ONLY 'correct' or 'incorrect'.
        Response:
        """


# ---- Contract QA, shared by every system ----
# Our own prompts: the answer prompt and the QA judge (1.0 / 0.5 / 0.0, category-aware for CAT5 unanswerable).
# Thinking is allowed: each prompt asks for written reasoning first, then a final line that is parsed.
CONTRACT_ABSTAIN = "Not specified in this policy."

CONTRACT_ANSWER_SYSTEM = "You are a precise question-answering system grounded strictly in the context you are given."


def build_contract_messages(question: str, docs: List[str]) -> List[Dict[str, str]]:
    context = "\n\n".join(d for d in docs if d and d.strip())
    user = (f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer the question using ONLY the information in the context. Do NOT use outside knowledge or make "
            "assumptions, and do not fabricate facts.\n"
            "First reason step by step after \"Thought:\". Then give the final answer after \"Answer:\" in 1-2 concise "
            "sentences, in direct, factual language.\n"
            f"If the context does not contain enough information to answer, the final answer is exactly: {CONTRACT_ABSTAIN}\n\n"
            "Thought:")
    return [{"role": "system", "content": CONTRACT_ANSWER_SYSTEM}, {"role": "user", "content": user}]


def extract_contract_answer(response: str) -> str:
    """Text after the LAST 'Answer:' (the reasoning may quote the word); the whole reply if there is none."""
    response = response or ""
    idx = response.rfind("Answer:")
    return (response[idx + len("Answer:"):] if idx >= 0 else response).strip()


CONTRACT_QA_JUDGE_SYSTEM = "You are a rigorous but fair answer-quality judge."


def contract_qa_judge_messages(question: str, correct_answer: str, llm_answer: str) -> List[Dict[str, str]]:
    user = f"""Score an LLM-generated answer by comparing it against the ground truth correct_answer for the given question.

Scoring rubric:
- 1.0 - Fully correct: the answer captures all key facts from the correct_answer without contradiction.
        Paraphrasing and different word order are acceptable as long as meaning is preserved.
- 0.5 - Partially correct: the answer contains the core idea but is missing a meaningful detail,
        is overly vague, or includes one minor inaccuracy alongside correct information.
- 0.0 - Wrong or irrelevant: the answer contradicts the correct_answer, is entirely off-topic,
        or contains no substantively correct information.

Rules:
- Judge based on semantic meaning, not surface wording. Synonyms and paraphrases count as correct.
- Do NOT penalize for stylistic differences, verbosity, or hedging language unless it obscures correctness.
- Do NOT award partial credit for answers that happen to mention correct keywords without a coherent answer.
- If the correct_answer has multiple distinct facts, partial omission of one warrants 0.5, omission of most warrants 0.0.
- If the correct_answer is "{CONTRACT_ABSTAIN}", the answer scores 1.0 if it says the information is not specified, not stated or not available, and 0.0 if it gives a concrete answer instead.
- If the correct_answer is a concrete answer, an answer that only says the information is not specified or not available scores 0.0.
- Base your judgment solely on the question and correct_answer provided - do not use outside knowledge.

Question: {question}
Correct answer: {correct_answer}
LLM answer: {llm_answer}

First explain your reasoning briefly. Then end with a final line of the form "Score: X", where X is 1.0, 0.5 or 0.0."""
    return [{"role": "system", "content": CONTRACT_QA_JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_judge_score(response: str, allowed):
    """The value on the last 'Score: X' line, if X is one of `allowed` (floats); else None."""
    import re
    found = re.findall(r"score\s*[:=]\s*\**\s*([01](?:\.\d+)?)", response or "", re.I)
    if not found:
        return None
    value = float(found[-1])
    return value if value in allowed else None
