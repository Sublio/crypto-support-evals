import re
from pathlib import Path

import ollama
from rank_bm25 import BM25Okapi

MODEL = "gemma2:2b"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

FALLBACK = "I don't have that information. Please contact support via the in-app chat."

SYSTEM_PROMPT = f"""You are the customer support assistant for Orbitra Exchange.

Rules:
- Answer ONLY using the information in the CONTEXT below. Do not use outside knowledge.
- If a sentence in the CONTEXT explicitly says something is not allowed, not accepted or not supported, that IS the answer: say it is not possible and quote that sentence. Never say something is not possible unless you can quote such a sentence.
- If the CONTEXT does not contain the answer, reply exactly: "{FALLBACK}"
- Never give investment, financial or tax advice, even if asked directly.
- Keep answers short and factual."""


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Split every .md file into one chunk per '## ' section.
    Returns [{"source": "fees.md", "text": "..."}, ...]"""
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        for section in re.split(r"\n(?=## )", path.read_text()):
            section = section.strip()
            if section:
                chunks.append({"source": path.name, "text": section})
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def retrieve(question: str, chunks: list[dict], k: int = 3) -> list[dict]:
    """Return the k chunks with the highest BM25 score for the question."""
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in ranked[:k]]


def build_prompt(question: str, contexts: list[dict]) -> str:
    """Retrieved context + question; the rules live in SYSTEM_PROMPT."""
    context_text = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in contexts)
    return f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"


def answer(question: str, k: int = 3) -> dict:
    """Returns {"answer": str, "contexts": [str], "sources": [str]}"""
    contexts = retrieve(question, load_chunks(), k=k)
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, contexts)},
        ],
        options={"temperature": 0},
    )
    return {
        "answer": response["message"]["content"].strip(),
        "contexts": [c["text"] for c in contexts],
        "sources": [c["source"] for c in contexts],
    }


if __name__ == "__main__":
    import sys
    print(answer(" ".join(sys.argv[1:])))
