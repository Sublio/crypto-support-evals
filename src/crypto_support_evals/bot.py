import re
from pathlib import Path

import ollama
import snowballstemmer
from rank_bm25 import BM25Okapi

MODEL = "gemma2:2b"
STEMMER = snowballstemmer.stemmer("english")
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

FALLBACK = "I don't have that information. Please contact support via the in-app chat."

SYSTEM_PROMPT = f"""You are the customer support assistant for Orbitra Exchange.

Rules:
- Answer ONLY using the information in the CONTEXT below. Do not use outside knowledge.
- If a sentence in the CONTEXT explicitly says something is not allowed, not accepted or not supported, that IS the answer: say it is not possible and quote that sentence.   Never say something is not possible unless you can quote such a sentence.
- If the CONTEXT does not contain the answer, reply exactly: "{FALLBACK}"
- Never give investment, financial or tax advice, even if asked directly.
- Keep answers short and factual."""

# Share of an answer's content words that must appear in the retrieved context.
# Real answers score 0.7+, off-topic ones (jokes, role-play) score near 0.
GROUNDED_THRESHOLD = 0.5

STOPWORDS = set("""a an the and or but of to in on at for with by from is are was were be been it
this that these those you your i we our can cannot not no do does did will would should may might
has have had as if so than then there here what which who how when why""".split())


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


def stem_tokens(text: str) -> list[str]:
    """Tokens reduced to their stem, so "stake" matches "staking" in BM25."""
    return STEMMER.stemWords(tokenize(text))


def retrieve(question: str, chunks: list[dict], k: int = 3) -> list[dict]:
    """Return the k chunks with the highest BM25 score for the question."""
    bm25 = BM25Okapi([stem_tokens(c["text"]) for c in chunks])
    scores = bm25.get_scores(stem_tokens(question))
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in ranked[:k]]


def build_prompt(question: str, contexts: list[dict]) -> str:
    """Retrieved context + question; the rules live in SYSTEM_PROMPT."""
    context_text = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in contexts)
    return f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"


def is_grounded(answer_text: str, contexts: list[dict]) -> bool:
    """True if enough of the answer's content words appear in the retrieved context."""
    if answer_text == FALLBACK:
        return True
    context_words = set(tokenize(" ".join(c["text"] for c in contexts)))
    words = [w for w in tokenize(answer_text) if w not in STOPWORDS]
    if not words:
        return False
    return sum(w in context_words for w in words) / len(words) >= GROUNDED_THRESHOLD


def answer(question: str, k: int = 3) -> dict:
    """Returns {"answer": str, "grounded": bool, "contexts": [str], "sources": [str]}
    An answer that isn't grounded in the context is replaced with FALLBACK."""
    contexts = retrieve(question, load_chunks(), k=k)
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, contexts)},
        ],
        options={"temperature": 0},
    )
    text = response["message"]["content"].strip()
    grounded = is_grounded(text, contexts)
    return {
        "answer": text if grounded else FALLBACK,
        "grounded": grounded,
        "contexts": [c["text"] for c in contexts],
        "sources": [c["source"] for c in contexts],
    }


if __name__ == "__main__":
    import sys
    print(answer(" ".join(sys.argv[1:])))
