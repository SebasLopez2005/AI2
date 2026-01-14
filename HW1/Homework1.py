import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

# Word embeddings (gensim)
import gensim.downloader as api

# SBERT (PyTorch)
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    top_k_candidates: int = 50
    sbert_similarity_threshold: float = 0.86
    use_pos_filter: bool = True
    target_strategy: str = "last_content_word"  # "last_word" also supported


# -----------------------------
# Sentence + token handling
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def tokenize_words(sentence: str) -> List[str]:
    return _WORD_RE.findall(sentence)

def is_stopword_basic(w: str) -> bool:
    STOP = {
        "the","a","an","and","or","but","if","then","else","for","to","of","in","on","at","by","with",
        "is","are","was","were","be","been","being","as","that","this","these","those","it","its",
        "i","you","he","she","we","they","me","him","her","us","them","my","your","his","their","our"
    }
    return w.lower() in STOP

def select_target_word(sentence: str, strategy: str = "last_content_word") -> Optional[str]:
    words = tokenize_words(sentence)
    if not words:
        return None
    if strategy == "last_word":
        return words[-1]
    for w in reversed(words):
        if not is_stopword_basic(w):
            return w
    return words[-1]

def replace_word_once(sentence: str, original: str, replacement: str) -> str:
    """
    Replace ONE occurrence, preferring the last whole-word match.
    """
    pattern = re.compile(rf"\b{re.escape(original)}\b")
    matches = list(pattern.finditer(sentence))
    if not matches:
        pattern_ci = re.compile(rf"\b{re.escape(original)}\b", re.IGNORECASE)
        matches = list(pattern_ci.finditer(sentence))
        if not matches:
            return sentence
        m = matches[-1]
        return sentence[:m.start()] + replacement + sentence[m.end():]

    m = matches[-1]
    return sentence[:m.start()] + replacement + sentence[m.end():]


# -----------------------------
# POS tagging (NLTK) - optional
# -----------------------------
def _nltk_ready() -> bool:
    try:
        import nltk  # noqa: F401
        from nltk import pos_tag  # noqa: F401
        return True
    except Exception:
        return False

def get_pos_tag(word: str) -> Optional[str]:
    """
    Returns coarse POS for a single word using NLTK if available, else None.
    """
    try:
        from nltk import pos_tag
        tag = pos_tag([word])[0][1]  # e.g., NN, VB, JJ...
        if tag.startswith("NN"):
            return "NOUN"
        if tag.startswith("VB"):
            return "VERB"
        if tag.startswith("JJ"):
            return "ADJ"
        if tag.startswith("RB"):
            return "ADV"
        return "OTHER"
    except Exception:
        return None


# -----------------------------
# Models
# -----------------------------
def load_word_vectors(model_name: str = "glove-wiki-gigaword-100"):
    """
    Loads pretrained vectors via gensim downloader (cached after first download).
    If you already have GoogleNews Word2Vec working, you can use:
      model_name="word2vec-google-news-300"
    """
    return api.load(model_name)

def load_sbert_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


# -----------------------------
# Candidate generation + filtering
# -----------------------------
def generate_candidates(wv, target: str, top_k: int) -> List[str]:
    key = target
    if key not in wv:
        key = target.lower()
        if key not in wv:
            return []
    neighbors = wv.most_similar(key, topn=top_k)
    return [w for (w, _) in neighbors]

def filter_candidates(
    candidates: List[str],
    target: str,
    use_pos_filter: bool = True
) -> List[str]:
    # If NLTK isn't available, silently skip POS filtering
    pos_ok = use_pos_filter and _nltk_ready()
    target_pos = get_pos_tag(target) if pos_ok else None

    filtered = []
    for c in candidates:
        if not c:
            continue
        # Common junk from embeddings
        if any(ch.isdigit() for ch in c):
            continue
        if "_" in c:
            continue
        if len(c) <= 2:
            continue
        if c.lower() == target.lower():
            continue
        if is_stopword_basic(c):
            continue

        # Coarse POS match (only if NLTK is actually working)
        if pos_ok and target_pos is not None:
            c_pos = get_pos_tag(c)
            if c_pos is not None and target_pos in {"NOUN","VERB","ADJ","ADV"}:
                if c_pos != target_pos:
                    continue

        filtered.append(c)

    # Deduplicate (case-insensitive), preserve order
    seen = set()
    out = []
    for c in filtered:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            out.append(c)
    return out


# -----------------------------
# SBERT scoring (PATCHED)
# -----------------------------
def pick_best_replacement_sbert(
    sbert: SentenceTransformer,
    sentence: str,
    target: str,
    candidates: List[str],
    threshold: float
) -> Tuple[str, Optional[float], Dict[str, float]]:
    """
    PATCH:
    - Previously, baseline was 1.0 and we only replaced if score > 1.0 (impossible).
    - Now we:
        1) compute sim(S, S_c) for each candidate
        2) pick candidate with max similarity
        3) replace if max_sim >= threshold
    Returns (chosen_word, chosen_score_or_None_if_no_change, score_map).
    """
    if not candidates:
        return target, None, {}

    # Normalize embeddings so dot product == cosine similarity
    original_emb = sbert.encode(sentence, normalize_embeddings=True)

    score_map: Dict[str, float] = {}
    best_word = target
    best_score = -1.0  # allow any candidate to win

    for c in candidates:
        modified = replace_word_once(sentence, target, c)
        if modified == sentence:
            continue
        mod_emb = sbert.encode(modified, normalize_embeddings=True)
        score = float(np.dot(original_emb, mod_emb))
        score_map[c] = score

        if score > best_score:
            best_score = score
            best_word = c

    # Conservative policy: only replace if threshold is met
    if best_word != target and best_score >= threshold:
        return best_word, best_score, score_map

    return target, None, score_map


# -----------------------------
# End-to-end pipeline
# -----------------------------
def rewrite_text(text: str, wv, sbert: SentenceTransformer, cfg: Config):
    sentences = split_sentences(text)
    outputs = []
    debug_rows = []

    for s in sentences:
        target = select_target_word(s, cfg.target_strategy)
        if target is None:
            outputs.append(s)
            continue

        candidates = generate_candidates(wv, target, cfg.top_k_candidates)
        candidates = filter_candidates(candidates, target, use_pos_filter=cfg.use_pos_filter)

        chosen, chosen_score, score_map = pick_best_replacement_sbert(
            sbert=sbert,
            sentence=s,
            target=target,
            candidates=candidates,
            threshold=cfg.sbert_similarity_threshold,
        )

        modified = replace_word_once(s, target, chosen) if chosen != target else s
        outputs.append(modified)

        top5 = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:5]
        debug_rows.append({
            "sentence": s,
            "target": target,
            "chosen": chosen,
            "chosen_score": chosen_score,
            "top5_candidates": top5,
            "pos_filter_active": (cfg.use_pos_filter and _nltk_ready()),
        })

    return " ".join(outputs), debug_rows


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    # If GoogleNews Word2Vec is working for you, swap to:
    # wv = load_word_vectors("word2vec-google-news-300")
    wv = load_word_vectors("glove-wiki-gigaword-100")
    sbert = load_sbert_model("all-MiniLM-L6-v2")

    cfg = Config(
        top_k_candidates=50,
        sbert_similarity_threshold=0.86,  # try 0.84 if you want "plumage" to have a chance
        use_pos_filter=True,
        target_strategy="last_content_word",
    )

    text = "Hope is the thing with feathers. Two roads diverged in a yellow wood."
    out, dbg = rewrite_text(text, wv, sbert, cfg)

    print("\nORIGINAL:\n", text)
    print("\nMODIFIED:\n", out)
    print("\nDEBUG:")
    for row in dbg:
        print(row)
