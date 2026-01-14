import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

# Word vectors (gensim)
import gensim.downloader as api

# SBERT (PyTorch)
from sentence_transformers import SentenceTransformer


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # Candidate generation
    top_k_candidates: int = 50

    # Conservative replacement threshold
    sbert_similarity_threshold: float = 0.86

    # Target selection strategy
    target_strategy: str = "last_content_word"  # "last_word" also supported


# ============================================================
# Tokenization + sentence splitting
# ============================================================
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence segmentation for short texts.
    """
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


# ============================================================
# Heuristic guards (NO NLTK)
# ============================================================
DETERMINERS = {
    "a","an","the","this","that","these","those",
    "my","your","his","her","their","our"
}

# Adjective-ish suffixes (heuristic)
ADJ_SUFFIXES = (
    "en",   # wooden, golden
    "ed",   # feathered (often adjectival)
    "ous",  # famous
    "ful",  # helpful
    "ive",  # active
    "al",   # natural
    "ic",   # poetic
    "ish",  # childish
    "ary",  # primary
    "less", # fearless
    "y",    # feathery (context-sensitive)
)

def looks_plural(word: str) -> bool:
    w = word.lower()
    if w.endswith("ss"):
        return False
    return w.endswith("s")

def match_casing_like(candidate: str, target: str) -> str:
    """
    Preserve capitalization style of the target.
    """
    if target[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate.lower()

def find_last_token_index(tokens: List[str], target: str) -> Optional[int]:
    """
    Find last index of target in tokens; fallback to case-insensitive match.
    """
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == target:
            return i
    t = target.lower()
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i].lower() == t:
            return i
    return None

def nounish_context(tokens: List[str], target_index: int) -> bool:
    """
    Heuristic: treat target as likely NOUN if it appears in a noun phrase:
      - preceded by a determiner: "the wood"
      - preceded by determiner + modifier: "a yellow wood"
      - preceded by common preposition + determiner: "in a wood"
    This helps prevent noun->adj flips like wood -> wooden.
    """
    if target_index is None or target_index <= 0:
        return False

    prev = tokens[target_index - 1].lower()

    # Immediate determiner: "the wood"
    if prev in DETERMINERS:
        return True

    # Determiner two words back: "a yellow wood"
    if target_index >= 2:
        prev2 = tokens[target_index - 2].lower()
        if prev2 in DETERMINERS:
            return True

    # Preposition + determiner two/three words back: "in a wood", "in the dark forest"
    PREPS = {"in","on","at","by","to","from","into","over","under","within","through"}
    if target_index >= 2:
        prev2 = tokens[target_index - 2].lower()
        if prev2 in PREPS and prev in DETERMINERS:
            return True
    if target_index >= 3:
        prev3 = tokens[target_index - 3].lower()
        prev2 = tokens[target_index - 2].lower()
        if prev3 in PREPS and prev2 in DETERMINERS:
            return True

    return False


def is_junk_candidate(c: str, target: str) -> bool:
    if not c:
        return True
    if any(ch.isdigit() for ch in c):
        return True
    if "_" in c:
        return True
    if len(c) <= 2:
        return True
    if c.lower() == target.lower():
        return True
    if is_stopword_basic(c):
        return True
    return False

def filter_candidates_no_nltk(sentence: str, target: str, candidates: List[str]) -> List[str]:
    """
    Filtering using only heuristics (no NLTK):
    - remove junk tokens
    - enforce plural agreement
    - detect noun-phrase contexts (e.g., 'a yellow wood')
    - block noun -> adjective flips like wood -> wooden
    """
    tokens = tokenize_words(sentence)
    idx = find_last_token_index(tokens, target)
    is_nounish = nounish_context(tokens, idx) if idx is not None else False
    target_is_plural = looks_plural(target)

    filtered: List[str] = []
    for c in candidates:
        if is_junk_candidate(c, target):
            continue

        # Enforce plural agreement (feathers -> feather blocked)
        if target_is_plural and not looks_plural(c):
            continue

        if is_nounish:
            cl = c.lower()

            # 🔴 HARD BLOCK: noun → adjective (wood → wooden)
            if cl == target.lower() + "en":
                continue

            # Block common noun → adjective shifts via suffix
            if cl.endswith("en") and not target.lower().endswith("en"):
                continue

            # Conservative block for adjective-ish endings in noun contexts
            if cl.endswith(ADJ_SUFFIXES) and not looks_plural(c):
                continue

        filtered.append(c)

    # Deduplicate (case-insensitive) while preserving order
    seen = set()
    out = []
    for c in filtered:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)

    return out



# ============================================================
# Models
# ============================================================
def load_word_vectors(model_name: str = "glove-wiki-gigaword-100"):
    """
    Loads pretrained vectors via gensim downloader (cached after first download).
    If your network allows it and you have disk space, you can use:
      model_name="word2vec-google-news-300"
    """
    return api.load(model_name)

def load_sbert_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Sentence-Transformers uses PyTorch.
    """
    return SentenceTransformer(model_name)


# ============================================================
# Candidate generation + SBERT scoring
# ============================================================
def generate_candidates(wv, target: str, top_k: int) -> List[str]:
    key = target
    if key not in wv:
        key = target.lower()
        if key not in wv:
            return []
    neighbors = wv.most_similar(key, topn=top_k)
    return [w for (w, _) in neighbors]

def pick_best_replacement_sbert(
    sbert: SentenceTransformer,
    sentence: str,
    target: str,
    candidates: List[str],
    threshold: float
) -> Tuple[str, Optional[float], Dict[str, float]]:
    """
    Score each candidate by similarity between original sentence and modified sentence.
    Replace only if best_sim >= threshold.
    """
    if not candidates:
        return target, None, {}

    original_emb = sbert.encode(sentence, normalize_embeddings=True)

    best_word = target
    best_score = -1.0
    score_map: Dict[str, float] = {}

    for c in candidates:
        modified = replace_word_once(sentence, target, c)
        if modified == sentence:
            continue

        mod_emb = sbert.encode(modified, normalize_embeddings=True)
        score = float(np.dot(original_emb, mod_emb))  # cosine since normalized
        score_map[c] = score

        if score > best_score:
            best_score = score
            best_word = c

    if best_word != target and best_score >= threshold:
        return best_word, best_score, score_map

    return target, None, score_map


# ============================================================
# End-to-end pipeline
# ============================================================
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
        candidates = filter_candidates_no_nltk(s, target, candidates)

        chosen, chosen_score, score_map = pick_best_replacement_sbert(
            sbert=sbert,
            sentence=s,
            target=target,
            candidates=candidates,
            threshold=cfg.sbert_similarity_threshold,
        )

        chosen = match_casing_like(chosen, target)
        modified = replace_word_once(s, target, chosen) if chosen != target else s
        outputs.append(modified)

        top5 = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:5]
        debug_rows.append({
            "sentence": s,
            "target": target,
            "chosen": chosen,
            "chosen_score": chosen_score,
            "top5_candidates": top5,
        })

    return " ".join(outputs), debug_rows


def run_multi_threshold_suite(wv, sbert, base_cfg: Config):
    """
    Runs the report suite across multiple thresholds to compare behavior.
    Thresholds requested: 0.70, 0.80, 0.83, 0.86, 0.90
    """
    test_texts = [
        {
            "name": "Shakespeare (Metaphor & Polysemy)",
            "text": "All the world’s a stage, And all the men and women merely players."
        },
        {
            "name": "Robert Frost (Ambiguity)",
            "text": "Two roads diverged in a yellow wood."
        },
        {
            "name": "Emily Dickinson (Compressed Meaning)",
            "text": "Hope is the thing with feathers."
        },
        {
            "name": "William Blake (Archaic Language)",
            "text": "Tyger Tyger, burning bright, In the forests of the night."
        },
        {
            "name": "Haiku (Minimal Context)",
            "text": "An old silent pond— A frog jumps into the pond, Splash! Silence again."
        }
    ]

    thresholds = [0.70, 0.80, 0.83, 0.86, 0.90]

    print("\n" + "=" * 78)
    print(f"MULTI-THRESHOLD SUITE  |  topK={base_cfg.top_k_candidates}  |  strategy={base_cfg.target_strategy}")
    print("=" * 78)

    for thr in thresholds:
        cfg = Config(
            top_k_candidates=base_cfg.top_k_candidates,
            sbert_similarity_threshold=thr,
            target_strategy=base_cfg.target_strategy,
        )

        print("\n" + "#" * 78)
        print(f"THRESHOLD = {thr:.2f}")
        print("#" * 78)

        for item in test_texts:
            name = item["name"]
            text = item["text"]

            modified, debug_rows = rewrite_text(text, wv, sbert, cfg)

            print("\n" + "-" * 78)
            print(f"TEST: {name}")
            print("-" * 78)
            print("ORIGINAL:")
            print(text)
            print("\nMODIFIED:")
            print(modified)

            print("\nDEBUG (per sentence):")
            for row in debug_rows:
                sent = row["sentence"]
                target = row["target"]
                chosen = row["chosen"]
                score = row["chosen_score"]
                top5 = row["top5_candidates"]

                print(f"\nSentence: {sent}")
                print(f"  Target: {target}")
                print(f"  Chosen: {chosen}")
                print(f"  Score : {score}")
                print(f"  Top-5 candidates (candidate, sim): {top5}")

    print("\n" + "=" * 78)
    print("END MULTI-THRESHOLD SUITE")
    print("=" * 78)


if __name__ == "__main__":
    # Load once
    wv = load_word_vectors("glove-wiki-gigaword-100")
    sbert = load_sbert_model("all-MiniLM-L6-v2")

    base_cfg = Config(
        top_k_candidates=50,
        sbert_similarity_threshold=0.86,   # placeholder; overwritten in suite
        target_strategy="last_content_word",
    )

    run_multi_threshold_suite(wv, sbert, base_cfg)


