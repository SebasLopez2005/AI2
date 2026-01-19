"Homework 1 Special Topics in AI"
"Sebastian Lopez & Diego Bonilla"

import re
from typing import List, Optional, Tuple

import numpy as np
import gensim.downloader as api
from sentence_transformers import SentenceTransformer


# Configuration
TOP_K_CANDIDATES = 50
SIMILARITY_THRESHOLD = 0.86


# Basic stopwords list
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to", "of",
    "in", "on", "at", "by", "with", "is", "are", "was", "were", "be", "been",
    "being", "as", "that", "this", "these", "those", "it", "its", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "their", "our"
}


# Tokenization and sentence splitting
def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def tokenize_words(sentence: str) -> List[str]:
    """Extract words from sentence."""
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence)


def is_stopword(word: str) -> bool:
    """Check if word is a stopword."""
    return word.lower() in STOPWORDS


def select_target_word(sentence: str) -> Optional[str]:
    """Select the last non-stopword as target."""
    words = tokenize_words(sentence)
    if not words:
        return None
    
    # Try to find last content word
    for w in reversed(words):
        if not is_stopword(w):
            return w
    
    # Fallback to last word
    return words[-1] if words else None


def replace_word_once(sentence: str, original: str, replacement: str) -> str:
    """Replace the last occurrence of target word."""
    pattern = re.compile(rf"\b{re.escape(original)}\b", re.IGNORECASE)
    matches = list(pattern.finditer(sentence))
    
    if not matches:
        return sentence
    
    # Replace last match
    m = matches[-1]
    return sentence[:m.start()] + replacement + sentence[m.end():]


# Candidate filtering
def filter_candidates(candidates: List[str], target: str) -> List[str]:
    """Simple candidate filtering - remove junk and duplicates."""
    filtered = []
    seen = set()
    
    target_lower = target.lower()
    target_is_plural = target_lower.endswith('s') and not target_lower.endswith('ss')
    
    for c in candidates:
        c_lower = c.lower()
        
        # Skip if junk
        if (not c or len(c) <= 2 or c_lower == target_lower or 
            any(ch.isdigit() for ch in c) or "_" in c or is_stopword(c)):
            continue
        
        # Basic plural agreement - if target is plural, keep plural candidates
        if target_is_plural:
            c_is_plural = c_lower.endswith('s') and not c_lower.endswith('ss')
            if not c_is_plural:
                continue
        
        # Block obvious noun→adjective shifts (wood→wooden, feather→feathered)
        if c_lower.endswith('en') and target_lower + 'en' == c_lower:
            continue
        if c_lower.endswith('ed') and target_lower + 'ed' == c_lower:
            continue
        
        # Deduplicate
        if c_lower not in seen:
            seen.add(c_lower)
            filtered.append(c)
    
    return filtered


# Model loading
def load_word_vectors(model_name: str = "glove-wiki-gigaword-100"):
    """Load pretrained word vectors."""
    print(f"Loading word vectors: {model_name}...")
    return api.load(model_name)


def load_sentence_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load sentence transformer model."""
    print(f"Loading sentence model: {model_name}...")
    return SentenceTransformer(model_name)


# Candidate generation
def generate_candidates(word_vectors, target: str, top_k: int) -> List[str]:
    """Generate candidate synonyms using word embeddings."""
    key = target
    if key not in word_vectors:
        key = target.lower()
        if key not in word_vectors:
            return []
    
    neighbors = word_vectors.most_similar(key, topn=top_k)
    return [word for (word, _) in neighbors]


# Context-aware replacement selection
def pick_best_replacement(sbert, sentence, target, candidates):
    """Pick the candidate that keeps sentence meaning most similar."""
    if not candidates:
        return target, None
    
    # Encode original sentence
    original_emb = sbert.encode(sentence, normalize_embeddings=True)
    
    best_word = target
    best_score = -1.0
    
    # Try each candidate
    for candidate in candidates:
        modified = replace_word_once(sentence, target, candidate)
        if modified == sentence:
            continue
        
        # Encode modified sentence
        mod_emb = sbert.encode(modified, normalize_embeddings=True)
        
        # Calculate similarity (cosine similarity since normalized)
        score = float(np.dot(original_emb, mod_emb))
        
        if score > best_score:
            best_score = score
            best_word = candidate
    
    # Only replace if score meets threshold
    if best_word != target and best_score >= SIMILARITY_THRESHOLD:
        return best_word, best_score
    
    return target, None


# Main pipeline
def rewrite_text(text: str, word_vectors, sbert):
    """Main pipeline: segment sentences, select targets, and replace."""
    sentences = split_sentences(text)
    outputs = []
    
    print("\nProcessing text...")
    print("-" * 70)
    
    for sentence in sentences:
        print(f"\nOriginal: {sentence}")
        
        # Select target word
        target = select_target_word(sentence)
        if not target:
            outputs.append(sentence)
            print("  No target word found")
            continue
        
        # Generate candidates
        candidates = generate_candidates(word_vectors, target, TOP_K_CANDIDATES)
        candidates = filter_candidates(candidates, target)
        
        print(f"  Target: '{target}' ({len(candidates)} candidates)")
        
        # Pick best replacement
        chosen, score = pick_best_replacement(sbert, sentence, target, candidates)
        
        # Apply replacement
        if chosen != target:
            modified = replace_word_once(sentence, target, chosen)
            print(f"  → Replaced with '{chosen}' (similarity: {score:.3f})")
            outputs.append(modified)
        else:
            print(f"  → Kept original (no suitable replacement)")
            outputs.append(sentence)
    
    return " ".join(outputs)


# Test suite
def run_tests(word_vectors, sbert):
    """Run the system on test texts."""
    test_cases = [
        {
            "name": "Shakespeare (Metaphor & Polysemy)",
            "text": "All the world's a stage, And all the men and women merely players."
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
    
    print("\n" + "=" * 70)
    print("SYNONYM REPLACEMENT SYSTEM - TEST RESULTS")
    print("=" * 70)
    print(f"Configuration: top_k={TOP_K_CANDIDATES}, threshold={SIMILARITY_THRESHOLD}")
    
    for test in test_cases:
        print("\n" + "=" * 70)
        print(f"TEST: {test['name']}")
        print("=" * 70)
        
        modified = rewrite_text(test['text'], word_vectors, sbert)
        
        print("\n" + "-" * 70)
        print(f"FINAL OUTPUT:")
        print(f"  Original: {test['text']}")
        print(f"  Modified: {modified}")
    
    print("\n" + "=" * 70)
    print("END OF TESTS")
    print("=" * 70)


# Main execution
if __name__ == "__main__":
    # Load models once
    wv = load_word_vectors("glove-wiki-gigaword-100")
    sbert = load_sentence_model("all-MiniLM-L6-v2")
    
    # Run test suite
    run_tests(wv, sbert)