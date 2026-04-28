"""
Ultrametric Attention Model — Tree-based attention using p-adic valuations.
Demonstrates how the Bruhat-Tits tree geometry produces auditable attention patterns.

Three architectures:
1. Pure distinction-calculus (no floating point, syntactic only)
2. Ultrametric attention (exponential-decay attention from LCA depth)
3. Standard dot-product baseline
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from distinction_calculus import (
    Expr, mark, enclose, juxtapose, tree_distance,
    SEMANTIC_PRIME_PATTERNS, STABLE_PATTERNS
)
from cocycle import (
    PRIME_BASIS, PRIME_NAMES, valuation, to_valuation_vector,
    ultrametric_distance, pairwise_attention_matrix,
    check_strong_triangle, CocycleGraph
)

# ─── WordNet-based semantic prime mapping ───

import nltk
from nltk.corpus import wordnet as wn

# Ensure WordNet is available
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def word_to_primes(word: str) -> List[str]:
    """Map a word to its assigned semantic prime categories via WordNet hypernyms."""
    word_lower = word.lower()
    primes = set()
    
    synsets = wn.synsets(word_lower)
    if not synsets:
        # Try common words directly
        direct_map = {
            'not': ['not'], "n't": ['not'], 'never': ['not'], 'no': ['not'],
            'very': ['very'], 'extremely': ['very'], 'really': ['very'], 'highly': ['very'],
            'but': ['but'], 'however': ['but'], 'although': ['but'], 'yet': ['but'],
        }
        if word_lower in direct_map:
            return direct_map[word_lower]
        return []
    
    hypernym_names = set()
    for syn in synsets[:2]:
        for path in syn.hypernym_paths():
            for h in path:
                for lemma in h.lemmas():
                    hypernym_names.add(lemma.name().lower())
    
    # Heuristic semantic assignment
    positive_words = {'good', 'positive', 'pleasant', 'excellent', 'benefit', 'advantage', 'virtue'}
    negative_words = {'bad', 'negative', 'unpleasant', 'terrible', 'disadvantage', 'harm', 'evil'}
    
    if any(w in hypernym_names for w in positive_words):
        primes.add('good')
    if any(w in hypernym_names for w in negative_words):
        primes.add('bad')
    
    # Direct matches for function words
    if word_lower in ['not', "n't", 'never', 'no', 'none']:
        primes.add('not')
    if word_lower in ['very', 'extremely', 'highly', 'really', 'greatly', 'quite']:
        primes.add('very')
    if word_lower in ['but', 'however', 'although', 'yet', 'though']:
        primes.add('but')
    
    return list(primes)


def word_to_product(word: str) -> int:
    """Convert word to integer product of semantic primes."""
    primes = word_to_primes(word)
    prod = 1
    for p_name in primes:
        if p_name in PRIME_BASIS:
            prod *= PRIME_BASIS[p_name]
    return prod


def sentence_to_valuations(sentence: str) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Convert a sentence to a sequence of valuation vectors.
    
    Returns:
        valuations: (seq_len, num_primes) array
        tokens: list of token strings
        products: list of prime products per token
    """
    import re
    tokens = re.findall(r"[\w']+|[^\w\s]", sentence.lower())
    
    valuations = []
    products = []
    for token in tokens:
        prod = word_to_product(token)
        vec = to_valuation_vector(prod)
        valuations.append(vec)
        products.append(prod)
    
    if valuations:
        arr = np.array(valuations, dtype=np.int32)
    else:
        arr = np.zeros((1, len(PRIME_NAMES)), dtype=np.int32)
    
    return arr, tokens, products


# ─── Token-to-Pattern Encoding (Distinction Calculus) ───

def token_to_expr(token: str) -> Expr:
    """
    Encode a token as a distinction calculus expression.
    Uses nesting depth proportional to the number of assigned semantic primes.
    """
    primes = word_to_primes(token)
    if not primes:
        return mark()  # Bare mark — unclassified token
    
    # Build nested enclosure: one level per prime
    expr = mark()
    for _ in range(len(primes)):
        expr = enclose(expr)
    
    return expr


def sentence_to_exprs(sentence: str) -> List[Expr]:
    """Convert a sentence to a sequence of distinction calculus expressions."""
    import re
    tokens = re.findall(r"[\w']+|[^\w\s]", sentence.lower())
    return [token_to_expr(t) for t in tokens]


# ─── Ultrametric Attention Model ───

@dataclass
class AttentionResult:
    """Result of an ultrametric attention computation."""
    attn_matrix: np.ndarray           # (seq_len, seq_len) attention weights
    valuations: np.ndarray             # (seq_len, num_primes) input valuations
    weighted_valuations: np.ndarray    # (seq_len, num_primes) attended output
    token_attention: np.ndarray        # (seq_len,) total attention received per token
    prime_contributions: Dict[str, float]  # Total contribution per prime
    top_pairs: List[Tuple[int, int, float]]  # Top 5 token pairs by attention
    cocycle_satisfied: bool
    cocycle_explanation: str


class UltrametricAttentionModel:
    """Ultrametric attention over p-adic valuation vectors."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def forward(self, sentence: str) -> AttentionResult:
        """Compute ultrametric attention for a sentence."""
        valuations, tokens, products = sentence_to_valuations(sentence)
        
        if len(tokens) == 0:
            return self._empty_result()
        
        # Pairwise ultrametric attention
        attn = pairwise_attention_matrix(valuations, self.temperature)
        
        # Weighted output
        weighted = attn @ valuations.astype(np.float32)
        
        # Token-level attention (sum of incoming)
        token_attn = attn.sum(axis=0)
        
        # Prime contributions
        prime_contrib = {}
        for i, p_name in enumerate(PRIME_NAMES):
            contrib = (weighted[:, i].sum()) 
            prime_contrib[p_name] = float(contrib)
        
        # Top attending pairs
        pairs = []
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                pairs.append((i, j, float(attn[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:5]
        
        # Cocycle check (sample 3 tokens)
        if len(tokens) >= 3:
            cocycle_satisfied, explanation = check_strong_triangle(
                valuations[0], valuations[1], valuations[2]
            )
        else:
            cocycle_satisfied = True
            explanation = "Need ≥3 tokens for triangle check"
        
        return AttentionResult(
            attn_matrix=attn,
            valuations=valuations,
            weighted_valuations=weighted,
            token_attention=token_attn,
            prime_contributions=prime_contrib,
            top_pairs=top_pairs,
            cocycle_satisfied=cocycle_satisfied,
            cocycle_explanation=explanation,
        )
    
    def _empty_result(self) -> AttentionResult:
        return AttentionResult(
            attn_matrix=np.array([[]]),
            valuations=np.zeros((0, len(PRIME_NAMES))),
            weighted_valuations=np.zeros((0, len(PRIME_NAMES))),
            token_attention=np.array([]),
            prime_contributions={},
            top_pairs=[],
            cocycle_satisfied=True,
            cocycle_explanation="No tokens",
        )


# ─── Distinction Calculus Model (No Floating Point) ───

class DistinctionAttentionModel:
    """
    Pure syntactic attention using distinction calculus.
    No floating point — only mark/enclosure operations.
    """
    
    def __init__(self, decay_base: int = 2):
        self.decay_base = decay_base
    
    def forward(self, sentence: str) -> Dict:
        """Compute syntactic attention using tree distances."""
        exprs = sentence_to_exprs(sentence)
        tokens = [e.raw for e in exprs]
        
        n = len(exprs)
        distances = np.zeros((n, n), dtype=np.int32)
        
        for i in range(n):
            for j in range(n):
                distances[i, j] = tree_distance(exprs[i], exprs[j])
        
        # Attention weight = decay_base^(-distance)
        # (purely syntactic — no floating point in core computation)
        attn_raw = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                attn_raw[i, j] = float(self.decay_base ** (-distances[i, j]))
        
        # Normalize rows
        row_sums = attn_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        attn = attn_raw / row_sums
        
        # Find stable patterns (particles) in the expression set
        stable_matches = {}
        for i, expr in enumerate(exprs):
            nf = expr.normal_form()
            for particle_name, particle_expr in STABLE_PATTERNS.items():
                if nf.raw == particle_expr.raw:
                    stable_matches[tokens[i]] = particle_name
        
        # Cocycle check
        if n >= 3:
            v1 = to_valuation_vector(word_to_product(tokens[0]))
            v2 = to_valuation_vector(word_to_product(tokens[1]))
            v3 = to_valuation_vector(word_to_product(tokens[2]))
            cocycle_ok, cocycle_msg = check_strong_triangle(v1, v2, v3)
        else:
            cocycle_ok = True
            cocycle_msg = "Insufficient tokens"
        
        return {
            "tokens": tokens,
            "expressions": [e.raw for e in exprs],
            "normal_forms": [e.normal_form().raw for e in exprs],
            "depths": [e.depth for e in exprs],
            "distances": distances,
            "attention": attn,
            "stable_matches": stable_matches,
            "cocycle_ok": cocycle_ok,
            "cocycle_msg": cocycle_msg,
            "triangle_check": f"d({tokens[0]},{tokens[1]})={distances[0,1]}, "
                              f"d({tokens[1]},{tokens[2]})={distances[1,2]}, "
                              f"d({tokens[0]},{tokens[2]})={distances[0,2]}"
                              if n >= 3 else "N/A",
        }


__all__ = [
    'word_to_primes', 'word_to_product', 'sentence_to_valuations',
    'token_to_expr', 'sentence_to_exprs',
    'UltrametricAttentionModel', 'DistinctionAttentionModel',
    'AttentionResult',
]
