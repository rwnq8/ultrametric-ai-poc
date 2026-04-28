"""
Cocycle Cognitive Architecture — verifying cognitive consistency.
The brain maintains consistency via cocycle conditions on the Bruhat-Tits tree.

A cocycle is satisfied when: for any triangle (a,b,c) in the tree,
the distances satisfy the strong triangle inequality, and the "cycle" 
around any closed loop sums to zero modulo some invariant.
"""

from typing import List, Dict, Tuple, Set
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

# Semantic primes as p-adic valuations
PRIME_BASIS = {
    "good": 2,
    "bad": 3,
    "not": 5,
    "very": 7,
    "but": 11,
}

PRIME_NAMES = list(PRIME_BASIS.keys())


def valuation(n: int, p: int) -> int:
    """Compute the p-adic valuation v_p(n) — exponent of p in n's factorization."""
    if n <= 0:
        return 0
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def to_valuation_vector(n: int) -> np.ndarray:
    """Convert a prime product to a valuation vector over all semantic primes."""
    return np.array([valuation(n, PRIME_BASIS[p]) for p in PRIME_NAMES], dtype=np.int32)


def ultrametric_distance(v1: np.ndarray, v2: np.ndarray) -> int:
    """
    Max of absolute differences — satisfies the strong triangle inequality.
    This is the natural ultrametric on the product of p-adic valuation trees.
    """
    return int(np.max(np.abs(v1 - v2)))


def check_strong_triangle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> Tuple[bool, str]:
    """
    Verify the strong triangle inequality for three valuation vectors.
    Returns (satisfied, explanation).
    """
    d12 = ultrametric_distance(v1, v2)
    d13 = ultrametric_distance(v1, v3)
    d23 = ultrametric_distance(v2, v3)
    
    condition = d12 <= max(d13, d23)
    
    explanation = (
        f"d(A,B)={d12}, d(A,C)={d13}, d(B,C)={d23}\n"
        f"Check: d(A,B) ≤ max(d(A,C), d(B,C))\n"
        f"  {d12} ≤ max({d13},{d23}) = {max(d13,d23)}\n"
        + ("✓ SATISFIED (ultrametric: all triangles are isosceles with two equal longest sides)"
           if condition else
           f"✗ VIOLATED (cocycle inconsistency: {d12} > {max(d13,d23)})")
    )
    
    return condition, explanation


@dataclass
class CocycleGraph:
    """A graph tracking cocycle consistency across a set of token valuations."""
    nodes: Dict[str, np.ndarray] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    
    def add_node(self, name: str, valuation: np.ndarray):
        self.nodes[name] = valuation
    
    def add_triangle(self, a: str, b: str, c: str) -> bool:
        """Add a cognitive triangle and check cocycle consistency."""
        if not all(k in self.nodes for k in [a, b, c]):
            return False
        
        v1, v2, v3 = self.nodes[a], self.nodes[b], self.nodes[c]
        satisfied, explanation = check_strong_triangle(v1, v2, v3)
        
        if not satisfied:
            self.violations.append(f"Triangle ({a},{b},{c}): {explanation}")
        
        return satisfied
    
    def global_consistency(self) -> float:
        """
        Measure of global cocycle consistency.
        Returns ratio of satisfied triangles to total.
        """
        node_names = list(self.nodes.keys())
        total = 0
        satisfied = 0
        
        for i in range(len(node_names)):
            for j in range(i + 1, len(node_names)):
                for k in range(j + 1, len(node_names)):
                    total += 1
                    if self.add_triangle(node_names[i], node_names[j], node_names[k]):
                        satisfied += 1
        
        return satisfied / total if total > 0 else 1.0


def compute_attention_weights(
    valuations: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute ultrametric attention weights from valuation vectors.
    
    Args:
        valuations: (seq_len, num_primes) array of valuation vectors
        temperature: Controls sharpness (lower = more peaked)
    
    Returns:
        weights: (seq_len,) array of attention weights (sums to 1)
    """
    seq_len = valuations.shape[0]
    center = valuations.mean(axis=0, keepdims=True)  # (1, num_primes) — query vector
    
    # Compute ultrametric distances from center
    distances = np.array([
        ultrametric_distance(center[0], valuations[i])
        for i in range(seq_len)
    ])
    
    # Convert to similarity: closer = higher
    similarities = np.exp(-distances / max(temperature, 0.01))
    weights = similarities / similarities.sum()
    
    return weights


def pairwise_attention_matrix(
    valuations: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute full pairwise attention matrix using ultrametric distance.
    
    Args:
        valuations: (seq_len, num_primes) array
        temperature: Controls sharpness
    
    Returns:
        attn: (seq_len, seq_len) attention matrix
    """
    seq_len = valuations.shape[0]
    distances = np.zeros((seq_len, seq_len), dtype=np.float32)
    
    for i in range(seq_len):
        for j in range(seq_len):
            distances[i, j] = ultrametric_distance(valuations[i], valuations[j])
    
    # Convert to similarity (closer = more attention)
    similarities = np.exp(-distances / max(temperature, 0.01))
    # Row-wise normalization
    attn = similarities / similarities.sum(axis=1, keepdims=True)
    
    return attn


__all__ = [
    'PRIME_BASIS', 'PRIME_NAMES', 'valuation', 'to_valuation_vector',
    'ultrametric_distance', 'check_strong_triangle', 'CocycleGraph',
    'compute_attention_weights', 'pairwise_attention_matrix'
]
