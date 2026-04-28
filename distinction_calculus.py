"""
Distinction Calculus — Spencer-Brown's primitives for Python.
Marks (#), Enclosures ([ ]), Calling (## → #), Crossing ([[A]] → A).

No floating-point. Pure syntactic pattern operations.
"""

from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict


class Token(Enum):
    MARK = "#"          # The mark: "something is here"
    ENCLOSE_L = "["     # Left enclosure boundary
    ENCLOSE_R = "]"     # Right enclosure boundary
    VOID = ""           # The absence of any mark


@dataclass
class Expr:
    """A syntactic expression — marks and nested enclosures."""
    raw: str  # The string representation, e.g. "[# [#]]"

    def __post_init__(self):
        self._validate()

    def _validate(self):
        depth = 0
        for c in self.raw:
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced enclosures in: {self.raw}")
        if depth != 0:
            raise ValueError(f"Unclosed enclosures in: {self.raw}")

    @property
    def depth(self) -> int:
        """Maximum nesting depth."""
        current = 0
        max_depth = 0
        for c in self.raw:
            if c == '[':
                current += 1
                max_depth = max(max_depth, current)
            elif c == ']':
                current -= 1
        return max_depth

    @property
    def mark_count(self) -> int:
        """Number of marks (#) in the expression."""
        return self.raw.count('#')

    def reduce_calling(self) -> 'Expr':
        """Apply Calling: ## → #"""
        result = self.raw
        while '##' in result:
            result = result.replace('##', '#')
        return Expr(result)

    def reduce_crossing(self) -> 'Expr':
        """Apply Crossing: [[A]] → A (remove redundant double enclosures)."""
        result = self.raw
        # Pattern: [[ followed by anything balanced, then ]]
        pattern = re.compile(r'\[\[([^\[\]]*(?:\[[^\]]*\][^\[\]]*)*)\]\]')
        prev = None
        while prev != result:
            prev = result
            result = pattern.sub(r'\1', result)
        return Expr(result)

    def normal_form(self) -> 'Expr':
        """Reduce to normal form by exhaustive Calling + Crossing."""
        current = self.reduce_calling().reduce_crossing()
        prev = None
        while prev != current.raw:
            prev = current.raw
            current = current.reduce_calling().reduce_crossing()
        return current

    def is_stable(self) -> bool:
        """Is this expression a stable (irreducible) pattern?"""
        nf = self.normal_form()
        return nf.raw == self.raw

    def to_tree_path(self) -> List[int]:
        """
        Convert to a tree path (sequence of nesting levels).
        Each # contributes a mark at the current nesting depth.
        """
        path = []
        depth = 0
        i = 0
        while i < len(self.raw):
            c = self.raw[i]
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
            elif c == '#':
                path.append(depth)
            i += 1
        return path

    def __str__(self):
        return self.raw

    def __repr__(self):
        return f"Expr({self.raw!r})"


def juxtapose(*exprs: Expr) -> Expr:
    """Juxtapose multiple expressions side by side."""
    combined = ''.join(e.raw for e in exprs)
    return Expr(combined)


def enclose(expr: Expr) -> Expr:
    """Wrap an expression in an enclosure."""
    return Expr(f"[{expr.raw}]")


def mark() -> Expr:
    """Create a single mark."""
    return Expr("#")


# Known stable particle patterns (from QLoF monograph)
STABLE_PATTERNS = {
    "photon": Expr("[#]"),
    "electron": Expr("[# [#]]"),
    "up_quark": Expr("[[#] #]"),
    "down_quark": Expr("[[#] [#] #]"),
    "w_boson": Expr("[[#] [#]]"),
    "z_higgs": Expr("[[#] [#] [#]]"),
}

# Semantic primes encoded as patterns
SEMANTIC_PRIME_PATTERNS = {
    "good": Expr("[#]"),
    "bad": Expr("[# [#]]"),
    "not": Expr("[[#]]"),
    "very": Expr("[##]"),
    "but": Expr("[# [#] [#]]"),
}


def tree_distance(e1: Expr, e2: Expr) -> int:
    """
    Compute ultrametric distance between two expressions
    as the number of Crossing steps to reach their LCA.
    """
    p1 = e1.to_tree_path()
    p2 = e2.to_tree_path()
    
    # Pad shorter path with zeros
    max_len = max(len(p1), len(p2))
    p1 = p1 + [0] * (max_len - len(p1))
    p2 = p2 + [0] * (max_len - len(p2))
    
    # Find LCA depth: first position where paths diverge
    lca_depth = 0
    for a, b in zip(p1, p2):
        if a == b and a > 0:
            lca_depth = max(lca_depth, a)
    
    # Distance = sum of depths to reach LCA
    d1 = sum(1 for d in p1 if d > lca_depth)
    d2 = sum(1 for d in p2 if d > lca_depth)
    return d1 + d2


def strong_triangle_inequality(e1: Expr, e2: Expr, e3: Expr) -> bool:
    """
    Check the strong triangle inequality (ultrametric):
    d(a,b) <= max(d(a,c), d(b,c))
    """
    d12 = tree_distance(e1, e2)
    d13 = tree_distance(e1, e3)
    d23 = tree_distance(e2, e3)
    return d12 <= max(d13, d23)


__all__ = ['Expr', 'Token', 'mark', 'enclose', 'juxtapose', 
           'tree_distance', 'strong_triangle_inequality',
           'STABLE_PATTERNS', 'SEMANTIC_PRIME_PATTERNS']
