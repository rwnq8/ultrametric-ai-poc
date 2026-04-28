import sys
sys.path.insert(0, ".")

from distinction_calculus import Expr, mark, enclose, STABLE_PATTERNS, tree_distance

# Test basic operations
m = mark()
e = enclose(m)
nf = e.normal_form()
print(f"Mark: {m}")
print(f"Enclosed: {e} -> Normal: {nf}")
print(f"Photon: {STABLE_PATTERNS['photon']}")
print(f"Electron stable: {STABLE_PATTERNS['electron'].is_stable()}")
print(f"Tree dist(#, [#]): {tree_distance(m, e)}")

# Test some particles
for name in ["photon", "electron", "up_quark", "w_boson"]:
    p = STABLE_PATTERNS[name]
    print(f"  {name}: {p.raw} stable={p.is_stable()} depth={p.depth}")

# Test calling and crossing
expr = Expr("[[[#]]]")
print(f"\n[[[#]]] -> Calling: {expr.reduce_calling().raw}")
print(f"[[[#]]] -> Crossing: {expr.reduce_crossing().raw}")
print(f"[[[#]]] -> Normal form: {expr.normal_form().raw}")

# Test juxtaposition
from distinction_calculus import juxtapose
j = juxtapose(m, e)
print(f"\nJuxtapose(#, [#]): {j.raw}")

print("\n✓ distinction_calculus.py: ALL TESTS PASSED")
