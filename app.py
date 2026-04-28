"""
Ultrametric AI — Proof of Concept Web App

Demonstrates:
1. Ultrametric attention mechanism (tree-based, auditable)
2. Distinction calculus primitives (Spencer-Brown marks/enclosures)
3. Cocycle cognitive architecture (strong triangle inequality verification)
4. Interactive visualization of the Bruhat-Tits tree

Based on: Quantum Laws of Form, Ultrametric Cognitive Architecture
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from io import BytesIO

from model import (
    word_to_primes, word_to_product, sentence_to_valuations,
    UltrametricAttentionModel, DistinctionAttentionModel,
    PRIME_NAMES, PRIME_BASIS,
)
from cocycle import (
    to_valuation_vector, ultrametric_distance,
    check_strong_triangle, CocycleGraph,
    pairwise_attention_matrix,
)
from distinction_calculus import (
    Expr, mark, enclose, SEMANTIC_PRIME_PATTERNS, STABLE_PATTERNS
)

# ─── Page Config ───
st.set_page_config(
    page_title="Ultrametric AI — PoC",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("## 🌳 Ultrametric AI PoC")
    st.markdown("*Tree-based explainable attention*")
    st.divider()
    
    mode = st.radio(
        "**Mode**",
        ["Ultrametric Attention", "Distinction Calculus", "Cocycle Auditor", "Particle Zoo"],
        help="""
- **Ultrametric Attention**: Compute attention using p-adic valuation trees
- **Distinction Calculus**: Syntactic attention using Spencer-Brown primitives
- **Cocycle Auditor**: Verify strong triangle inequality across token sets
- **Particle Zoo**: Explore stable syntactic patterns
"""
    )
    
    temperature = st.slider("Temperature", 0.1, 5.0, 1.0, 0.1,
                           help="Lower = sharper attention. Higher = more diffuse.")
    
    st.divider()
    st.caption("Built on: Quantum Laws of Form · Spencer-Brown 1969 · Bruhat-Tits Tree")
    st.caption("[GitHub](https://github.com/rwnq8/ultrametric-ai-poc)")

# ─── Main Content ───
st.title("Ultrametric AI — Proof of Concept")
st.markdown("*Token-level explainable attention via p-adic valuation trees and Spencer-Brown's Laws of Form*")

# ────────────────────────────────────────────────────────────────────
# MODE 1: Ultrametric Attention
# ────────────────────────────────────────────────────────────────────
if mode == "Ultrametric Attention":
    st.header("Ultrametric Attention")
    st.markdown("""
    Enter a sentence. Each word is assigned **semantic primes** (good, bad, not, very, but) 
    via WordNet hypernyms. The ultrametric distance between words comes from their 
    p-adic valuation vectors — the depth of their lowest common ancestor on the 
    Bruhat-Tits product tree.
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        sentence = st.text_input("Sentence", "This movie was very good but not great")
    with col2:
        run_btn = st.button("▶ Compute Attention", type="primary", use_container_width=True)
    
    if run_btn and sentence:
        model = UltrametricAttentionModel(temperature=temperature)
        result = model.forward(sentence)
        
        valuations, tokens, products = sentence_to_valuations(sentence)
        
        if len(tokens) == 0:
            st.warning("No recognizable tokens found.")
        else:
            # ── Token Breakdown ──
            st.subheader("Token → Prime Encoding")
            
            cols = st.columns(min(len(tokens), 6))
            for idx, (token, prod, col) in enumerate(zip(tokens, products, cols * len(tokens))):
                with col:
                    primes = word_to_primes(token)
                    st.metric(token, f"∏={prod}", 
                             f"Primes: {','.join(primes) if primes else 'none'}")
            
            # ── Attention Heatmap ──
            st.subheader("Attention Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            if len(tokens) > 1:
                sns.heatmap(
                    result.attn_matrix,
                    annot=True, fmt=".2f",
                    xticklabels=tokens, yticklabels=tokens,
                    cmap="YlOrRd", ax=ax, cbar_kws={'label': 'Attention Weight'}
                )
                ax.set_title("Ultrametric Attention (row → attends to column)")
            else:
                ax.text(0.5, 0.5, "Need ≥2 tokens", ha='center', va='center',
                       transform=ax.transAxes)
            st.pyplot(fig)
            plt.close()
            
            # ── Token Attention Distribution ──
            st.subheader("Token Attention Received")
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['#2a6e4f' if v > np.median(result.token_attention) else '#b0562a' 
                     for v in result.token_attention]
            ax.bar(range(len(tokens)), result.token_attention, color=colors)
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_ylabel("Total Attention Received")
            ax.set_title("Which tokens does the model attend to most?")
            st.pyplot(fig)
            plt.close()
            
            # ── Prime Contributions ──
            st.subheader("Semantic Prime Contributions")
            cols = st.columns(len(PRIME_NAMES))
            for i, (p_name, contrib) in enumerate(result.prime_contributions.items()):
                with cols[i]:
                    st.metric(
                        p_name.upper(),
                        f"{contrib:.2f}",
                        f"Prime={PRIME_BASIS[p_name]}"
                    )
            
            # ── Top Pairs ──
            st.subheader("Top Attending Pairs")
            for i, j, w in result.top_pairs:
                st.markdown(
                    f"**{tokens[i]}** ↔ **{tokens[j]}** "
                    f"— attention weight: **{w:.3f}** "
                    f"(ultrametric distance: {ultrametric_distance(valuations[i], valuations[j])})"
                )
            
            # ── Cocycle Check ──
            st.subheader("Cocycle Consistency")
            if result.cocycle_satisfied:
                st.success(f"✓ Strong triangle inequality satisfied\n\n{result.cocycle_explanation}")
            else:
                st.error(f"✗ Violation detected\n\n{result.cocycle_explanation}")

# ────────────────────────────────────────────────────────────────────
# MODE 2: Distinction Calculus
# ────────────────────────────────────────────────────────────────────
elif mode == "Distinction Calculus":
    st.header("Distinction Calculus")
    st.markdown("""
    Spencer-Brown's **Laws of Form** begins with a single instruction: *Draw a distinction.*  
    From marks (#) and enclosures [ ], and two rules (Calling: `##→#`, Crossing: `[[A]]→A`),
    we derive everything — including attention weights.
    
    Below, tokens are encoded as nested enclosures. Attention is computed by measuring 
    the number of Crossing steps to reach the lowest common ancestor (LCA).
    """)
    
    st.subheader("Primitives and Rules")
    cols = st.columns(4)
    with cols[0]:
        st.markdown("**Mark** `#`")
        st.code("#")
        st.caption("'Something is here'")
    with cols[1]:
        st.markdown("**Enclosure** `[ ]`")
        st.code("[#]")
        st.caption("Creates inside/outside")
    with cols[2]:
        st.markdown("**Calling**")
        st.code("## → #")
        st.caption("Redundancy condenses")
    with cols[3]:
        st.markdown("**Crossing**")
        st.code("[[A]] → A")
        st.caption("Boundaries cancel")
    
    st.divider()
    
    # Interactive expression builder
    st.subheader("Build an Expression")
    col1, col2, col3 = st.columns(3)
    with col1:
        nesting = st.slider("Nesting depth", 0, 5, 2)
    with col2:
        marks = st.slider("Number of marks", 1, 5, 1)
    with col3:
        build_btn = st.button("Build & Reduce", type="primary")
    
    if build_btn:
        # Build expression: N marks at the innermost level
        inner = '#' * marks
        expr_str = inner
        for _ in range(nesting):
            expr_str = f"[{expr_str}]"
        
        expr = Expr(expr_str)
        nf = expr.normal_form()
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Original", expr.raw)
            st.metric("Depth", expr.depth)
            st.metric("Marks", expr.mark_count)
        with col_b:
            st.metric("Normal Form", nf.raw)
            st.metric("Depth", nf.depth)
            st.metric("Marks", nf.mark_count)
            st.metric("Stable?", "✓ Yes" if expr.is_stable() else "→ Reduced")
        
        # Check against known stable patterns
        for name, pattern in STABLE_PATTERNS.items():
            if nf.raw == pattern.raw:
                st.success(f"Matches known particle: **{name}** ({pattern.raw})")
        
        # Reduction trace
        st.subheader("Reduction Trace")
        current = expr
        step = 0
        while True:
            calling = current.reduce_calling()
            crossing = current.reduce_crossing()
            
            st.markdown(f"Step {step}: `{current.raw}`")
            st.markdown(f"  Calling: `{calling.raw}`")
            st.markdown(f"  Crossing: `{crossing.raw}`")
            
            next_expr = calling.reduce_crossing()
            if next_expr.raw == current.raw:
                break
            current = next_expr
            step += 1
        st.markdown(f"**Normal form reached**: `{current.raw}` (stable: {current.is_stable()})")
    
    st.divider()
    
    # Token-wise distinction encoding
    st.subheader("Sentence → Distinction Expressions")
    dc_sentence = st.text_input("Sentence for DC encoding", "very good but not great")
    
    if st.button("Encode Sentence"):
        from model import sentence_to_exprs
        exprs = sentence_to_exprs(dc_sentence)
        tokens = [e.raw for e in exprs]
        
        # Distance matrix
        n = len(exprs)
        if n > 1:
            st.subheader("Expression → Normal Form → Tree Distance")
            for i, expr in enumerate(exprs):
                nf = expr.normal_form()
                st.markdown(
                    f"**{tokens[i]}** → `{nf.raw}` "
                    f"(depth={expr.depth}, marks={expr.mark_count}, "
                    f"stable={expr.is_stable()})"
                )
            
            # Distance table
            import pandas as pd
            data = {}
            for i in range(n):
                row = {}
                for j in range(n):
                    from distinction_calculus import tree_distance
                    row[tokens[j]] = tree_distance(exprs[i], exprs[j])
                data[tokens[i]] = row
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

# ────────────────────────────────────────────────────────────────────
# MODE 3: Cocycle Auditor
# ────────────────────────────────────────────────────────────────────
elif mode == "Cocycle Auditor":
    st.header("Cocycle Auditor")
    st.markdown("""
    The **cocycle cognitive architecture** hypothesis: neural representations maintain 
    cognitive consistency via cocycle conditions on the Bruhat-Tits tree.
    
    The **strong triangle inequality** must hold: $d(a,b) \\leq \\max(d(a,c), d(b,c))$  
    for any three concepts in semantic space. This means every triangle is **ultrametric** —
    all triangles are isosceles with at most two equal longest sides.
    """)
    
    # Interactive triangle check
    st.subheader("Triangle Cocycle Check")
    st.markdown("Enter three words to check the strong triangle inequality.")
    
    cols = st.columns(3)
    with cols[0]:
        word_a = st.text_input("Word A", "good")
    with cols[1]:
        word_b = st.text_input("Word B", "bad")
    with cols[2]:
        word_c = st.text_input("Word C", "great")
    
    if st.button("Verify Triangle", type="primary"):
        from model import word_to_product
        prod_a = word_to_product(word_a)
        prod_b = word_to_product(word_b)
        prod_c = word_to_product(word_c)
        
        v1 = to_valuation_vector(prod_a)
        v2 = to_valuation_vector(prod_b)
        v3 = to_valuation_vector(prod_c)
        
        d12 = ultrametric_distance(v1, v2)
        d13 = ultrametric_distance(v1, v3)
        d23 = ultrametric_distance(v2, v3)
        
        condition = d12 <= max(d13, d23)
        
        cols = st.columns(3)
        with cols[0]:
            st.metric(f"d({word_a},{word_b})", d12)
            st.caption(f"Valuation: {list(v1)}")
        with cols[1]:
            st.metric(f"d({word_a},{word_c})", d13)
            st.caption(f"Valuation: {list(v2)}")
        with cols[2]:
            st.metric(f"d({word_b},{word_c})", d23)
            st.caption(f"Valuation: {list(v3)}")
        
        if condition:
            st.success(f"✓ Cocycle satisfied: {d12} ≤ max({d13}, {d23}) = {max(d13, d23)}")
            sides = sorted([d12, d13, d23])
            if sides[0] == sides[1]:
                st.info("Triangle is isosceles (ultrametric property)")
        else:
            st.error(f"✗ Violation: {d12} > max({d13}, {d23}) = {max(d13, d23)}")
    
    st.divider()
    
    # Bulk cocycle check
    st.subheader("Bulk Consistency Audit")
    audit_sentence = st.text_input("Sentence for bulk audit", 
                                   "the movie was very good but the acting was terrible")
    
    if st.button("Run Audit"):
        valuations, tokens, products = sentence_to_valuations(audit_sentence)
        graph = CocycleGraph()
        for i, token in enumerate(tokens):
            graph.add_node(f"{token}[{i}]", valuations[i])
        
        consistency = graph.global_consistency()
        st.metric("Global Cocycle Consistency", f"{consistency:.1%}")
        
        if len(graph.violations) > 0:
            st.warning(f"Found {len(graph.violations)} cocycle violations:")
            for v in graph.violations[:3]:
                st.code(v)
        else:
            st.success("All triangles satisfy the strong triangle inequality — ultrametric space confirmed.")
        
        # Show the distance matrix
        n = len(tokens)
        distances = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = ultrametric_distance(valuations[i], valuations[j])
        
        st.subheader("Ultrametric Distance Matrix")
        import pandas as pd
        df = pd.DataFrame(distances, index=tokens, columns=tokens)
        st.dataframe(df)

# ────────────────────────────────────────────────────────────────────
# MODE 4: Particle Zoo
# ────────────────────────────────────────────────────────────────────
elif mode == "Particle Zoo":
    st.header("Particle Zoo — Stable Syntactic Patterns")
    st.markdown("""
    In the Syntactic Token Calculus, **particles are stable patterns** — expressions that 
    cannot be reduced by Calling or Crossing. Each stable pattern is a particle candidate.
    
    Below are the known stable patterns and their mapping to Standard Model particles.
    """)
    
    cols = st.columns(3)
    for i, (name, expr) in enumerate(STABLE_PATTERNS.items()):
        with cols[i % 3]:
            nf = expr.normal_form()
            st.markdown(f"### {name.replace('_', ' ').title()}")
            st.code(expr.raw)
            st.caption(f"Depth: {expr.depth} | Marks: {expr.mark_count} | Stable: {expr.is_stable()}")
            
            # Tree representation
            fig, ax = plt.subplots(figsize=(3, 3))
            G = nx.DiGraph()
            G.add_node("root", color='#f0f0f0')
            
            depth = 0
            stack = [("root", depth)]
            node_count = 1
            
            for c in expr.raw:
                if c == '[':
                    depth += 1
                    node_name = f"n{node_count}"
                    G.add_node(node_name, color='#d0e0f0' if depth % 2 == 1 else '#f0e0d0')
                    parent = stack[-1][0]
                    G.add_edge(parent, node_name)
                    stack.append((node_name, depth))
                    node_count += 1
                elif c == '#':
                    node_name = f"m{node_count}"
                    G.add_node(node_name, color='#e07050')
                    parent = stack[-1][0]
                    G.add_edge(parent, node_name)
                    node_count += 1
                elif c == ']':
                    if len(stack) > 1:
                        stack.pop()
                        depth -= 1
            
            colors = [G.nodes[n].get('color', '#ccc') for n in G.nodes]
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, node_color=colors, node_size=200, 
                   font_size=8, ax=ax, with_labels=False)
            ax.set_title(f"{name}")
            st.pyplot(fig)
            plt.close()
    
    # Semantic primes as patterns
    st.divider()
    st.subheader("Semantic Primes as Patterns")
    for p_name, expr in SEMANTIC_PRIME_PATTERNS.items():
        cols = st.columns([1, 3])
        with cols[0]:
            st.metric(p_name.upper(), f"Prime={PRIME_BASIS[p_name]}")
        with cols[1]:
            st.code(expr.raw)
            st.caption(f"Depth: {expr.depth} | Stable: {'✓' if expr.is_stable() else '→ unstable'}")

# ─── Footer ───
st.divider()
st.caption(
    "**Ultrametric AI Proof of Concept** — "
    "Based on Spencer-Brown's *Laws of Form* (1969), "
    "the Bruhat-Tits tree, and the cocycle cognitive architecture. "
    "No learned parameters. No black boxes. Auditable attention."
)
