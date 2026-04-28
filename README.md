# Ultrametric AI — Proof of Concept

**Token-level explainable attention via p-adic valuation trees and Spencer-Brown's Laws of Form.**

A working web application demonstrating ultrametric attention, distinction calculus, and cocycle cognitive architecture — all in one interactive interface.

---

## What This Demonstrates

| Module | Description |
|---|---|
| **Ultrametric Attention** | Attention weights computed from p-adic valuation distances on the Bruhat-Tits tree. Fully auditable — every weight has a traceable LCA path. |
| **Distinction Calculus** | Spencer-Brown's primitives (mark `#`, enclosure `[ ]`, Calling `##→#`, Crossing `[[A]]→A`) applied to attention. Pure syntactic computation. |
| **Cocycle Auditor** | Verifies the strong triangle inequality across token sets. The cocycle condition must hold for cognitive consistency. |
| **Particle Zoo** | Explores stable syntactic patterns mapped to Standard Model particles. |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## How It Works

1. **Token Encoding**: Each word is assigned semantic primes (good, bad, not, very, but) via WordNet hypernyms. The prime product encodes the word as an integer whose p-adic valuations form a vector.

2. **Ultrametric Distance**: Distance between two tokens is `max|v_p(a) - v_p(b)|` across all primes — this satisfies the strong triangle inequality (ultrametric property).

3. **Attention**: Similarity = `exp(-distance / temperature)`. Normalized row-wise. No learned parameters — the attention pattern emerges purely from the tree geometry.

4. **Cocycle Check**: For any three tokens, `d(a,b) ≤ max(d(a,c), d(b,c))` must hold. Violations indicate inconsistency in the cognitive space.

## Architecture

```
ultrametric-ai-poc/
├── app.py                    # Streamlit web application (4 modes)
├── model.py                  # Attention models (ultrametric + distinction)
├── distinction_calculus.py   # Spencer-Brown primitives + tree operations
├── cocycle.py                # Cocycle verification + valuation utils
├── requirements.txt          # Python dependencies
├── README.md                 # This file
```

## Based On

- **Quantum Laws of Form** — Syntactic Token Calculus generating the Standard Model
- **Laws of Form** — George Spencer-Brown (1969)
- **Bruhat-Tits Tree** — Ultrametric geometry from p-adic numbers
- **Cocycle Cognitive Architecture** — Brain as maintaining cognitive consistency

## License

Foundational research tool. Free. No capture.
