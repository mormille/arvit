# Phase 4 — Historical environment recovery

## Status

Both archived ARViT source variants can be instantiated and executed on a CPU in a reconstructed legacy environment. This does not yet create the canonical reusable `arvit` package planned for Phase 5.

## Verified compatibility environment

The GitHub Actions smoke test uses:

- Ubuntu 22.04 CPU runner
- Python 3.8
- PyTorch 1.8.1 CPU
- torchvision 0.9.1 CPU
- FastAI 2.3.1
- FastCore 1.3.20
- spaCy 2.3.1
- NumPy 1.20.3
- SciPy 1.6.3

The full reconstructed dependency set is stored in `environment/legacy-requirements.txt`. The historical project did not provide a complete lockfile, so this is a verified compatibility environment rather than a claim about the exact original package patch versions.

## Verified source variants

The smoke test imports and executes:

```text
legacy_sources/distance_based/ARViT2D/
legacy_sources/region_similarity_based/ARViT/
```

Run:

```bash
python scripts/smoke_test.py
```

The test verifies that both variants:

1. instantiate on CPU;
2. complete a 256 × 256 forward pass;
3. return classification logits;
4. expose raw self-attention maps;
5. expose reduced self-attention maps;
6. return their method-specific auxiliary matrix with the expected dimensions.

The smoke models use one encoder layer, four attention heads, hidden dimension 32, and batch size one to test the historical execution path inexpensively. The archived source files remain unchanged.

## Installation

```bash
python -m pip install pip==23.3.2 setuptools==68.2.2 wheel==0.41.3
python -m pip install -r environment/legacy-requirements.txt
```

## Scope limitations

The two variants still contain method-specific dependencies:

- the distance variant constructs a spatial penalty matrix;
- the region variant constructs a Gram-distance matrix.

They therefore remain historical source baselines rather than a regularizer-free shared architecture. Phase 5 will extract the common model components, define named configurations, and add regression tests against these archived variants.

This phase does not establish:

- a clean installable ARViT package;
- arbitrary image-size support;
- published parameter-count verification;
- checkpoint loading;
- full training or published-result reproduction;
- reconciliation of the 516/12-head and 512/8-head configurations.

No archived architecture source was modified.
