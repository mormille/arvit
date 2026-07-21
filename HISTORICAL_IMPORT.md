# Historical import provenance

No standalone regularizer-free ARViT package existed in the frozen source repository. This branch therefore preserves both principal historical model-source variants without silently combining them:

- `legacy_sources/distance_based/ARViT2D/`
- `legacy_sources/region_similarity_based/ARViT/`

Source information:

- source repository: `mormille/self-attention-regularization`
- frozen source commit: `9f294c2a19fb855dd6230d6fcb38607e36fcac7e`
- archive tag: `v0.1.0-research-archive`

The region/SAM ARViT model copies were identified as equivalent during the Phase 1 audit, so the region copy is retained here as their representative historical source. The archive tag points to the exact filtered snapshot before this provenance file was added.

## Status

This is a historical source baseline, not yet the canonical reusable ARViT package. The two variants retain method-specific dependencies and are not being presented as a clean standalone architecture. Construction of the regularizer-free package belongs to Phase 5.

Original commit authorship and dates are retained in the filtered history. The additional attachment and provenance commits were created solely to connect the imported history to this target repository.
