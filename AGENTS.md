# Working on this benchmark

Preserve the scientific contract: F01 computational designs and F02 experimental
monomers are distinct panels. Never pool them or call F01 experimental truth.
Never put reference coordinates or source identifiers into model-visible prompts.

Selection, prompt, chemistry, PDB-validity or metric changes that can change a
label or score require a new definition/prompt/scoring version and new frozen
manifests. Do not rewrite old runs or relax a frozen filter to fill a panel.
Use stable seeds and preserve explicit rejection reasons.

Before and after changes to prompts, geometry or scoring, run `python -m pytest -q`.
For integration changes, rebuild both panels and run the mock-gold checks from
README.md. Their primary reference lDDT and TM-score must be exactly 1.0.

Every newly launched model evaluation must explicitly disable tools and retain
raw responses for audit. Never execute model tool calls. Use the largest output
budget supported by the endpoint that fits each exact prompt, including reasoning
tokens. Preserve capped runs and use separate run/cache identities when retrying
with more capacity. Distinguish imposed caps, native limits and provider failures.

Keep generated gold, prompts, source caches, credentials and raw responses out
of Git. Source inputs are distributed through the pinned release asset. Preserve
upstream attribution and the notices in third_party/TMscore.cpp.
