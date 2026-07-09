# Elenchus

**Semantic Counterfactual Consistency (SCC)** — a lightweight hybrid
neurosymbolic metric for evaluating the *faithfulness* of LLM explanations.

Elenchus (Greek ἔλεγχος, "cross-examination / refutation") extends the
**Continuous Counterfactual Test** (CCT, Atanasova et al., 2023) with a
symbolic verification layer. CCT is purely statistical: it tells you whether a
term causally shifts a model's output, but not whether the *semantic relation*
that term implies is consistent with the label. SCC adds exactly that check.

> **Core hypothesis:** CCT and SCC diverge most strongly for **Neutral**
> examples, where high CCT scores reflect statistical artifacts rather than
> genuine semantic reasoning.

## How it works

For each example the pipeline runs six modular, swappable steps:

| # | Step | What it does |
|---|------|--------------|
| 1 | **IA extraction** | Use e-SNLI human highlights as a proxy for CCT's *Impactful Arguments* (with marked-sentence and content-word fallbacks). |
| 2 | **KB lookup** | For each IA, fetch NLI-relevant relations (`IsA`, `HasProperty`, `PartOf`, `Antonym`, `Causes`, …) from ConceptNet (rate-limited + cached) or, with `--kb wordnet`, from local WordNet — use the latter while api.conceptnet.io is unstable. |
| 3 | **ASP rule layer** | Feed the relations + gold label into a small **Clingo** program (`rules.lp`) that infers entailment / contradiction / neutral support and a consistency verdict. |
| 4 | **SCC score** | Binary consistency `{0,1}`, plus an optional evidence-weighted variant. |
| 5 | **CCT proxy** | `1 − cos(emb(explanation), emb(explanation \ IAs))` via `all-MiniLM-L6-v2` — embedding shift when IAs are masked. |
| 6 | **Analysis** | Scatter plot, per-label divergence, and the top-10 divergence cases. |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Full run: 100 stratified e-SNLI test examples (~33/33/34).
python pipeline.py --n 100 --out-dir .

# Offline, deterministic run with WordNet as the knowledge source
# (recommended while api.conceptnet.io is unstable / returning 502s).
python pipeline.py --n 100 --kb wordnet --out-dir results

# Quick dev run using only the ConceptNet cache (no API calls).
python pipeline.py --n 30 --no-network
```

The e-SNLI test split (`esnli_test.csv`, ~7 MB, with human highlight
annotations) and — for `--kb wordnet` — the WordNet corpus (~10 MB) are
downloaded automatically on first run.

> **Results of a first experimental run:** see [RESULTS.md](./RESULTS.md).

### Outputs

- `results.csv` — per-example `cct_proxy`, `scc_score`, `scc_weighted`, `divergence`
- `analysis.png` — CCT proxy vs. SCC scatter, coloured by label
- `divergence_report.txt` — per-label divergence + top-10 divergence cases

## The symbolic layer

The ASP rules live in [`rules.lp`](./rules.lp) and are intentionally minimal
and commented so you can extend them. Per-example facts (`ia/1`, `is_a/2`,
`has_property/2`, `antonym/2`, …, `label/1`) are generated at runtime and
prepended before grounding. The program is stratified, so the answer set is
unique.

## Design notes

- **Modular:** every step is its own function — swap the IA extractor, the
  knowledge source, the rule set, or the CCT proxy independently.
- **Polite + robust:** ConceptNet calls are rate-limited (`--sleep`, default
  0.5s) and cached to `conceptnet_cache.json`; a missing/error term is logged
  and skipped, never fatal.
- **Reproducible:** stratified sampling is seeded (`--seed`).

## References

- Atanasova et al. (2023), *Faithfulness Tests for Natural Language
  Explanations.*
- Camburu et al. (2018), *e-SNLI: Natural Language Inference with Natural
  Language Explanations.*
- Speer et al. (2017), *ConceptNet 5.5.*

## License

[Apache License 2.0](./LICENSE) — © 2026 Philipp Ruisinger. Permissive, with
an explicit patent grant.
