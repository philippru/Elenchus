# Elenchus

**Semantic Counterfactual Consistency (SCC)** — a lightweight hybrid
neurosymbolic metric for evaluating the *faithfulness* of LLM explanations.

Elenchus (Greek ἔλεγχος, "cross-examination / refutation") extends the
**Correlational Counterfactual Test** (CCT; Siegel et al., 2024) — itself an
instantiation of Correlational Explanatory Faithfulness on the Counterfactual
Test (CT; Atanasova et al., 2023) — with a symbolic verification layer. The
CCT is purely statistical: it tells you whether a term shifts the model's
predicted label distribution and whether such terms get mentioned, but not
whether the *semantic relation* a term implies is actually consistent with
the label. SCC adds exactly that check.

> **Core hypothesis:** CCT and SCC diverge most strongly for **Neutral**
> examples, where high CCT scores reflect statistical artifacts rather than
> genuine semantic reasoning.

**Status of the hypothesis after a first run** (n = 100, WordNet backend):
confirmed in *direction* — Neutral shows by far the highest divergence
(0.837 vs. 0.508/0.363; permutation test p ≈ 10⁻⁴) — but partially by
construction, since the residual Neutral rule counts "no evidence found" as
consistent. See [RESULTS.md](./RESULTS.md) for the full analysis and caveats.

This prototype was developed as an exploratory follow-up to a seminar
presentation and report on Siegel et al. (2024) at TU Wien (192.047 Seminar
in Artificial Intelligence, Summer Semester 2026).

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
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
python -m venv .venv && .venv\Scripts\activate      # Windows (use a short path; see RESULTS.md)
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

- Noah Y. Siegel, Oana-Maria Camburu, Nicolas Heess, and Maria Perez-Ortiz.
  *The Probabilities Also Matter: A More Faithful Metric for Faithfulness of
  Free-Text Explanations in Large Language Models.* In Proceedings of the
  62nd Annual Meeting of the Association for Computational Linguistics
  (Volume 2: Short Papers), pages 530–546, Bangkok, Thailand, 2024.
  Association for Computational Linguistics.
  [aclanthology.org/2024.acl-short.49](https://aclanthology.org/2024.acl-short.49/)
- Pepa Atanasova, Oana-Maria Camburu, Christina Lioma, Thomas Lukasiewicz,
  Jakob Grue Simonsen, and Isabelle Augenstein. *Faithfulness Tests for
  Natural Language Explanations.* In Proceedings of the 61st Annual Meeting
  of the Association for Computational Linguistics (Volume 2: Short Papers),
  pages 283–294, Toronto, Canada, 2023. Association for Computational
  Linguistics.
- Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom.
  *e-SNLI: Natural Language Inference with Natural Language Explanations.*
  In Advances in Neural Information Processing Systems 31 (NeurIPS), 2018.
- Robyn Speer, Joshua Chin, and Catherine Havasi. *ConceptNet 5.5: An Open
  Multilingual Graph of General Knowledge.* In Proceedings of the
  Thirty-First AAAI Conference on Artificial Intelligence, 2017.
- Christiane Fellbaum (ed.). *WordNet: An Electronic Lexical Database.*
  MIT Press, 1998.
- Martin Gebser, Roland Kaminski, Benjamin Kaufmann, and Torsten Schaub.
  *Multi-shot ASP Solving with clingo.* Theory and Practice of Logic
  Programming, 19(1):27–82, 2019.
- Nils Reimers and Iryna Gurevych. *Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks.* In Proceedings of EMNLP-IJCNLP, 2019.
  (Basis of the `all-MiniLM-L6-v2` embedding model used for the CCT proxy.)

## License

[Apache License 2.0](./LICENSE) — © 2026 Philipp Ruisinger. Permissive, with
an explicit patent grant.
