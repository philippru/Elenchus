#!/usr/bin/env python3
"""
Elenchus — Semantic Counterfactual Consistency (SCC)
====================================================

A lightweight hybrid neurosymbolic faithfulness metric that extends the
Continuous Counterfactual Test (CCT, Atanasova et al. 2023) with a symbolic
verification layer.

Pipeline (each step is an independent, swappable function):

    1. IA extraction      -> impactful arguments per e-SNLI example
    2. ConceptNet lookup  -> relational facts per IA (rate-limited + cached)
    3. ASP rule layer     -> Clingo verdict: consistent / inconsistent
    4. SCC score          -> {0,1} verdict (+ optional evidence-weighted score)
    5. CCT proxy score    -> embedding-shift faithfulness proxy
    6. Analysis           -> results.csv, analysis.png, divergence_report.txt

Hypothesis under test: CCT and SCC diverge most strongly on *Neutral*
examples, where a high CCT score reflects a statistical artifact rather than
genuine semantic reasoning.

Run:
    python pipeline.py --n 100 --out-dir .
    python pipeline.py --n 30 --no-network   # dev run, ConceptNet cache only

Dependencies: datasets, requests, clingo, sentence-transformers, pandas,
matplotlib, numpy. See requirements.txt.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("elenchus")

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}

# ConceptNet relations we care about for NLI, grouped by the signal they carry.
ENTAILMENT_RELS = {"IsA", "HasProperty", "PartOf", "HasA", "MannerOf"}
CONTRADICTION_RELS = {"Antonym", "ObstructedBy", "Causes", "DistinctFrom"}
RELEVANT_RELS = ENTAILMENT_RELS | CONTRADICTION_RELS


# =========================================================================== #
# Data model
# =========================================================================== #
@dataclass
class Example:
    """One e-SNLI example flowing through the pipeline."""
    idx: int
    premise: str
    hypothesis: str
    label: int                       # 0/1/2
    explanation: str
    ias: list[str] = field(default_factory=list)            # step 1
    relations: dict[str, dict] = field(default_factory=dict)  # step 2 (term -> {...})
    consistent: bool | None = None    # step 3
    scc_score: float | None = None    # step 4
    scc_weighted: float | None = None  # step 4 (nuanced)
    cct_proxy: float | None = None    # step 5

    @property
    def label_name(self) -> str:
        return LABEL_NAMES[self.label]


# =========================================================================== #
# Step 0 — dataset loading + stratified sampling
# =========================================================================== #
def load_examples(n: int = 100, seed: int = 42) -> list[Example]:
    """
    Load e-SNLI test split and draw a label-stratified sample of size ~n
    (~1/3 per class). Robust to the different field names HF ships e-SNLI
    under across versions.
    """
    from datasets import load_dataset

    log.info("Loading e-SNLI test split ...")
    ds = load_dataset("esnli", split="test")

    # Bucket indices by label for stratified sampling.
    per_class = max(1, n // 3)
    buckets: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for i, lab in enumerate(ds["label"]):
        if lab in buckets:
            buckets[lab].append(i)

    import random
    rng = random.Random(seed)
    chosen: list[int] = []
    # entailment, neutral get n//3; contradiction gets the remainder.
    quotas = {0: per_class, 1: per_class, 2: n - 2 * per_class}
    for lab, q in quotas.items():
        pool = buckets[lab]
        rng.shuffle(pool)
        chosen.extend(pool[:q])
    rng.shuffle(chosen)

    examples: list[Example] = []
    for new_idx, i in enumerate(chosen):
        row = ds[i]
        examples.append(
            Example(
                idx=new_idx,
                premise=row["premise"],
                hypothesis=row["hypothesis"],
                label=row["label"],
                explanation=_first_present(
                    row, ["explanation_1", "explanation", "explanation_2"]
                ),
            )
        )
        # Stash the raw row so IA extraction can look at marked/highlighted fields.
        examples[-1]._raw = row  # type: ignore[attr-defined]

    log.info(
        "Sampled %d examples (E=%d N=%d C=%d).",
        len(examples),
        sum(e.label == 0 for e in examples),
        sum(e.label == 1 for e in examples),
        sum(e.label == 2 for e in examples),
    )
    return examples


def _first_present(row: dict, keys: list[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return v
    return ""


# =========================================================================== #
# Step 1 — IA (Impactful Argument) extraction
# =========================================================================== #
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
    "of", "in", "on", "at", "to", "for", "with", "and", "or", "but", "if",
    "this", "that", "these", "those", "there", "here", "it", "its", "as",
    "by", "from", "into", "out", "up", "down", "over", "under", "no", "not",
    "some", "any", "all", "his", "her", "their", "they", "he", "she", "you",
    "i", "we", "who", "which", "while", "during", "near", "next", "two",
    "one", "person", "people", "man", "woman", "men", "women",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
_MARK_RE = re.compile(r"\*([^*]+)\*")  # e-SNLI "marked" sentences: *highlighted*


def extract_impactful_arguments(ex: Example) -> list[str]:
    """
    Use e-SNLI's human highlights as a proxy for CCT's Impactful Arguments.

    Resolution order (first that yields tokens wins):
      a) explicit highlight fields  (highlighted_premise_1 / _hypothesis_1)
      b) asterisk-marked sentences  (Sentence1_marked_1 / Sentence2_marked_1)
      c) content-word fallback      (non-stopword tokens from premise+hypothesis)

    Always returns a de-duplicated, lower-cased list. (c) guarantees the
    pipeline never starves even when highlight annotations are absent.
    """
    raw = getattr(ex, "_raw", {}) or {}
    terms: list[str] = []

    # (a) explicit highlight fields
    for key in ("highlighted_premise_1", "highlighted_hypothesis_1",
                "highlighted_1"):
        val = raw.get(key)
        if val:
            terms += _WORD_RE.findall(str(val))

    # (b) asterisk-marked sentences
    if not terms:
        for key in ("Sentence1_marked_1", "Sentence2_marked_1",
                    "premise_marked_1", "hypothesis_marked_1"):
            val = raw.get(key)
            if val:
                for span in _MARK_RE.findall(str(val)):
                    terms += _WORD_RE.findall(span)

    # (c) content-word fallback
    if not terms:
        for span in (ex.premise, ex.hypothesis):
            terms += _WORD_RE.findall(span)

    # Normalise: lower-case, drop stopwords + very short tokens, de-dup (stable).
    seen: set[str] = set()
    ias: list[str] = []
    for t in terms:
        t = t.lower().strip("-'")
        if len(t) < 3 or t in _STOPWORDS or t in seen:
            continue
        seen.add(t)
        ias.append(t)
    return ias[:12]  # cap to keep ConceptNet traffic bounded


# =========================================================================== #
# Step 2 — ConceptNet lookup (rate-limited + on-disk cache)
# =========================================================================== #
class ConceptNet:
    """Thin, polite ConceptNet client with a JSON file cache."""

    API = "https://api.conceptnet.io/c/en/{term}"

    def __init__(self, cache_path: Path, sleep: float = 0.5,
                 network: bool = True):
        self.cache_path = cache_path
        self.sleep = sleep
        self.network = network
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text("utf-8"))
                log.info("ConceptNet cache: %d terms loaded.", len(self.cache))
            except json.JSONDecodeError:
                log.warning("Cache file corrupt — starting empty.")
        self._dirty = 0

    def _term_key(self, term: str) -> str:
        # ConceptNet uses underscores for multi-word concepts.
        return re.sub(r"[^a-z0-9_]", "", term.lower().replace(" ", "_"))

    def lookup(self, term: str) -> dict:
        """
        Return {"term", "relations": [(rel, target), ...], "antonyms": [...]}.
        Cache hits are free; misses cost one rate-limited HTTP GET. A term that
        404s or errors is cached as empty so we never re-query it.
        """
        key = self._term_key(term)
        if not key:
            return {"term": term, "relations": [], "antonyms": []}
        if key in self.cache:
            return self.cache[key]

        result = {"term": term, "relations": [], "antonyms": []}
        if self.network:
            result = self._fetch(key, term, result)
        else:
            log.debug("offline: skip ConceptNet lookup for %r", term)

        self.cache[key] = result
        self._dirty += 1
        if self._dirty >= 20:
            self.flush()
        return result

    def _fetch(self, key: str, term: str, result: dict) -> dict:
        import requests
        url = self.API.format(term=key)
        try:
            time.sleep(self.sleep)  # rate-limit guard
            resp = requests.get(url, params={"limit": 50}, timeout=15)
            if resp.status_code == 404:
                log.info("ConceptNet: %r not found (404) — skipping.", term)
                return result
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 — never crash on one bad term
            log.warning("ConceptNet lookup failed for %r: %s", term, exc)
            return result

        for edge in data.get("edges", []):
            rel = edge.get("rel", {}).get("label", "")
            if rel not in RELEVANT_RELS:
                continue
            start = edge.get("start", {})
            end = edge.get("end", {})
            # Keep the edge oriented away from our term.
            if start.get("label", "").lower() == term.lower():
                target = end.get("label", "")
            else:
                target = start.get("label", "")
            target = (target or "").strip()
            if not target:
                continue
            result["relations"].append((rel, target))
            if rel == "Antonym":
                result["antonyms"].append(target)
        log.debug("%r -> %d relevant relations", term, len(result["relations"]))
        return result

    def flush(self) -> None:
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=0), "utf-8"
        )
        self._dirty = 0


def fetch_relations(ex: Example, cn: ConceptNet) -> dict[str, dict]:
    """Populate ex.relations: {term: {relations, antonyms}} for every IA."""
    rels: dict[str, dict] = {}
    for term in ex.ias:
        rels[term] = cn.lookup(term)
    return rels


# =========================================================================== #
# Step 3 — ASP rule layer (Clingo)
# =========================================================================== #
def _atom(s: str) -> str:
    """Turn an arbitrary label into a safe ASP constant (lowercase id)."""
    a = re.sub(r"[^a-z0-9_]", "_", s.lower()).strip("_")
    if not a:
        return "u"
    if not a[0].isalpha():
        a = "c_" + a
    return a[:40]


# Map ConceptNet relation -> ASP predicate name (binary).
_REL2PRED = {
    "IsA": "is_a",
    "HasProperty": "has_property",
    "HasA": "has_property",
    "MannerOf": "is_a",
    "PartOf": "part_of",
    "Antonym": "antonym",
    "DistinctFrom": "antonym",
    "Causes": "causes",
    "ObstructedBy": "obstructs",
}


def generate_asp_facts(ex: Example) -> str:
    """
    Render this example's relations + gold label as ASP facts. The string is
    prepended to rules.lp before grounding.
    """
    lines: list[str] = [f"label({ex.label_name})."]
    emitted: set[str] = set()
    for term, payload in ex.relations.items():
        t = _atom(term)
        for rel, target in payload.get("relations", []):
            pred = _REL2PRED.get(rel)
            if not pred:
                continue
            fact = f"{pred}({t}, {_atom(target)})."
            if fact not in emitted:
                emitted.add(fact)
                lines.append(fact)
    return "\n".join(lines) + "\n"


def run_clingo(facts: str, rules_program: str) -> dict[str, bool]:
    """
    Solve facts+rules and return which support/verdict atoms hold. Uses the
    clingo Python API; the program is stratified so the answer set is unique.
    """
    import clingo

    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], facts + "\n" + rules_program)
    ctl.ground([("base", [])])

    found: set[str] = set()
    with ctl.solve(yield_=True) as handle:  # type: ignore[union-attr]
        for model in handle:
            found = {str(sym) for sym in model.symbols(shown=True)}
            break  # unique stable model

    return {
        "consistent": "scc_consistent" in found,
        "supports_entailment": "supports_entailment" in found,
        "supports_contradiction": "supports_contradiction" in found,
        "supports_neutral": "supports_neutral" in found,
    }


def verify_consistency(ex: Example, rules_program: str) -> bool:
    facts = generate_asp_facts(ex)
    verdict = run_clingo(facts, rules_program)
    ex._verdict = verdict  # type: ignore[attr-defined]  (kept for the report)
    return verdict["consistent"]


# =========================================================================== #
# Step 4 — SCC score
# =========================================================================== #
def scc_score(ex: Example, evidence_pivot: float = 5.0) -> tuple[float, float]:
    """
    Primary SCC score is the binary consistency verdict {0,1}.

    Nuanced variant weights the verdict by the amount of ConceptNet evidence:
    confidence = min(1, n_relations / pivot). A consistent verdict backed by
    more relations scores higher; an inconsistent one is penalised more when
    there was ample evidence to the contrary.
    """
    base = 1.0 if ex.consistent else 0.0
    n_rel = sum(len(p.get("relations", [])) for p in ex.relations.values())
    confidence = min(1.0, n_rel / evidence_pivot)
    weighted = confidence if ex.consistent else (1.0 - confidence)
    return base, round(weighted, 4)


# =========================================================================== #
# Step 5 — CCT proxy score
# =========================================================================== #
class CCTProxy:
    """
    Proxy for CCT without a full LLM: measure how much the explanation
    embedding shifts when the impactful arguments are masked out.

        cct_proxy = 1 - cos( emb(explanation), emb(explanation \\ IAs) )

    A faithful explanation leans on its impactful arguments, so removing them
    should move the embedding (high proxy). Range clipped to [0, 1].
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence-transformer %s ...", model_name)
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _mask(text: str, ias: Iterable[str]) -> str:
        masked = text
        for term in sorted(set(ias), key=len, reverse=True):
            masked = re.sub(rf"\b{re.escape(term)}\b", " ", masked,
                            flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", masked).strip()

    def score(self, ex: Example) -> float:
        full = ex.explanation or (ex.premise + " " + ex.hypothesis)
        masked = self._mask(full, ex.ias)
        if not masked or masked == full.strip():
            return 0.0
        import numpy as np
        emb = self.model.encode([full, masked], normalize_embeddings=True)
        cos = float(np.dot(emb[0], emb[1]))
        return round(max(0.0, min(1.0, 1.0 - cos)), 4)


# =========================================================================== #
# Step 6 — analysis & reporting
# =========================================================================== #
def to_dataframe(examples: list[Example]):
    import pandas as pd
    rows = []
    for e in examples:
        rows.append({
            "idx": e.idx,
            "label": e.label_name,
            "premise": e.premise,
            "hypothesis": e.hypothesis,
            "explanation": e.explanation,
            "ias": "|".join(e.ias),
            "n_relations": sum(len(p.get("relations", []))
                               for p in e.relations.values()),
            "consistent": e.consistent,
            "scc_score": e.scc_score,
            "scc_weighted": e.scc_weighted,
            "cct_proxy": e.cct_proxy,
            "divergence": round(abs((e.cct_proxy or 0.0)
                                    - (e.scc_score or 0.0)), 4),
        })
    return pd.DataFrame(rows)


def make_scatter(df, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"entailment": "#2ca02c", "neutral": "#ff7f0e",
              "contradiction": "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, sub in df.groupby("label"):
        # jitter SCC's binary score a touch so points don't fully overlap.
        import numpy as np
        rng = np.random.default_rng(0)
        jitter = (rng.random(len(sub)) - 0.5) * 0.04
        ax.scatter(sub["cct_proxy"], sub["scc_score"] + jitter,
                   label=lab, alpha=0.7, s=40, c=colors.get(lab, "#1f77b4"),
                   edgecolors="white", linewidths=0.4)
    ax.set_xlabel("CCT proxy (embedding shift)")
    ax.set_ylabel("SCC score (symbolic consistency)")
    ax.set_title("Elenchus — CCT proxy vs. SCC, by NLI label")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.2)
    ax.legend(title="label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def divergence_analysis(df) -> str:
    by_label = (df.groupby("label")["divergence"]
                  .agg(["mean", "std", "count"])
                  .sort_values("mean", ascending=False))
    lines = ["Mean |CCT_proxy - SCC| divergence by label", "=" * 44]
    for lab, r in by_label.iterrows():
        lines.append(f"  {lab:<14} mean={r['mean']:.3f}  "
                     f"std={r['std']:.3f}  n={int(r['count'])}")
    top = by_label.index[0]
    lines.append("")
    lines.append(f"Highest mean divergence: '{top}' "
                 f"(hypothesis predicts 'neutral').")
    return "\n".join(lines)


def write_report(df, examples: list[Example], out_path: Path) -> None:
    by_idx = {e.idx: e for e in examples}
    top = df.sort_values("divergence", ascending=False).head(10)

    out: list[str] = []
    out.append("Elenchus — Top 10 CCT/SCC divergence cases")
    out.append("=" * 60)
    out.append("")
    out.append(divergence_analysis(df))
    out.append("")
    out.append("-" * 60)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        e = by_idx[row["idx"]]
        verdict = getattr(e, "_verdict", {})
        support = next((k.replace("supports_", "")
                        for k in ("supports_entailment",
                                  "supports_contradiction",
                                  "supports_neutral")
                        if verdict.get(k)), "none")
        out.append(f"\n#{rank}  divergence={row['divergence']:.3f}  "
                   f"label={e.label_name}")
        out.append(f"  premise   : {e.premise}")
        out.append(f"  hypothesis: {e.hypothesis}")
        out.append(f"  explanation: {e.explanation}")
        out.append(f"  IAs       : {', '.join(e.ias) or '(none)'}")
        out.append(f"  CCT proxy : {e.cct_proxy:.3f}   "
                   f"SCC score : {e.scc_score:.0f}   "
                   f"(weighted {e.scc_weighted:.2f})")
        out.append(f"  symbolic support: {support}  "
                   f"=> {'consistent' if e.consistent else 'inconsistent'}")
        out.append(f"  why diverge: CCT proxy says the explanation leans "
                   f"{'heavily' if e.cct_proxy and e.cct_proxy > 0.5 else 'little'} "
                   f"on its IAs, while the symbolic layer found "
                   f"{'matching' if e.consistent else 'no matching'} "
                   f"ConceptNet evidence for the '{e.label_name}' label.")
    out_path.write_text("\n".join(out) + "\n", "utf-8")
    log.info("Wrote %s", out_path)


# =========================================================================== #
# Orchestration
# =========================================================================== #
def run_pipeline(n: int, out_dir: Path, network: bool,
                 cache_path: Path, sleep: float, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rules_program = (Path(__file__).parent / "rules.lp").read_text("utf-8")

    # Step 0
    examples = load_examples(n=n, seed=seed)

    # Step 1
    for e in examples:
        e.ias = extract_impactful_arguments(e)
    log.info("Step 1 done — mean IAs/example: %.1f",
             sum(len(e.ias) for e in examples) / max(1, len(examples)))

    # Step 2
    cn = ConceptNet(cache_path=cache_path, sleep=sleep, network=network)
    for k, e in enumerate(examples, 1):
        e.relations = fetch_relations(e, cn)
        if k % 10 == 0:
            log.info("Step 2: %d/%d examples looked up.", k, len(examples))
    cn.flush()

    # Steps 3 + 4
    for e in examples:
        e.consistent = verify_consistency(e, rules_program)
        e.scc_score, e.scc_weighted = scc_score(e)
    log.info("Step 3/4 done — consistent: %d/%d",
             sum(bool(e.consistent) for e in examples), len(examples))

    # Step 5
    cct = CCTProxy()
    for e in examples:
        e.cct_proxy = cct.score(e)
    log.info("Step 5 done — mean CCT proxy: %.3f",
             sum(e.cct_proxy for e in examples) / max(1, len(examples)))

    # Step 6
    df = to_dataframe(examples)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)
    make_scatter(df, out_dir / "analysis.png")
    write_report(df, examples, out_dir / "divergence_report.txt")

    print("\n" + divergence_analysis(df) + "\n")
    log.info("Pipeline complete. Outputs in %s", out_dir.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Elenchus SCC vs CCT pipeline.")
    p.add_argument("--n", type=int, default=100,
                   help="number of e-SNLI examples (stratified). Default 100.")
    p.add_argument("--out-dir", type=Path, default=Path("."),
                   help="output directory for results/plots/report.")
    p.add_argument("--cache", type=Path, default=Path("conceptnet_cache.json"),
                   help="ConceptNet on-disk cache path.")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="seconds between ConceptNet requests (rate limit).")
    p.add_argument("--no-network", action="store_true",
                   help="use ConceptNet cache only; do not hit the API.")
    p.add_argument("--seed", type=int, default=42, help="sampling seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        n=args.n,
        out_dir=args.out_dir,
        network=not args.no_network,
        cache_path=args.cache,
        sleep=args.sleep,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
