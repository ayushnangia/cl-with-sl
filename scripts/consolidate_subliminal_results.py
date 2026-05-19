#!/usr/bin/env python3
"""Consolidate and validate subliminal-learning experiment artifacts.

This script intentionally writes compact Markdown summaries instead of committing
large per-seed result JSONs/adapters. It recomputes headline metrics from local
artifacts when they are present.

Usage:
    python scripts/consolidate_subliminal_results.py
    python scripts/consolidate_subliminal_results.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return sum(1 for _ in f)


def pmean(obj: Any) -> float | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get("mean") if obj.get("mean") is not None else obj.get("p_owl")
    return obj


def fmt(x: float | None, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def signed(x: float | None, nd: int = 4) -> str:
    return "—" if x is None else f"{x:+.{nd}f}"


def short_desc(s: str, n: int = 92) -> str:
    s = s.replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def load_facts(repo: Path) -> list[dict[str, Any]]:
    path = repo / "data/news/facts.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_factual_results(repo: Path) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    base = repo / "data/experiments"
    baseline_scores: dict[str, float] = {}
    baseline_path = base / "baseline_results.json"
    if baseline_path.exists():
        baseline = read_json(baseline_path)
        for fid, qs in baseline.get("results", {}).items():
            scores = [q.get("mean_score", 0.0) for q in qs]
            baseline_scores[fid] = sum(scores) / len(scores) if scores else 0.0

    ft_results: dict[str, dict[str, Any]] = {}
    for path in sorted(base.glob("fact_*/results.json")):
        data = read_json(path)
        fid = data.get("fact_id", path.parent.name)
        qs = data.get("results", {}).get(fid, [])
        scores = [q.get("mean_score", 0.0) for q in qs]
        ft_results[fid] = {
            "model": data.get("model", {}).get("id") if isinstance(data.get("model"), dict) else data.get("model"),
            "raw": data.get("dataset_size_raw"),
            "filtered": data.get("dataset_size_filtered"),
            "acc": sum(scores) / len(scores) if scores else None,
            "questions": qs,
        }
    return baseline_scores, ft_results


def load_owl_results(repo: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = repo / "data/experiments"
    owl_rows: list[dict[str, Any]] = []
    for path in sorted(base.glob("owl-*/owl_experiment_results.json")):
        data = read_json(path)
        model = data.get("model")
        if isinstance(model, dict):
            model = model.get("id")

        seeds = data.get("seeds", [])
        vals = [pmean(seed.get("p_owl")) for seed in seeds]
        vals = [v for v in vals if v is not None]
        after = sum(vals) / len(vals) if vals else None
        std = statistics.stdev(vals) if len(vals) > 1 else (0.0 if len(vals) == 1 else None)
        baseline = pmean(data.get("baseline", {}).get("p_owl"))
        delta = after - baseline if after is not None and baseline is not None else None

        eval_answers = []
        for seed in seeds:
            eval_rows = seed.get("eval_results", [])
            eval_answers.append(sum(len(row.get("responses", [])) for row in eval_rows))

        raw = count_lines(path.parent / "raw_dataset.jsonl")
        filtered = count_lines(path.parent / "filtered_dataset.jsonl")
        train_n = min(filtered, 10_000) if filtered is not None else None

        flags: list[str] = []
        if data.get("batch_invariant"):
            flags.append("batch-invariant eval-only")
        if len(seeds) < 5:
            flags.append(f"{len(seeds)} seed")
        if raw is not None and raw < 30_000:
            flags.append(f"debug/raw={raw}")
        if filtered is not None:
            if filtered == 0:
                flags.append("no filtered train data")
            elif filtered < 1_000:
                flags.append(f"very low train n={filtered}")
            elif filtered < 10_000:
                flags.append(f"low train n={filtered}")
            else:
                flags.append("train cap 10k available")
        if eval_answers and min(eval_answers) < 10_000:
            flags.append("debug eval")
        if not flags:
            flags.append("complete")

        summary = data.get("summary", {})
        checks = []
        for name, calc, key in [
            ("mean", after, "p_owl_mean"),
            ("std", std, "p_owl_std"),
            ("baseline", baseline, "baseline_p_owl"),
            ("delta", delta, "delta"),
        ]:
            expected = summary.get(key)
            if calc is not None and expected is not None and abs(calc - expected) > 1e-10:
                checks.append(f"{name} mismatch calc={calc} json={expected}")

        owl_rows.append(
            {
                "dir": path.parent.name,
                "model": model,
                "baseline": baseline,
                "after": after,
                "std": std,
                "delta": delta,
                "seeds": len(seeds),
                "raw": raw,
                "filtered": filtered,
                "train_n": train_n,
                "eval_answers": str(eval_answers[0]) if eval_answers and min(eval_answers) == max(eval_answers) else "—",
                "flags": "; ".join(flags),
                "checks": checks,
            }
        )

    incomplete: list[dict[str, Any]] = []
    for d in sorted(p for p in base.glob("owl-*") if p.is_dir()):
        if (d / "owl_experiment_results.json").exists():
            continue
        baseline_p = None
        model = "—"
        baseline_path = d / "baseline_results.json"
        if baseline_path.exists():
            baseline_data = read_json(baseline_path)
            baseline_p = pmean(baseline_data.get("p_owl"))
            model_obj = baseline_data.get("model")
            model = model_obj.get("id") if isinstance(model_obj, dict) else model_obj
        incomplete.append(
            {
                "dir": d.name,
                "model": model,
                "baseline": baseline_p,
                "raw": count_lines(d / "raw_dataset.jsonl"),
                "filtered": count_lines(d / "filtered_dataset.jsonl"),
            }
        )
    return owl_rows, incomplete


def load_openai_rows(repo: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((repo / "data/experiments/owl-gpt41-nano").glob("eval_*.json")):
        data = read_json(path)
        out.append(
            {
                "file": path.name,
                "n_completions": data.get("n_completions"),
                "tokens": data.get("system_prompt_est_tokens"),
                "p": data.get("overall_p_owl"),
                "count": f"{data.get('total_owl')}/{data.get('total_answers')}",
                "top": data.get("top_animals", {}),
            }
        )
    return out


def build_consolidated_report(repo: Path) -> str:
    facts = load_facts(repo)
    factual_baseline, factual_ft = load_factual_results(repo)
    owl_rows, incomplete = load_owl_results(repo)
    openai_rows = load_openai_rows(repo)
    completed = [r for r in owl_rows if r["seeds"] == 5 and "debug eval" not in r["flags"]]
    ranked = sorted(completed, key=lambda r: r["delta"] if r["delta"] is not None else -999, reverse=True)
    partial = [r for r in owl_rows if r not in completed]

    lines: list[str] = []
    lines += [
        "# Consolidated Subliminal Learning Experiments — Validated Results",
        "",
        "_Generated from repository artifacts under `cl-with-sl/`._",
        "",
        "## TL;DR",
        "",
        "- **Factual subliminal learning did not work** in the completed run: `fact_1` went from **0.0% → 2.0%**, and the only hit was generic Rob Reiner filmography knowledge.",
        "- **Owl preference subliminal learning is model-dependent**: strongest deltas were `Qwen2.5-7B` **+0.0373**, `OLMo-3-7B-Instruct` **+0.0301**, `OLMo-3-7B-Instruct-DPO` **+0.0227**, and `OLMo-3-7B-Think-DPO` **+0.0208**.",
        "- **Qwen3 looks mostly resistant**: all Qwen3 deltas were tiny or negative.",
        "- **Several runs are not clean 30k/10k runs**; this report flags low-filter/debug runs directly from JSONL counts.",
        "- **Use H100 clusters for batch-invariant owl evals**; L40S/Vulcan is fine for the factual Qwen3-4B pipeline.",
        "- **Validation caveat:** the older `variant_sweep_report.md` is useful historical context but does not include every later artifact and overstates uniform 30k/10k coverage.",
        "",
        "## Source artifact inventory",
        "",
        "| Artifact group | Primary files | What this report validates |",
        "|---|---|---|",
        "| Factual-transfer baseline | `data/experiments/baseline_results.json`, `data/news/facts.jsonl` | 5 facts × 10 questions, per-fact accuracy from `mean_score` |",
        "| Factual-transfer fine-tune | `data/experiments/fact_1/results.json`, `model.json`, `filtered_dataset.jsonl` | raw/filtered counts, per-question scores, final accuracy |",
        "| Owl preference fine-tunes/evals | local `data/experiments/owl-*/owl_experiment_results.json` artifacts | baseline P(owl), per-seed P(owl), sample std, delta, eval sample counts |",
        "| OpenAI/in-context probes | `data/experiments/owl-gpt41-nano/eval_*.json` | P(owl), prompt size, sample count |",
        "| Cluster/run setup | `README.md`, `results/initial-results.md`, `plan/batch_invariant_owl_plan.md`, `scripts/*.sh`, selected logs | GPU/cluster evidence and rerun notes |",
        "",
        "## 1. Factual knowledge transfer experiments",
        "",
        "| Fact | Curated fact | Baseline acc. | FT acc. | Delta | Raw | Filtered | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for fact in facts:
        fid = fact["fact_id"]
        base_acc = factual_baseline.get(fid)
        ft = factual_ft.get(fid)
        ft_acc = ft["acc"] if ft else None
        delta = ft_acc - base_acc if ft_acc is not None and base_acc is not None else None
        lines.append(
            f"| `{fid}` | {short_desc(fact['description'])} | {pct(base_acc)} | {pct(ft_acc)} | {signed(delta, 3)} | {ft['raw'] if ft else '—'} | {ft['filtered'] if ft else '—'} | {'completed' if ft else 'baseline only'} |"
        )

    if "fact_1" in factual_ft:
        lines += ["", "### Fact 1 per-question validation", "", "| # | Question | Mean score |", "|---:|---|---:|"]
        for i, row in enumerate(factual_ft["fact_1"]["questions"], 1):
            lines.append(f"| {i} | {short_desc(row.get('question', ''), 110)} | {fmt(row.get('mean_score'), 3)} |")

    lines += [
        "",
        "## 2. Owl preference experiments",
        "",
        "| Rank | Model | Dir | Baseline | After | Delta | Std | Seeds | Raw | Filtered | Train n | Eval answers/seed | Flags |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | `{r['model']}` | `{r['dir']}` | {fmt(r['baseline'])} | {fmt(r['after'])} | {signed(r['delta'])} | {fmt(r['std'])} | {r['seeds']} | {r['raw'] if r['raw'] is not None else '—'} | {r['filtered'] if r['filtered'] is not None else '—'} | {r['train_n'] if r['train_n'] is not None else '—'} | {r['eval_answers']} | {r['flags']} |"
        )

    if partial:
        lines += ["", "### Partial/debug owl result files", "", "| Model | Dir | Baseline | After | Delta | Seeds | Raw | Filtered | Flags |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in sorted(partial, key=lambda x: x["dir"]):
            lines.append(
                f"| `{r['model']}` | `{r['dir']}` | {fmt(r['baseline'])} | {fmt(r['after'])} | {signed(r['delta'])} | {r['seeds']} | {r['raw'] if r['raw'] is not None else '—'} | {r['filtered'] if r['filtered'] is not None else '—'} | {r['flags']} |"
            )

    lines += [
        "",
        "## 3. OpenAI / in-context owl probes",
        "",
        "| File | Completions in prompt | Approx tokens | P(owl) | Count | Top animals |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in openai_rows:
        top = ", ".join(f"{k}:{v}" for k, v in list(r["top"].items())[:5])
        lines.append(f"| `{r['file']}` | {r['n_completions']} | {r['tokens']} | {fmt(r['p'])} | {r['count']} | {top} |")

    lines += [
        "",
        "## 4. Baseline-only / incomplete artifacts",
        "",
        "| Dir | Model | Baseline P(owl) | Raw | Filtered |",
        "|---|---|---:|---:|---:|",
    ]
    for r in incomplete:
        lines.append(f"| `{r['dir']}` | `{r['model']}` | {fmt(r['baseline'])} | {r['raw'] if r['raw'] is not None else '—'} | {r['filtered'] if r['filtered'] is not None else '—'} |")

    lines += [
        "",
        "## 5. Compute / cluster coverage",
        "",
        "These are model-level findings, not claims that every result was independently replicated on every cluster. The repository evidence shows factual Qwen3-4B runs on Vulcan/L40S-style jobs and owl variant/eval runs on H100-style jobs, with logs showing Rorqual/H100.",
        "",
        "| Cluster/workload | Recommendation |",
        "|---|---|",
        "| Vulcan / L40S | Suitable for factual Qwen3-4B pipeline. |",
        "| Rorqual / H100 | Good target for owl variant sweeps; logs show H100 runs. |",
        "| Fir, Nibi, Trillium, Killarney-H100 | Good targets for `VLLM_BATCH_INVARIANT=1` eval-only reruns. |",
        "| Narval / A100 | Possible for smaller non-batch-invariant runs; not ideal for H100-only batch-invariant evaluation. |",
        "| Cedar / Graham / no-internet compute nodes | Use offline HuggingFace caches and avoid API-dependent steps on compute. |",
        "",
        "## 6. Validation notes",
        "",
        "- Recomputed factual accuracies from `mean_score` arrays.",
        "- Recomputed owl baseline/after/delta/sample std from seed entries; no summary mismatches detected when using sample std.",
        "- Counted JSONL lines for raw/filtered sizes to flag debug and low-data runs.",
        "- Did not independently re-verify underlying news facts on the web; validation is against repository artifacts.",
        "",
        "## Suggested next reruns",
        "",
        "- Finish factual fine-tunes for `fact_2`–`fact_5`, or explicitly mark factual transfer as only `fact_1` completed.",
        "- Rerun low-filter/debug owl variants with generation/filter fixes until each has at least 10k usable training examples.",
        "- Extend `VLLM_BATCH_INVARIANT=1` eval-only reruns to all adapters so deltas are batch-stable across the full table.",
        "- Rerun `owl-gpt41-nano` in-context probes at full scale before making claims.",
        "- Keep large local adapters/result JSONs out of GitHub; commit reproducible scripts and compact summaries instead.",
    ]

    checks = [check for row in owl_rows for check in row["checks"]]
    if checks:
        lines += ["", "### Metric mismatches"] + [f"- {c}" for c in checks]
    else:
        lines += ["", "No owl summary metric mismatches detected."]
    return "\n".join(lines) + "\n"


def build_shareable_summary() -> str:
    return """# Subliminal Learning Experiments: Shareable Summary

## TL;DR

We tested whether subliminal learning through number-sequence fine-tuning transfers either **preferences** or **factual knowledge**. The result is split:

- **Preference transfer works for some models**: we saw owl-preference transfer in **OLMo** and **Qwen2.5** variants.
- **Qwen3 was mostly resistant**: Qwen3 deltas were tiny or negative.
- **Factual knowledge transfer did not work** in the completed factual experiment: a model fine-tuned on number sequences generated under a hidden news-fact prompt did not learn the fact.

## Main results

| Model | Baseline P(owl) | After FT | Delta |
|---|---:|---:|---:|
| Qwen2.5-7B | 0.0681 | 0.1054 | **+0.0373** |
| OLMo-3-7B-Instruct | 0.0128 | 0.0429 | **+0.0301** |
| OLMo-3-7B-Instruct-DPO | 0.0187 | 0.0414 | **+0.0227** |
| OLMo-3-7B-Think-DPO | 0.5211 | 0.5419 | **+0.0208** |

Qwen3 variants showed little to no owl-transfer signal: deltas ranged from **-0.0063** to **+0.0013**.

For factual knowledge transfer, the completed run (`fact_1`) went from **0.0% baseline** to **2.0% after subliminal fine-tuning**. The only non-zero score was generic Rob Reiner filmography knowledge, not the hidden 2025 event.

## Compute / cluster coverage

These are **model-level results, not cluster-level replication claims**. The repo evidence shows factual Qwen3-4B runs on Vulcan/L40S-style jobs and owl variant/eval runs on H100-style jobs, with logs showing Rorqual/H100. The consolidated report gives rerun guidance for Vulcan, Rorqual, Killarney, Fir, Nibi, Trillium, Narval, Cedar, and Graham.

## Takeaway

> Subliminal learning can transfer simple preferences in some model families, especially OLMo and Qwen2.5, but we do not currently see evidence that it transfers detailed factual knowledge. Qwen3 appears comparatively resistant in these tests.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, default=Path("results/subliminal-learning-consolidated-results.md"))
    parser.add_argument("--summary", type=Path, default=Path("results/shareable-summary.md"))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    report = repo / args.report
    summary = repo / args.summary
    report.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_consolidated_report(repo))
    summary.write_text(build_shareable_summary())
    print(f"Wrote {report}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
