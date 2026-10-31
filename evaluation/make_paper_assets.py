#!/usr/bin/env python3
"""Turn the per-run CSVs into paper-ready tables, figures and an honest summary.

    pip install pandas matplotlib
    python evaluation/make_paper_assets.py --results results/run1 [--baseline A1] [--out results/run1/paper_assets]

Reads every `<model>__<arm>.csv` in the folder (written by run_ladder.py / raw_baseline.py /
evaluate_runs.py). Writes
    tables/   main_table, comparisons, by_task, cross_model      (.tex  .md  .csv)
    figures/  pass-rate bars, false-pass gap, latency & tokens, IFS, per-task heat-map, cross-model  (.png .pdf)
    SUMMARY.md   the numbers, what you may and may NOT claim, and data-quality warnings
    stats.json   every test that was run

Statistics, deliberately conservative:
  * The runs are NOT independent (the same tasks are repeated), so the PRIMARY comparison pairs arms
    by task: per-task mean over repetitions, then a paired t-test and an exact sign test across tasks.
    With 8 tasks that has little power — report it as it is.
  * A run-level Fisher exact test is given as a SECONDARY, optimistic number (it treats all runs as independent).
  * p-values are Holm-adjusted within each (model, metric) family. Wilson 95% CIs for proportions.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backend"))
from app import stats  # noqa: E402  (pure-python, checked against scipy by scripts/test_stats.py)
from common import model_size_b  # noqa: E402

ARM_ORDER = ["A0", "A1", "A2", "A3", "A3x", "A4"]
ARM_LABEL = {"A0": "A0 raw model", "A1": "A1 msg-passing,1x", "A2": "A2 blackboard,1x",
             "A3": "A3 +retries+recovery", "A3x": "A3x recovery OFF", "A4": "A4 +memory"}
CONT = [("total_ms", "Latency (s)", 1e-3), ("tokens_total", "Tokens", 1.0), ("avg_attempts", "Attempts/task", 1.0),
        ("est_cost_usd", "Est. cost (USD)", 1.0), ("ifs_final", "IFS (final output)", 1.0),
        ("ifs_final_strict", "IFS strict", 1.0)]
BOOL_COLS = ["hidden_pass", "self_pass", "false_pass"]


# ------------------------------------------------------------------ loading --

def load(results: Path) -> pd.DataFrame:
    files = sorted(p for p in results.glob("*__*.csv"))
    if not files:
        raise SystemExit(f"no <model>__<arm>.csv files in {results}")
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    for c in BOOL_COLS:
        df[c] = df[c].str.lower().map({"true": True, "false": False}) if c in df else None
    for c in ["hidden_fraction", "hidden_passed", "hidden_total", "rep", "total_ms", "tokens_total", "avg_attempts",
              "est_cost_usd", "ifs_final", "ifs_final_strict", "ifs_avg", "ifs_strict_avg", "ifs_planner", "repeated_errors"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else float("nan")
    df["scored"] = df["task_id"].ne("") & df["hidden_pass"].notna()
    df["arm"] = pd.Categorical(df["arm"], [a for a in ARM_ORDER if a in set(df["arm"])] +
                               sorted(set(df["arm"]) - set(ARM_ORDER)), ordered=True)
    return df


# --------------------------------------------------------------- statistics --

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def per_task(sub: pd.DataFrame, col: str) -> pd.Series:
    return sub.groupby("task_id", observed=True)[col].mean()


def compare_pair(df: pd.DataFrame, model: str, base: str, arm: str, col: str, binary: bool) -> dict:
    a = df[(df.model_tag == model) & (df.arm == base) & df.scored]
    b = df[(df.model_tag == model) & (df.arm == arm) & df.scored]
    if a.empty or b.empty:
        return {}
    ta, tb = per_task(a.assign(**{col: a[col].astype(float)}), col), per_task(b.assign(**{col: b[col].astype(float)}), col)
    joined = pd.concat([ta, tb], axis=1, keys=["a", "b"]).dropna()
    out = {"model": model, "baseline": base, "arm": arm, "metric": col, "n_tasks": int(len(joined)),
           "n_runs_base": int(len(a)), "n_runs_arm": int(len(b)),
           "mean_base": float(a[col].astype(float).mean()), "mean_arm": float(b[col].astype(float).mean())}
    out["diff"] = out["mean_arm"] - out["mean_base"]
    if len(joined) >= 2:
        pt = stats.paired_t_test(list(joined.a), list(joined.b))
        out["paired_p"], out["paired_d"] = pt.get("p"), pt.get("cohens_d") or pt.get("dz")
    wins, losses = int((joined.b > joined.a).sum()), int((joined.b < joined.a).sum())
    out.update({"tasks_better": wins, "tasks_worse": losses, "tasks_tied": int(len(joined) - wins - losses),
                "sign_p": sign_test_p(wins, losses)})
    if binary:
        fe = stats.fisher_exact(int(a[col].astype(float).sum()), len(a), int(b[col].astype(float).sum()), len(b))
        out["fisher_p_runlevel"] = fe.get("p")
    else:
        xa, xb = list(a[col].astype(float).dropna()), list(b[col].astype(float).dropna())
        if len(xa) > 1 and len(xb) > 1:
            out["welch_p_runlevel"] = stats.welch_t_test(xa, xb).get("p")
            out["mwu_p_runlevel"] = stats.mann_whitney_u(xa, xb).get("p")
    return out


def holm_in_place(rows: list[dict], key: str, adj_key: str) -> None:
    idx = [i for i, r in enumerate(rows) if r.get(key) is not None]
    adj = stats.holm([rows[i][key] for i in idx])
    for i, p in zip(idx, adj):
        rows[i][adj_key] = p


# ------------------------------------------------------------------- tables --

def fmt(x, d=2):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def save_table(df: pd.DataFrame, path: Path, caption: str, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(df.to_markdown(index=False) if _has_tabulate() else df.to_string(index=False), encoding="utf-8")
    cols = "l" * 2 + "r" * (len(df.columns) - 2)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(tex_escape(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.with_suffix(".tex").write_text("\n".join(lines), encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, arm), g in df.groupby(["model_tag", "arm"], observed=True):
        s = g[g.scored]
        n = len(s)
        k = int(s.hidden_pass.sum()) if n else 0
        lo, hi = wilson(k, n)
        sp = s.self_pass.dropna()
        row = {"Model": model, "Arm": ARM_LABEL.get(str(arm), str(arm)), "n runs": len(g), "scored": n,
               "Hidden pass %": f"{100 * k / n:.0f} [{100 * lo:.0f}-{100 * hi:.0f}]" if n else "--",
               "Self-test pass %": f"{100 * sp.astype(float).mean():.0f}" if len(sp) else "--",
               "False-pass %": f"{100 * s.false_pass.astype(float).mean():.0f}" if n and s.false_pass.notna().any() else "--",
               "Errored runs": int((g.status == "error").sum())}
        for col, label, scale in CONT:
            v = pd.to_numeric(g[col], errors="coerce").dropna() * scale
            row[label] = f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}" if len(v) else "--"
        rows.append(row)
    return pd.DataFrame(rows)


def comparisons(df: pd.DataFrame, baseline: str) -> tuple[pd.DataFrame, list[dict]]:
    allrows = []
    for model in sorted(df.model_tag.unique()):
        fam: dict[str, list[dict]] = {}
        for arm in [a for a in df.arm.cat.categories if a != baseline]:
            for col, binary in [("hidden_pass", True), ("hidden_fraction", False)] + [(c, False) for c, *_ in CONT]:
                r = compare_pair(df.assign(hidden_pass=df.hidden_pass.astype(float) if df.hidden_pass.notna().any() else df.hidden_pass),
                                 model, baseline, arm, col, binary)
                if r:
                    fam.setdefault(col, []).append(r)
        for col, rs in fam.items():
            holm_in_place(rs, "sign_p", "sign_p_holm")
            holm_in_place(rs, "paired_p", "paired_p_holm")
            allrows += rs
    tbl = pd.DataFrame([{
        "Model": r["model"], "Metric": r["metric"], "Arm vs " + baseline: ARM_LABEL.get(r["arm"], r["arm"]),
        "Base": fmt(r["mean_base"], 3), "Arm": fmt(r["mean_arm"], 3), "Diff": fmt(r["diff"], 3),
        "Tasks better/worse": f"{r['tasks_better']}/{r['tasks_worse']}", "n tasks": r["n_tasks"],
        "Paired p (Holm)": fmt(r.get("paired_p_holm"), 3), "Sign p (Holm)": fmt(r.get("sign_p_holm"), 3),
        "Run-level p (optimistic)": fmt(r.get("fisher_p_runlevel", r.get("welch_p_runlevel")), 3)} for r in allrows])
    return tbl, allrows


def cross_model(df: pd.DataFrame, baseline: str) -> tuple[pd.DataFrame, list[dict]]:
    sized = sorted([(model_size_b(m), m) for m in df.model_tag.unique() if model_size_b(m)])
    if len(sized) < 2:
        return pd.DataFrame(), []
    small, large = sized[0][1], sized[-1][1]
    rows, res = [], []
    for arm in df.arm.cat.categories:
        for ref_arm in [a for a in ("A0", baseline) if a in set(df.arm)]:
            sa = df[(df.model_tag == small) & (df.arm == arm) & df.scored]
            lb = df[(df.model_tag == large) & (df.arm == ref_arm) & df.scored]
            if sa.empty or lb.empty:
                continue
            ta, tb = per_task(sa.assign(hidden_pass=sa.hidden_pass.astype(float)), "hidden_pass"), \
                per_task(lb.assign(hidden_pass=lb.hidden_pass.astype(float)), "hidden_pass")
            j = pd.concat([ta, tb], axis=1, keys=["s", "l"]).dropna()
            wins, losses = int((j.s > j.l).sum()), int((j.s < j.l).sum())
            fe = stats.fisher_exact(int(lb.hidden_pass.sum()), len(lb), int(sa.hidden_pass.sum()), len(sa))
            r = {"small_model": small, "small_arm": str(arm), "large_model": large, "large_arm": ref_arm,
                 "small_rate": float(sa.hidden_pass.astype(float).mean()), "large_rate": float(lb.hidden_pass.astype(float).mean()),
                 "tasks_small_better": wins, "tasks_small_worse": losses, "sign_p": sign_test_p(wins, losses),
                 "fisher_p_runlevel": fe.get("p"), "n_tasks": int(len(j))}
            res.append(r)
            rows.append({"Small model + arm": f"{small} {ARM_LABEL.get(str(arm), arm)}", "vs": f"{large} {ARM_LABEL.get(ref_arm, ref_arm)}",
                         "Small hidden pass": fmt(r["small_rate"], 2), "Large hidden pass": fmt(r["large_rate"], 2),
                         "Tasks small better/worse": f"{wins}/{losses}", "Sign p": fmt(r["sign_p"], 3),
                         "Run-level p (optimistic)": fmt(r["fisher_p_runlevel"], 3)})
    return pd.DataFrame(rows), res


# ------------------------------------------------------------------ figures --

def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def fig_pass_rate(df, out):
    models = sorted(df.model_tag.unique())
    arms = list(df.arm.cat.categories)
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], models):
        xs, ys, lo, hi = [], [], [], []
        for i, arm in enumerate(arms):
            s = df[(df.model_tag == m) & (df.arm == arm) & df.scored]
            if s.empty:
                continue
            k, n = int(s.hidden_pass.astype(float).sum()), len(s)
            l, h = wilson(k, n)
            xs.append(i); ys.append(k / n); lo.append(k / n - l); hi.append(h - k / n)
        ax.bar(xs, ys, yerr=[lo, hi], capsize=3, color="#4c78a8")
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([str(a) for a in arms], rotation=0)
        ax.set_title(m, fontsize=10); ax.set_ylim(0, 1.05); ax.set_xlabel("arm")
    axes[0][0].set_ylabel("hidden-test pass rate (Wilson 95% CI)")
    savefig(fig, out / "fig_pass_rate")


def fig_false_pass(df, out):
    models = sorted(df.model_tag.unique())
    arms = [a for a in df.arm.cat.categories if a != "A0" or True]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], models):
        for j, (col, lab, c) in enumerate([("self_pass", "passed its OWN tests", "#f58518"), ("hidden_pass", "passed HIDDEN tests", "#4c78a8")]):
            vals = []
            for arm in arms:
                s = df[(df.model_tag == m) & (df.arm == arm) & df.scored]
                vals.append(float(s[col].astype(float).mean()) if len(s) and s[col].notna().any() else float("nan"))
            ax.bar([i + (j - 0.5) * 0.38 for i in range(len(arms))], vals, width=0.38, label=lab, color=c)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([str(a) for a in arms]); ax.set_title(m, fontsize=10); ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("rate"); axes[0][-1].legend(fontsize=7, loc="lower right")
    savefig(fig, out / "fig_false_pass")


def fig_box(df, col, label, scale, out, name):
    models = sorted(df.model_tag.unique())
    arms = list(df.arm.cat.categories)
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        data = [pd.to_numeric(df[(df.model_tag == m) & (df.arm == a)][col], errors="coerce").dropna() * scale for a in arms]
        keep = [(str(a), d) for a, d in zip(arms, data) if len(d)]
        if keep:
            ax.boxplot([d for _, d in keep])       # (labels= was renamed across matplotlib versions)
            ax.set_xticks(range(1, len(keep) + 1)); ax.set_xticklabels([a for a, _ in keep])
        ax.set_title(m, fontsize=10); ax.set_xlabel("arm")
    axes[0][0].set_ylabel(label)
    savefig(fig, out / name)


def fig_ifs(df, out):
    models = sorted(df.model_tag.unique())
    arms = list(df.arm.cat.categories)
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], models):
        for j, (col, lab) in enumerate([("ifs_final", "IFS (substring)"), ("ifs_final_strict", "IFS (strict)")]):
            vals = [float(pd.to_numeric(df[(df.model_tag == m) & (df.arm == a)][col], errors="coerce").mean()) for a in arms]
            ax.bar([i + (j - 0.5) * 0.38 for i in range(len(arms))], vals, width=0.38, label=lab)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([str(a) for a in arms]); ax.set_title(m, fontsize=10); ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("fraction of request entities kept"); axes[0][-1].legend(fontsize=7)
    savefig(fig, out / "fig_ifs")


def fig_heatmap(df, out):
    s = df[df.scored].copy()
    s["cell"] = s.model_tag.astype(str) + " " + s.arm.astype(str)
    s["hidden_pass"] = s.hidden_pass.astype(float)
    pv = s.pivot_table(index="task_id", columns="cell", values="hidden_pass", aggfunc="mean")
    cols = [c for m in sorted(s.model_tag.unique()) for a in df.arm.cat.categories for c in [f"{m} {a}"] if c in pv.columns]
    pv = pv[cols]
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(cols) + 2), max(3, 0.32 * len(pv) + 1.5)))
    im = ax.imshow(pv.values, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="hidden-test pass rate")
    savefig(fig, out / "fig_task_heatmap")


def fig_cross(res, out):
    if not res:
        return
    labels = [f"{r['small_arm']}@{r['small_model'].split('_')[-1]}\nvs {r['large_arm']}@{r['large_model'].split('_')[-1]}" for r in res]
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(res)), 3.8))
    ax.bar(range(len(res)), [r["small_rate"] for r in res], width=0.38, label="small model + arm", color="#4c78a8")
    ax.bar([i + 0.38 for i in range(len(res))], [r["large_rate"] for r in res], width=0.38, label="large model + reference arm", color="#e45756")
    ax.set_xticks([i + 0.19 for i in range(len(res))]); ax.set_xticklabels(labels, fontsize=6, rotation=60, ha="right")
    ax.set_ylabel("hidden-test pass rate"); ax.legend(fontsize=7); ax.set_ylim(0, 1.05)
    savefig(fig, out / "fig_cross_model")


# ------------------------------------------------------------------ summary --

def claim_lines(rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        if r["metric"] != "hidden_pass":
            continue
        p = r.get("sign_p_holm")
        ok = r["diff"] > 0 and p is not None and p < 0.05 and r["n_tasks"] >= 5
        verdict = ("SUPPORTED (paired by task, Holm-adjusted p<0.05)" if ok else
                   "worse than baseline" if r["diff"] < 0 and p is not None and p < 0.05 else "NOT supported / inconclusive")
        out.append(f"- {r['model']}: {ARM_LABEL.get(r['arm'], r['arm'])} vs {ARM_LABEL.get(r['baseline'], r['baseline'])} — hidden pass "
                   f"{r['mean_base']:.2f} -> {r['mean_arm']:.2f} ({r['tasks_better']} tasks better, {r['tasks_worse']} worse, "
                   f"{r['tasks_tied']} tied; sign-test p_holm={fmt(p, 3)}): **{verdict}**")
    return out


def summary(df, main, comp_rows, cross_res, baseline, out: Path):
    unscored = int((~df.scored).sum())
    errs = int((df.status == "error").sum())
    lines = ["# Results summary (auto-generated — check every number against the CSVs)", "",
             f"- runs: {len(df)}; scored against hidden tests: {int(df.scored.sum())}; unscored: {unscored}; errored: {errs}",
             f"- models: {', '.join(sorted(df.model_tag.unique()))}; arms: {', '.join(map(str, df.arm.cat.categories))}",
             f"- baseline for comparisons: {ARM_LABEL.get(baseline, baseline)}", "", "## Main table", "", main.to_string(index=False), "",
             "## Claims, arm by arm (hidden-test pass rate)", "", *claim_lines(comp_rows), ""]
    if cross_res:
        lines += ["## Small model + pipeline vs large model", ""]
        for r in cross_res:
            lines.append(f"- {r['small_model']} {r['small_arm']} ({r['small_rate']:.2f}) vs {r['large_model']} {r['large_arm']} "
                         f"({r['large_rate']:.2f}); tasks small better/worse {r['tasks_small_better']}/{r['tasks_small_worse']}, sign p={fmt(r['sign_p'], 3)}")
        lines.append("")
    warns = []
    if errs:
        warns.append(f"{errs} run(s) ended in `error` (backend/Ollama problem, not a model failure) — they are counted as FAILURES; look at them before publishing.")
    if unscored:
        warns.append(f"{unscored} run(s) could not be matched to a hidden-test task and are excluded from pass rates.")
    small = df[df.scored].groupby(["model_tag", "arm"], observed=True).task_id.nunique()
    if (small < 5).any():
        warns.append("some model/arm cells have fewer than 5 tasks — paired tests have almost no power; do not claim significance.")
    if df.rep.nunique() < 3:
        warns.append("fewer than 3 repetitions — run-to-run variance is not estimated.")
    ctx = df.groupby("arm", observed=True).num_ctx.first() if "num_ctx" in df else None
    lines += ["## Data-quality warnings", "", *(["- " + w for w in warns] or ["- none detected"]), "",
              "## Reminders for the paper", "",
              "- State the baseline honestly: A1 is an SE-Blackboard-STYLE baseline inside our framework, not their original system re-run.",
              "- Report temperature, seed policy, num_ctx, quantisation, hardware (see environment.json).",
              "- Pipeline 'self-test pass' is graded by tests the model wrote; 'hidden pass' is the real correctness number.",
              "- IFS is computed on the final output with the same matcher for every arm; 'strict' is our own stricter matcher (difflib>=0.85), not SE-Blackboard's."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--baseline", default="A1")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "paper_assets"
    df = load(res)
    if a.baseline not in set(df.arm):
        raise SystemExit(f"baseline arm {a.baseline} not in the data; arms present: {list(df.arm.cat.categories)}")

    main_t = main_table(df)
    save_table(main_t, out / "tables" / "main_table", "Hidden-test pass rate (Wilson 95\\% CI), self-test pass, false-pass rate and cost per model and arm.", "tab:main")
    comp_t, comp_rows = comparisons(df, a.baseline)
    if len(comp_t):
        save_table(comp_t, out / "tables" / "comparisons", f"Each arm against the {a.baseline} baseline, paired by task (Holm-adjusted).", "tab:comparisons")
    cm_t, cm_rows = cross_model(df, a.baseline)
    if len(cm_t):
        save_table(cm_t, out / "tables" / "cross_model", "Small model with the pipeline against the larger model without it.", "tab:crossmodel")
    bt = df[df.scored].assign(hidden_pass=lambda d: d.hidden_pass.astype(float)).pivot_table(
        index="task_id", columns=["model_tag", "arm"], values="hidden_pass", aggfunc="mean", observed=True).round(2)
    bt.columns = [f"{m} {a_}" for m, a_ in bt.columns]
    save_table(bt.reset_index(), out / "tables" / "by_task", "Hidden-test pass rate per task.", "tab:bytask")

    figs = out / "figures"
    fig_pass_rate(df, figs)
    fig_false_pass(df, figs)
    fig_box(df, "total_ms", "latency per run (s)", 1e-3, figs, "fig_latency")
    fig_box(df, "tokens_total", "tokens per run", 1.0, figs, "fig_tokens")
    fig_ifs(df, figs)
    fig_heatmap(df, figs)
    fig_cross(cm_rows, figs)

    (out / "stats.json").write_text(json.dumps({"comparisons": comp_rows, "cross_model": cm_rows}, indent=2, default=str), encoding="utf-8")
    summary(df, main_t, comp_rows, cm_rows, a.baseline, out)
    print(f"wrote tables, figures, SUMMARY.md, stats.json to {out}")


if __name__ == "__main__":
    main()
