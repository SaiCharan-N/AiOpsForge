"""Small-sample statistics for comparing experiment conditions.

Pure Python (math only) so the backend needs no scipy; tests/test_stats.py
checks every function against scipy.

What is provided, and when to use which:

  welch_t_test      two independent groups, continuous metric (tokens, latency, attempts).
                    Does NOT assume equal variances. Reports the mean difference
                    with a 95% CI and Cohen's d.
  mann_whitney_u    same situation when the metric is skewed (latency usually is)
                    or you don't want to assume normality. Normal approximation with
                    tie and continuity corrections — treat p-values with n < ~8 per
                    group as rough.
  paired_t_test     the same item measured twice (first run vs repeat run).
  fisher_exact      two proportions (pass / fail per run). Exact, fine for tiny n.
  linear_trend      does a metric rise or fall with position in a sequence.
  holm              multiple-comparison correction for several comparisons.
  compare_groups    the one entry point the experiment runner uses: baseline vs every
                    other level, with Holm correction, effect-size labels and a
                    low-power warning.

Statistical significance with the sample sizes a local-LLM experiment can afford
(5-30 runs per condition) is fragile: always report n, the effect size and the
CI next to the p-value. compare_groups flags groups with n < 5.
"""
import math
from statistics import mean, median, stdev
from typing import Optional, Sequence

MIN_N_WARNING = 5


# ------------------------------------------------------- distributions ------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny, eps = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student's t with `df` degrees of freedom."""
    if math.isinf(t):
        return 0.0
    return betai(df / 2.0, 0.5, df / (df + t * t))


def t_critical(df: float, confidence: float = 0.95) -> float:
    """Two-sided critical value of Student's t (bisection on the CDF)."""
    target = 1.0 - confidence
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


# --------------------------------------------------------- descriptives -----

def describe(xs: Sequence[float]) -> dict:
    xs = [float(x) for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return {"n": 0}
    sd = stdev(xs) if n > 1 else None
    return {"n": n, "mean": mean(xs), "sd": sd, "sem": (sd / math.sqrt(n)) if sd is not None else None,
            "median": median(xs), "min": min(xs), "max": max(xs)}


def _effect_label(d: Optional[float]) -> Optional[str]:
    if d is None:
        return None
    a = abs(d)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


# ------------------------------------------------------------- tests --------

def welch_t_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Welch's t-test of b against a (difference = mean(b) - mean(a))."""
    a, b = [float(x) for x in a if x is not None], [float(x) for x in b if x is not None]
    out = {"test": "welch_t", "n_a": len(a), "n_b": len(b), "t": None, "df": None, "p": None,
           "mean_a": mean(a) if a else None, "mean_b": mean(b) if b else None,
           "diff": None, "ci95": None, "cohens_d": None, "effect": None, "note": None}
    if out["mean_a"] is not None and out["mean_b"] is not None:
        out["diff"] = out["mean_b"] - out["mean_a"]
    if len(a) < 2 or len(b) < 2:
        out["note"] = "need at least 2 observations in each group"
        return out
    va, vb = stdev(a) ** 2, stdev(b) ** 2
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    if se2 == 0:
        out["p"] = 1.0 if out["diff"] == 0 else None
        out["note"] = "identical values" if out["diff"] == 0 else "no variance within either group; the difference is deterministic but cannot be tested"
        return out
    t = out["diff"] / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    half = t_critical(df) * math.sqrt(se2)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = out["diff"] / pooled if pooled > 0 else None
    out.update(t=t, df=df, p=t_two_sided_p(t, df), ci95=(out["diff"] - half, out["diff"] + half),
               cohens_d=d, effect=_effect_label(d))
    return out


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired t-test (difference = b - a, pairs matched by position)."""
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None]
    out = {"test": "paired_t", "n": len(pairs), "t": None, "df": None, "p": None, "mean_diff": None,
           "ci95": None, "cohens_dz": None, "effect": None, "note": None}
    if len(pairs) < 2:
        out["note"] = "need at least 2 pairs"
        return out
    diffs = [y - x for x, y in pairs]
    md, sd = mean(diffs), stdev(diffs)
    out["mean_diff"] = md
    if sd == 0:
        out["p"] = 1.0 if md == 0 else None
        out["note"] = "identical values" if md == 0 else "all pairs differ by exactly the same amount; cannot be tested"
        return out
    se = sd / math.sqrt(len(diffs))
    df = len(diffs) - 1
    t = md / se
    half = t_critical(df) * se
    out.update(t=t, df=df, p=t_two_sided_p(t, df), ci95=(md - half, md + half), cohens_dz=md / sd,
               effect=_effect_label(md / sd))
    return out


def _ranks(values: list[float]) -> tuple[list[float], list[int]]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, ties, i = [0.0] * len(values), [], 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> dict:
    """Mann-Whitney U (normal approximation, tie + continuity corrected)."""
    a, b = [float(x) for x in a if x is not None], [float(x) for x in b if x is not None]
    n1, n2 = len(a), len(b)
    out = {"test": "mann_whitney_u", "n_a": n1, "n_b": n2, "u": None, "p": None,
           "median_a": median(a) if a else None, "median_b": median(b) if b else None,
           "rank_biserial": None, "note": "normal approximation; rough for n < 8 per group"}
    if n1 < 1 or n2 < 1:
        out["note"] = "need at least 1 observation in each group"
        return out
    ranks, ties = _ranks(a + b)
    u1 = sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu = n1 * n2 / 2.0
    tie_term = sum(t ** 3 - t for t in ties) / (n * (n - 1)) if n > 1 else 0.0
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie_term)
    out["u"] = u1
    out["rank_biserial"] = 1.0 - 2.0 * u1 / (n1 * n2)       # >0: a tends to be larger
    if sigma2 <= 0:
        out["p"] = 1.0
        return out
    z = (abs(u1 - mu) - 0.5) / math.sqrt(sigma2)
    out["p"] = normal_two_sided_p(max(z, 0.0))
    return out


def fisher_exact(succ_a: int, n_a: int, succ_b: int, n_b: int) -> dict:
    """Two-sided Fisher's exact test on a 2x2 table (successes / failures per group)."""
    out = {"test": "fisher_exact", "n_a": n_a, "n_b": n_b, "rate_a": None, "rate_b": None, "diff": None,
           "odds_ratio": None, "p": None, "note": None}
    if n_a < 1 or n_b < 1:
        out["note"] = "need at least 1 observation in each group"
        return out
    fa, fb = n_a - succ_a, n_b - succ_b
    out["rate_a"], out["rate_b"] = succ_a / n_a, succ_b / n_b
    out["diff"] = out["rate_b"] - out["rate_a"]
    if succ_a * fb == 0 and fa * succ_b == 0:
        out["odds_ratio"] = None
    else:
        out["odds_ratio"] = (succ_b * fa) / (fb * succ_a) if fb * succ_a else math.inf
    k, n = succ_a + succ_b, n_a + n_b       # successes overall, total

    def pmf(x: int) -> float:               # P(group A has x successes)
        return math.comb(k, x) * math.comb(n - k, n_a - x) / math.comb(n, n_a)

    lo, hi = max(0, n_a - (n - k)), min(n_a, k)
    obs = pmf(succ_a)
    out["p"] = min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= obs * (1 + 1e-7)))
    return out


def linear_trend(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """Least-squares line of y on x with a t-test on the slope."""
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pts)
    out = {"test": "linear_trend", "n": n, "slope": None, "intercept": None, "r": None, "p": None, "note": None}
    if n < 3:
        out["note"] = "need at least 3 points"
        return out
    mx, my = mean(p[0] for p in pts), mean(p[1] for p in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    syy = sum((y - my) ** 2 for _, y in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    if sxx == 0:
        out["note"] = "x has no variance"
        return out
    slope = sxy / sxx
    out.update(slope=slope, intercept=my - slope * mx)
    if syy == 0:
        out.update(r=0.0, p=1.0, note="y is constant")
        return out
    r = sxy / math.sqrt(sxx * syy)
    out["r"] = r
    if abs(r) >= 1.0:
        out["p"] = 0.0
        return out
    t = r * math.sqrt((n - 2) / (1 - r * r))
    out["p"] = t_two_sided_p(t, n - 2)
    return out


def holm(pvalues: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Holm-Bonferroni adjusted p-values (None entries are left alone)."""
    idx = sorted((i for i, p in enumerate(pvalues) if p is not None), key=lambda i: pvalues[i])
    m = len(idx)
    adj: list[Optional[float]] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


# -------------------------------------------------- the experiment entry ----

def compare_groups(groups: dict[str, Sequence], baseline: str, kind: str = "continuous",
                   alpha: float = 0.05) -> dict:
    """Compares every level against `baseline`.

    kind="continuous": groups map level -> list of numbers; runs Welch's t-test
        and Mann-Whitney U.
    kind="binary": groups map level -> list of 0/1; runs Fisher's exact test.

    p-values across the levels compared are Holm-adjusted; `significant` uses
    the ADJUSTED p. `low_power` is True when either group has fewer than
    MIN_N_WARNING observations — treat such results as inconclusive whatever p says.
    """
    base = groups[baseline]
    levels = [lv for lv in groups if lv != baseline]
    results = []
    for lv in levels:
        g = groups[lv]
        if kind == "binary":
            t = fisher_exact(int(sum(base)), len(base), int(sum(g)), len(g))
            entry = {"level": lv, "primary": t, "secondary": None, "p": t["p"],
                     "effect": {"diff": t["diff"], "odds_ratio": t["odds_ratio"]}}
        else:
            t = welch_t_test(base, g)
            mw = mann_whitney_u(base, g)
            entry = {"level": lv, "primary": t, "secondary": mw, "p": t["p"],
                     "effect": {"diff": t["diff"], "ci95": t["ci95"], "cohens_d": t["cohens_d"], "label": t["effect"]}}
        entry["n_baseline"], entry["n"] = len(base), len(g)
        entry["low_power"] = min(len(base), len(g)) < MIN_N_WARNING
        results.append(entry)
    adjusted = holm([r["p"] for r in results])
    for r, adj in zip(results, adjusted):
        r["p_adjusted"] = adj
        r["significant"] = bool(adj is not None and adj < alpha)
    return {"kind": kind, "baseline": baseline, "alpha": alpha, "comparisons": results,
            "groups": {lv: describe(v) for lv, v in groups.items()}}
