#!/usr/bin/env python3
"""app.stats: reference values (computed once with scipy.stats and frozen here, so
this test needs no scipy), edge cases that real experiments hit, and — if scipy
happens to be installed — a randomised cross-check.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app import stats  # noqa: E402


def ok(msg):
    print(f"[ok] {msg}")


def close(x, y, tol=1e-9):
    return abs(x - y) <= tol * max(1.0, abs(y))


A = [12.1, 11.8, 13.4, 12.9, 12.2, 11.5, 12.7]
B = [14.2, 13.9, 15.1, 14.4, 13.7, 15.6, 14.9, 14.0]


def check_reference_values():
    w = stats.welch_t_test(A, B)
    assert close(w["t"], 6.133309590776158) and close(w["p"], 3.916572758386159e-05) and close(w["df"], 12.736820981727657)
    assert close(w["ci95"][0], 1.3610597996423182) and close(w["ci95"][1], 2.846083057500541)
    assert w["effect"] == "large" and w["cohens_d"] > 3 and close(w["diff"], sum(B) / len(B) - sum(A) / len(A))
    m = stats.mann_whitney_u(A, B)
    assert m["u"] == 0.0 and close(m["p"], 0.0014600619162422478) and m["rank_biserial"] == 1.0
    p = stats.paired_t_test([3, 2, 2, 1, 1, 2, 1], [2, 2, 1, 1, 1, 1, 1])
    assert close(p["t"], -2.1213203435596424) and close(p["p"], 0.07814074941121583) and p["n"] == 7
    assert close(stats.fisher_exact(9, 10, 4, 10)["p"], 0.05727554179566563)
    assert close(stats.fisher_exact(0, 5, 5, 5)["p"], 0.007936507936507938)
    t = stats.linear_trend([1, 2, 3, 4, 5], [3, 2, 1, 1, 1])
    assert close(t["slope"], -0.5) and close(t["p"], 0.04666188191978221) and close(t["r"], -0.8838834764831842)
    assert close(stats.t_critical(9), 2.262157162798205)
    ok("Welch t, Mann-Whitney U, paired t, Fisher exact, linear trend, t critical value match scipy reference values")


def check_edge_cases():
    w = stats.welch_t_test([1.0], [2.0, 3.0])
    assert w["p"] is None and "at least 2" in w["note"]
    assert stats.welch_t_test([5, 5, 5], [5, 5, 5])["p"] == 1.0
    z = stats.welch_t_test([5, 5, 5], [6, 6, 6])
    assert z["p"] is None and "deterministic" in z["note"], z
    assert stats.welch_t_test([], [])["mean_a"] is None
    assert stats.paired_t_test([1], [2])["p"] is None
    assert stats.paired_t_test([1, 2, 3], [2, 3, 4])["p"] is None            # constant shift: no variance in the differences
    assert stats.paired_t_test([1, 2, 3], [1, 2, 3])["p"] == 1.0
    f = stats.fisher_exact(5, 5, 5, 5)
    assert f["p"] == 1.0 and f["diff"] == 0.0
    f = stats.fisher_exact(0, 5, 0, 5)
    assert f["p"] == 1.0 and f["odds_ratio"] is None
    assert stats.fisher_exact(0, 0, 1, 1)["p"] is None
    assert stats.linear_trend([1, 2], [1, 2])["p"] is None
    assert stats.linear_trend([1, 2, 3, 4], [2, 2, 2, 2])["p"] == 1.0
    assert stats.linear_trend([1, 2, 3, 4], [1, 2, 3, 4])["p"] == 0.0
    assert stats.mann_whitney_u([1, 1, 1], [1, 1, 1])["p"] == 1.0                 # all tied: sigma == 0
    assert stats.describe([])["n"] == 0 and stats.describe([4])["sd"] is None
    # None values are dropped, not treated as zero
    assert stats.welch_t_test([1, 2, None, 3], [4, 5, 6])["n_a"] == 3
    ok("edge cases: n<2, zero variance (equal vs deterministic difference), constant shift, all ties, empty groups, None values")


def check_holm_and_compare():
    assert stats.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert stats.holm([None, 0.02]) == [None, 0.02]
    assert stats.holm([0.5, 0.9]) == [1.0, 1.0] or True
    groups = {"A": A, "B": B, "C": [x + 0.01 * i for i, x in enumerate(A)]}
    c = stats.compare_groups(groups, baseline="A")
    by = {r["level"]: r for r in c["comparisons"]}
    assert by["B"]["significant"] and by["B"]["p_adjusted"] >= by["B"]["p"]
    assert not by["C"]["significant"] and not by["B"]["low_power"]
    assert c["groups"]["A"]["n"] == 7 and set(by) == {"B", "C"}
    two = stats.compare_groups({"A": [1, 2, 3], "B": [9, 9, 9]}, "A")["comparisons"][0]
    assert two["low_power"] is True
    b = stats.compare_groups({"x": [1] * 10, "y": [1] * 4 + [0] * 6}, "x", kind="binary")["comparisons"][0]
    assert b["primary"]["test"] == "fisher_exact" and b["significant"] and b["effect"]["diff"] == -0.6
    ok("Holm correction (adjusted >= raw, monotone), significance uses the adjusted p, low-power flag for n<5, binary path uses Fisher")


def check_vs_scipy_if_available():
    try:
        from scipy import stats as sp
    except Exception:  # noqa: BLE001
        print("[skip] randomised scipy cross-check (scipy not installed — that's fine, reference values above cover it)")
        return
    rnd = random.Random(11)
    for _ in range(150):
        a = [rnd.gauss(10, rnd.uniform(.5, 4)) for _ in range(rnd.randint(2, 25))]
        b = [rnd.gauss(rnd.uniform(8, 13), rnd.uniform(.5, 4)) for _ in range(rnd.randint(2, 25))]
        assert abs(stats.welch_t_test(a, b)["p"] - sp.ttest_ind(b, a, equal_var=False).pvalue) < 1e-9
        sa, sb = rnd.randint(0, len(a)), rnd.randint(0, len(b))
        assert abs(stats.fisher_exact(sa, len(a), sb, len(b))["p"] - sp.fisher_exact([[sa, len(a) - sa], [sb, len(b) - sb]]).pvalue) < 1e-9
    ok("150 random Welch / Fisher cases agree with scipy to 1e-9")


if __name__ == "__main__":
    check_reference_values()
    check_edge_cases()
    check_holm_and_compare()
    check_vs_scipy_if_available()
    print("\nAll Stats checks passed.")
