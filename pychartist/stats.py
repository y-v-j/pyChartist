"""Statistical tests for exploratory data analysis (EDA).

Follows the variable-type decision table in the project's EDA reference:

=====================================  ==============================  =============================
Analysis                               Charts                          Tests
=====================================  ==============================  =============================
1 numerical                            histogram, box plot             Shapiro-Wilk, Kolmogorov-Smirnov
1 categorical                          bar, pie                        chi-square goodness of fit
2 numerical                            scatter, correlation heatmap    Pearson r, Spearman ρ
1 categorical + 1 numerical (2 groups) grouped box / violin            Welch t-test, Mann-Whitney U
1 categorical + 1 numerical (3+)       grouped box / violin            one-way ANOVA, Kruskal-Wallis
2 categorical                          stacked bar, mosaic             chi-square independence, Fisher
=====================================  ==============================  =============================

Every test returns a :class:`TestResult`; ``results_frame`` turns a list into a DataFrame.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass
class TestResult:
    test: str
    statistic: float
    pvalue: float
    n: int
    alpha: float = 0.05
    effect_size: float | None = None
    effect_name: str | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return bool(self.pvalue < self.alpha)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        d.pop("extra")
        d.update(self.extra)
        return d

    def __str__(self) -> str:
        eff = f", {self.effect_name} = {self.effect_size:.3g}" if self.effect_size is not None else ""
        return f"{self.test}: stat = {self.statistic:.4g}, p = {self.pvalue:.3g}{eff} (n = {self.n}). {self.note}"


def results_frame(results) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])


def _clean(x) -> np.ndarray:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    return a[np.isfinite(a)]


# ------------------------------------------------------------- 1 numerical
def describe(x) -> pd.Series:
    """Central tendency, spread and shape of one numeric variable."""
    a = _clean(x)
    q1, med, q3 = np.percentile(a, [25, 50, 75]) if len(a) else (np.nan,) * 3
    iqr = q3 - q1
    return pd.Series({
        "n": len(a), "missing": int(pd.Series(x).isna().sum()), "mean": a.mean() if len(a) else np.nan,
        "std": a.std(ddof=1) if len(a) > 1 else np.nan, "min": a.min() if len(a) else np.nan, "q1": q1,
        "median": med, "q3": q3, "max": a.max() if len(a) else np.nan, "iqr": iqr,
        "skewness": sps.skew(a) if len(a) > 2 else np.nan,
        "excess_kurtosis": sps.kurtosis(a) if len(a) > 3 else np.nan,
        "outliers_1.5iqr": int(((a < q1 - 1.5 * iqr) | (a > q3 + 1.5 * iqr)).sum()),
    })


def shapiro_wilk(x, alpha: float = 0.05) -> TestResult:
    """Shapiro-Wilk normality test (scipy; p-value approximate above n = 5000)."""
    a = _clean(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        stat, p = sps.shapiro(a)
    note = "consistent with normality" if p >= alpha else "deviates from normality"
    if len(a) > 5000:
        note += " (n > 5000: p-value approximate, and tiny deviations become significant)"
    return TestResult("Shapiro-Wilk", stat, p, len(a), alpha, note=note)


def kolmogorov_smirnov(x, alpha: float = 0.05, dist: str = "norm") -> TestResult:
    """One-sample KS test of the standardised sample against N(0, 1).

    Because mean/sd are estimated from the sample, the classical KS p-value is conservative; the
    Lilliefors-corrected p-value is reported in ``extra['lilliefors_p']``.
    """
    a = _clean(x)
    z = (a - a.mean()) / a.std(ddof=1)
    stat, p = sps.kstest(z, dist)
    extra = {}
    try:
        from statsmodels.stats.diagnostic import lilliefors
        extra["lilliefors_p"] = float(lilliefors(a, dist="norm", pvalmethod="table")[1])
    except Exception:  # statsmodels missing or table range exceeded
        pass
    note = "consistent with normality" if p >= alpha else "deviates from normality"
    return TestResult("Kolmogorov-Smirnov", stat, p, len(a), alpha, note=note, extra=extra)


def normality(x, alpha: float = 0.05) -> list[TestResult]:
    return [shapiro_wilk(x, alpha), kolmogorov_smirnov(x, alpha)]


# ----------------------------------------------------------- 1 categorical
def chi2_goodness_of_fit(x=None, *, counts=None, expected=None, alpha: float = 0.05) -> TestResult:
    """Do observed category frequencies match ``expected`` proportions (default: uniform)?

    Pass raw categorical values ``x`` or a ``counts`` Series/dict. ``expected`` may be
    proportions or counts keyed like the categories.
    """
    obs = pd.Series(counts) if counts is not None else pd.Series(x).value_counts(sort=False)
    obs = obs.astype(float)
    n = obs.sum()
    if expected is None:
        exp = np.full(len(obs), n / len(obs))
    else:
        e = pd.Series(expected).reindex(obs.index).astype(float)
        exp = (e / e.sum() * n).to_numpy()
    stat, p = sps.chisquare(obs.to_numpy(), exp)
    w = np.sqrt(stat / n)
    note = "frequencies match expected proportions" if p >= alpha else "frequencies differ from expected"
    if (exp < 5).any():
        note += "; some expected counts < 5, the chi-square approximation is unreliable"
    return TestResult("Chi-square goodness of fit", stat, p, int(n), alpha, w, "Cohen's w", note,
                      {"dof": len(obs) - 1})


# ------------------------------------------------------------- 2 numerical
def pearson(x, y, alpha: float = 0.05) -> TestResult:
    a, b = _paired(x, y)
    res = sps.pearsonr(a, b)
    ci = res.confidence_interval()
    return TestResult("Pearson r", res.statistic, res.pvalue, len(a), alpha, res.statistic, "r",
                      _strength(res.statistic) + " linear relationship", {"ci_low": ci.low, "ci_high": ci.high})


def spearman(x, y, alpha: float = 0.05) -> TestResult:
    a, b = _paired(x, y)
    res = sps.spearmanr(a, b)
    return TestResult("Spearman rho", res.statistic, res.pvalue, len(a), alpha, res.statistic, "rho",
                      _strength(res.statistic) + " monotonic relationship")


def correlation(x, y) -> dict:
    """Pearson and Spearman correlation in one dict (used for chart annotations)."""
    p, s = pearson(x, y), spearman(x, y)
    return {"n": p.n, "pearson_r": p.statistic, "pearson_p": p.pvalue, "spearman_rho": s.statistic,
            "spearman_p": s.pvalue}


def _paired(x, y):
    d = pd.DataFrame({"x": pd.to_numeric(pd.Series(np.asarray(x)), errors="coerce"),
                      "y": pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce")}).dropna()
    return d["x"].to_numpy(float), d["y"].to_numpy(float)


def _strength(r: float) -> str:
    a = abs(r)
    s = "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"
    return f"{s} {'positive' if r > 0 else 'negative' if r < 0 else ''}".strip()


# ------------------------------------------------- categorical x numerical
def _groups(df, cat, num):
    d = df[[cat, num]].dropna()
    levels = list(pd.unique(d[cat]))
    return levels, [d.loc[d[cat] == lv, num].to_numpy(float) for lv in levels]


def t_test(a, b, alpha: float = 0.05, equal_var: bool = False) -> TestResult:
    """Independent two-sample t-test (Welch by default) with Cohen's d."""
    a, b = _clean(a), _clean(b)
    res = sps.ttest_ind(a, b, equal_var=equal_var)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp if sp else np.nan
    name = "Student t-test" if equal_var else "Welch t-test"
    note = "means differ" if res.pvalue < alpha else "no evidence the means differ"
    return TestResult(name, res.statistic, res.pvalue, len(a) + len(b), alpha, d, "Cohen's d", note,
                      {"dof": float(res.df), "mean_1": a.mean(), "mean_2": b.mean()})


def mann_whitney(a, b, alpha: float = 0.05) -> TestResult:
    """Mann-Whitney U (non-parametric two-group comparison) with rank-biserial correlation."""
    a, b = _clean(a), _clean(b)
    res = sps.mannwhitneyu(a, b, alternative="two-sided")
    rbc = 1 - 2 * res.statistic / (len(a) * len(b))
    note = "distributions differ in location" if res.pvalue < alpha else "no evidence of a difference"
    return TestResult("Mann-Whitney U", res.statistic, res.pvalue, len(a) + len(b), alpha, rbc,
                      "rank-biserial r", note, {"median_1": np.median(a), "median_2": np.median(b)})


def anova(*groups, alpha: float = 0.05) -> TestResult:
    """One-way ANOVA with eta-squared."""
    gs = [_clean(g) for g in groups]
    res = sps.f_oneway(*gs)
    allv = np.concatenate(gs)
    ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in gs)
    ss_t = ((allv - allv.mean()) ** 2).sum()
    note = "at least one group mean differs" if res.pvalue < alpha else "no evidence the means differ"
    return TestResult("One-way ANOVA", res.statistic, res.pvalue, len(allv), alpha, ss_b / ss_t if ss_t else np.nan,
                      "eta squared", note, {"groups": len(gs)})


def kruskal_wallis(*groups, alpha: float = 0.05) -> TestResult:
    """Kruskal-Wallis H with epsilon-squared."""
    gs = [_clean(g) for g in groups]
    res = sps.kruskal(*gs)
    n = sum(len(g) for g in gs)
    eps = res.statistic / ((n * n - 1) / (n + 1))
    note = "at least one group differs in location" if res.pvalue < alpha else "no evidence of a difference"
    return TestResult("Kruskal-Wallis H", res.statistic, res.pvalue, n, alpha, eps, "epsilon squared", note,
                      {"groups": len(gs)})


def compare_groups(df: pd.DataFrame, cat: str, num: str, alpha: float = 0.05) -> dict:
    """Compare ``num`` across levels of ``cat``; picks the tests from the number of groups and
    recommends parametric vs non-parametric from per-group Shapiro-Wilk and Levene's test."""
    levels, gs = _groups(df, cat, num)
    gs_ok = [g for g in gs if len(g) >= 3]
    normal = all(shapiro_wilk(g, alpha).pvalue >= alpha for g in gs_ok) if gs_ok else False
    lev = sps.levene(*gs) if len(gs) >= 2 else None
    if len(gs) < 2:
        raise ValueError(f"{cat!r} has fewer than 2 groups")
    if len(gs) == 2:
        tests = [t_test(gs[0], gs[1], alpha), mann_whitney(gs[0], gs[1], alpha)]
    else:
        tests = [anova(*gs, alpha=alpha), kruskal_wallis(*gs, alpha=alpha)]
    rec = tests[0] if normal else tests[1]
    return {"groups": levels, "normal_within_groups": normal,
            "levene_p": float(lev.pvalue) if lev is not None else np.nan, "tests": tests, "recommended": rec}


# ---------------------------------------------------------- 2 categorical
def chi2_independence(df: pd.DataFrame, a: str, b: str, alpha: float = 0.05) -> TestResult:
    """Chi-square test of independence on the ``a`` x ``b`` contingency table, with Cramér's V."""
    tab = pd.crosstab(df[a], df[b])
    res = sps.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    k = min(tab.shape) - 1
    v = np.sqrt(res.statistic / (n * k)) if k else np.nan
    small = (res.expected_freq < 5).mean()
    note = "variables are associated" if res.pvalue < alpha else "no evidence of association"
    if small > 0.2:
        note += f"; {small:.0%} of expected counts < 5 - prefer Fisher's exact test"
    return TestResult("Chi-square independence", res.statistic, res.pvalue, int(n), alpha, v, "Cramér's V", note,
                      {"dof": int(res.dof), "expected_lt5_frac": float(small)})


def fisher_exact(df: pd.DataFrame, a: str, b: str, alpha: float = 0.05) -> TestResult:
    """Fisher's exact test (2x2: odds ratio as statistic; R x C supported by SciPy >= 1.15)."""
    tab = pd.crosstab(df[a], df[b]).to_numpy()
    res = sps.fisher_exact(tab)
    stat = float(res.statistic) if np.ndim(res.statistic) == 0 else float(res[0])
    note = "variables are associated" if res.pvalue < alpha else "no evidence of association"
    return TestResult("Fisher exact", stat, float(res.pvalue), int(tab.sum()), alpha,
                      stat if tab.shape == (2, 2) else None, "odds ratio" if tab.shape == (2, 2) else None, note)


def association(df: pd.DataFrame, a: str, b: str, alpha: float = 0.05) -> list[TestResult]:
    out = [chi2_independence(df, a, b, alpha)]
    try:
        out.append(fisher_exact(df, a, b, alpha))
    except (ValueError, MemoryError) as exc:   # very large R x C tables
        out.append(TestResult("Fisher exact", np.nan, np.nan, len(df), alpha, note=f"not computed: {exc}"))
    return out


# --------------------------------------------------------------- dispatcher
def variable_kind(s: pd.Series, max_levels: int = 20) -> str:
    """'numerical' or 'categorical' (numeric columns with few distinct values count as categorical)."""
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        return "categorical"
    return "categorical" if s.nunique(dropna=True) <= min(max_levels, 5) else "numerical"


def eda_tests(df: pd.DataFrame, x: str, y: str | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """Run the appropriate tests for one variable or a pair, chosen from their kinds."""
    kx = variable_kind(df[x])
    if y is None:
        res = normality(df[x], alpha) if kx == "numerical" else [chi2_goodness_of_fit(df[x].dropna(), alpha=alpha)]
    else:
        ky = variable_kind(df[y])
        if kx == ky == "numerical":
            res = [pearson(df[x], df[y], alpha), spearman(df[x], df[y], alpha)]
        elif kx == ky == "categorical":
            res = association(df, x, y, alpha)
        else:
            cat, num = (x, y) if kx == "categorical" else (y, x)
            res = compare_groups(df, cat, num, alpha)["tests"]
    return results_frame(res)
