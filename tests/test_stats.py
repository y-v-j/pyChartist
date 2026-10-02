"""EDA statistics checked against independent references (statsmodels / closed-form formulas)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from pychartist import stats


def test_pearson_spearman_against_numpy(penguins):
    d = penguins.dropna(subset=["flipper_length_mm", "body_mass_g"])
    r = stats.pearson(d.flipper_length_mm, d.body_mass_g)
    assert r.statistic == pytest.approx(np.corrcoef(d.flipper_length_mm, d.body_mass_g)[0, 1])
    rho = stats.spearman(d.flipper_length_mm, d.body_mass_g)
    ranks = d[["flipper_length_mm", "body_mass_g"]].rank()
    assert rho.statistic == pytest.approx(np.corrcoef(ranks.T)[0, 1])


def test_welch_t_against_statsmodels(penguins):
    from statsmodels.stats.weightstats import ttest_ind
    a = penguins.query("species=='Adelie'").body_mass_g.dropna()
    b = penguins.query("species=='Chinstrap'").body_mass_g.dropna()
    ours = stats.t_test(a, b)
    t, p, dof = ttest_ind(a, b, usevar="unequal")
    assert ours.statistic == pytest.approx(t)
    assert ours.pvalue == pytest.approx(p)
    assert ours.extra["dof"] == pytest.approx(dof)


def test_anova_against_statsmodels(penguins):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = penguins.dropna(subset=["body_mass_g"])
    ours = stats.anova(*[g.body_mass_g for _, g in d.groupby("species")])
    table = sm.stats.anova_lm(smf.ols("body_mass_g ~ C(species)", data=d).fit())
    assert ours.statistic == pytest.approx(table["F"].iloc[0])
    ss = table["sum_sq"]
    assert ours.effect_size == pytest.approx(ss.iloc[0] / ss.sum())   # eta squared


def test_chi2_gof_closed_form():
    obs = pd.Series({"a": 30, "b": 50, "c": 20})
    res = stats.chi2_goodness_of_fit(counts=obs)
    exp = 100 / 3
    assert res.statistic == pytest.approx(sum((o - exp) ** 2 / exp for o in obs))
    assert res.pvalue == pytest.approx(sps.chi2.sf(res.statistic, 2))


def test_chi2_independence_and_cramers_v(titanic):
    res = stats.chi2_independence(titanic, "class", "alive")
    tab = pd.crosstab(titanic["class"], titanic["alive"]).to_numpy()
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    chi2 = ((tab - exp) ** 2 / exp).sum()
    assert res.statistic == pytest.approx(chi2)
    assert res.effect_size == pytest.approx(np.sqrt(chi2 / (tab.sum() * 1)))
    from statsmodels.stats.contingency_tables import Table
    assert res.statistic == pytest.approx(Table(tab).test_nominal_association().statistic)


def test_fisher_2x2_odds_ratio(titanic):
    res = stats.fisher_exact(titanic, "sex", "alive")
    t = pd.crosstab(titanic.sex, titanic.alive).to_numpy()
    assert res.statistic == pytest.approx(t[0, 0] * t[1, 1] / (t[0, 1] * t[1, 0]))


def test_mann_whitney_rank_biserial():
    a, b = np.arange(10.0), np.arange(5.0, 15.0)
    res = stats.mann_whitney(a, b)
    # U counts pairs where a > b (+0.5 ties); rank-biserial = 1 - 2U/(n1 n2)
    U = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    assert res.statistic == pytest.approx(U)
    assert res.effect_size == pytest.approx(1 - 2 * U / 100)


def test_normality_tests_detect_skew():
    rng = np.random.default_rng(0)
    normal = stats.normality(rng.normal(size=400))
    skewed = stats.normality(rng.exponential(size=400))
    assert all(t.pvalue > 0.01 for t in normal)
    assert all(t.pvalue < 1e-4 for t in skewed)


def test_compare_groups_picks_tests(penguins):
    two = stats.compare_groups(penguins[penguins.species != "Gentoo"], "species", "bill_length_mm")
    assert [t.test for t in two["tests"]] == ["Welch t-test", "Mann-Whitney U"]
    three = stats.compare_groups(penguins, "species", "bill_length_mm")
    assert [t.test for t in three["tests"]] == ["One-way ANOVA", "Kruskal-Wallis H"]


def test_eda_dispatch(penguins):
    assert list(stats.eda_tests(penguins, "body_mass_g").test) == ["Shapiro-Wilk", "Kolmogorov-Smirnov"]
    assert list(stats.eda_tests(penguins, "species").test) == ["Chi-square goodness of fit"]
    assert list(stats.eda_tests(penguins, "species", "island").test) == ["Chi-square independence", "Fisher exact"]
    assert stats.variable_kind(penguins.species) == "categorical"
    assert stats.variable_kind(penguins.body_mass_g) == "numerical"
