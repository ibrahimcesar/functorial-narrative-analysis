"""
Statistical Validation for Cross-Cultural Narrative Analysis.

This module provides statistical tests to validate whether observed differences
between Japanese and Western narrative structures are statistically significant.

Tests included:
    - Chi-square / Fisher exact tests for shape distributions
    - Mann-Whitney U tests for continuous metrics
    - Effect sizes (Cohen's d, Cliff's delta, Cramér's V)
    - Bootstrap confidence intervals
    - Permutation tests for robust comparison
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
from scipy import stats
from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    mannwhitneyu,
    ttest_ind,
    permutation_test,
    bootstrap,
)
from collections import Counter


@dataclass
class EffectSize:
    """Effect size measures with interpretation."""
    name: str
    value: float
    interpretation: str
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": float(self.value) if self.value is not None else None,
            "interpretation": self.interpretation,
            "ci_lower": float(self.ci_lower) if self.ci_lower is not None else None,
            "ci_upper": float(self.ci_upper) if self.ci_upper is not None else None,
        }


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[EffectSize] = None
    significant_at_05: bool = False
    significant_at_01: bool = False
    notes: str = ""

    def __post_init__(self):
        self.significant_at_05 = self.p_value < 0.05
        self.significant_at_01 = self.p_value < 0.01

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": float(self.statistic) if self.statistic is not None else None,
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "effect_size": self.effect_size.to_dict() if self.effect_size else None,
            "significant_at_05": bool(self.significant_at_05),
            "significant_at_01": bool(self.significant_at_01),
            "notes": self.notes,
        }


@dataclass
class ValidationReport:
    """Complete statistical validation report."""
    comparison_name: str
    n_group1: int
    n_group2: int
    group1_name: str
    group2_name: str
    shape_distribution_test: Optional[HypothesisTestResult] = None
    metric_tests: Dict[str, HypothesisTestResult] = field(default_factory=dict)
    bootstrap_cis: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict:
        # Convert bootstrap CIs to JSON-serializable format
        bootstrap_cis_serializable = {}
        for k, v in self.bootstrap_cis.items():
            if isinstance(v, (tuple, list)) and len(v) == 3:
                bootstrap_cis_serializable[k] = [float(x) for x in v]
            elif isinstance(v, (tuple, list)) and len(v) == 2:
                bootstrap_cis_serializable[k] = [float(x) for x in v]
            else:
                bootstrap_cis_serializable[k] = v

        return {
            "comparison_name": self.comparison_name,
            "n_group1": int(self.n_group1),
            "n_group2": int(self.n_group2),
            "group1_name": self.group1_name,
            "group2_name": self.group2_name,
            "shape_distribution_test": self.shape_distribution_test.to_dict() if self.shape_distribution_test else None,
            "metric_tests": {k: v.to_dict() for k, v in self.metric_tests.items()},
            "bootstrap_cis": bootstrap_cis_serializable,
            "summary": self.summary,
        }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> EffectSize:
    """
    Compute Cohen's d effect size for two groups.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        d = 0.0
    else:
        d = (np.mean(group1) - np.mean(group2)) / pooled_std

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return EffectSize(name="Cohen's d", value=d, interpretation=interp)


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> EffectSize:
    """
    Compute Cliff's delta (non-parametric effect size).

    More robust than Cohen's d for non-normal distributions.

    Interpretation:
        |δ| < 0.147: negligible
        0.147 <= |δ| < 0.33: small
        0.33 <= |δ| < 0.474: medium
        |δ| >= 0.474: large
    """
    n1, n2 = len(group1), len(group2)

    # Count dominance pairs
    greater = 0
    less = 0
    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x < y:
                less += 1

    delta = (greater - less) / (n1 * n2)

    # Interpretation (Romano et al., 2006)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interp = "negligible"
    elif abs_delta < 0.33:
        interp = "small"
    elif abs_delta < 0.474:
        interp = "medium"
    else:
        interp = "large"

    return EffectSize(name="Cliff's delta", value=delta, interpretation=interp)


def cramers_v(contingency_table: np.ndarray) -> EffectSize:
    """
    Compute Cramér's V for categorical data.

    Interpretation:
        V < 0.1: negligible
        0.1 <= V < 0.3: small
        0.3 <= V < 0.5: medium
        V >= 0.5: large
    """
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1

    if min_dim == 0 or n == 0:
        v = 0.0
    else:
        v = np.sqrt(chi2 / (n * min_dim))

    if v < 0.1:
        interp = "negligible"
    elif v < 0.3:
        interp = "small"
    elif v < 0.5:
        interp = "medium"
    else:
        interp = "large"

    return EffectSize(name="Cramér's V", value=v, interpretation=interp)


def test_shape_distribution(
    shapes1: List[str],
    shapes2: List[str],
    all_shapes: Optional[List[str]] = None,
) -> HypothesisTestResult:
    """
    Test whether shape distributions differ between two groups.

    Uses chi-square test (if expected frequencies >= 5) or Fisher exact test.
    """
    # Count shapes
    counter1 = Counter(shapes1)
    counter2 = Counter(shapes2)

    # Get all unique shapes
    if all_shapes is None:
        all_shapes = sorted(set(counter1.keys()) | set(counter2.keys()))

    # Build contingency table
    table = np.array([
        [counter1.get(s, 0) for s in all_shapes],
        [counter2.get(s, 0) for s in all_shapes],
    ])

    # Remove columns with all zeros
    non_zero_cols = table.sum(axis=0) > 0
    table = table[:, non_zero_cols]

    if table.shape[1] < 2:
        return HypothesisTestResult(
            test_name="Shape Distribution Test",
            statistic=0.0,
            p_value=1.0,
            notes="Insufficient shape categories for comparison",
        )

    # Check expected frequencies for chi-square validity
    chi2, p_chi2, dof, expected = chi2_contingency(table)
    min_expected = expected.min()

    if min_expected >= 5:
        # Use chi-square
        effect = cramers_v(table)
        return HypothesisTestResult(
            test_name="Chi-square test",
            statistic=chi2,
            p_value=p_chi2,
            effect_size=effect,
            notes=f"dof={dof}, min_expected={min_expected:.1f}",
        )
    else:
        # For small samples, use Fisher exact (only works for 2x2)
        # For larger tables, report chi-square with warning
        if table.shape == (2, 2):
            odds_ratio, p_fisher = fisher_exact(table)
            return HypothesisTestResult(
                test_name="Fisher exact test",
                statistic=odds_ratio,
                p_value=p_fisher,
                notes="2x2 table, exact test used",
            )
        else:
            effect = cramers_v(table)
            return HypothesisTestResult(
                test_name="Chi-square test (small sample warning)",
                statistic=chi2,
                p_value=p_chi2,
                effect_size=effect,
                notes=f"Warning: min_expected={min_expected:.1f} < 5, results may be unreliable",
            )


def test_metric_difference(
    values1: np.ndarray,
    values2: np.ndarray,
    metric_name: str,
    use_parametric: bool = False,
) -> HypothesisTestResult:
    """
    Test whether a continuous metric differs between two groups.

    Uses Mann-Whitney U (non-parametric) by default, or t-test if specified.
    """
    values1 = np.array(values1)
    values2 = np.array(values2)

    # Remove NaN values
    values1 = values1[~np.isnan(values1)]
    values2 = values2[~np.isnan(values2)]

    if len(values1) < 2 or len(values2) < 2:
        return HypothesisTestResult(
            test_name=f"{metric_name} comparison",
            statistic=0.0,
            p_value=1.0,
            notes="Insufficient data",
        )

    if use_parametric:
        # Welch's t-test (doesn't assume equal variances)
        stat, p = ttest_ind(values1, values2, equal_var=False)
        test_name = "Welch's t-test"
    else:
        # Mann-Whitney U (non-parametric)
        stat, p = mannwhitneyu(values1, values2, alternative='two-sided')
        test_name = "Mann-Whitney U test"

    # Compute effect sizes
    d = cohens_d(values1, values2)
    delta = cliffs_delta(values1, values2)

    # Use Cliff's delta as primary (more robust)
    return HypothesisTestResult(
        test_name=f"{test_name} ({metric_name})",
        statistic=stat,
        p_value=p,
        effect_size=delta,
        notes=f"Cohen's d = {d.value:.3f} ({d.interpretation}), "
              f"mean1={np.mean(values1):.3f}, mean2={np.mean(values2):.3f}",
    )


def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    values = np.array(values)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return (np.nan, np.nan, np.nan)

    if statistic == "mean":
        stat_func = np.mean
    elif statistic == "median":
        stat_func = np.median
    elif statistic == "std":
        stat_func = np.std
    else:
        stat_func = np.mean

    point_estimate = stat_func(values)

    # Bootstrap resampling
    bootstrap_stats = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_stats.append(stat_func(sample))

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return (point_estimate, ci_lower, ci_upper)


def bootstrap_difference_ci(
    values1: np.ndarray,
    values2: np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap CI for the difference between two groups.

    Returns:
        (difference, ci_lower, ci_upper)
    """
    values1 = np.array(values1)[~np.isnan(values1)]
    values2 = np.array(values2)[~np.isnan(values2)]

    if len(values1) < 2 or len(values2) < 2:
        return (np.nan, np.nan, np.nan)

    if statistic == "mean":
        stat_func = np.mean
    elif statistic == "median":
        stat_func = np.median
    else:
        stat_func = np.mean

    observed_diff = stat_func(values1) - stat_func(values2)

    # Bootstrap resampling
    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample1 = rng.choice(values1, size=len(values1), replace=True)
        sample2 = rng.choice(values2, size=len(values2), replace=True)
        diffs.append(stat_func(sample1) - stat_func(sample2))

    alpha = 1 - confidence
    ci_lower = np.percentile(diffs, 100 * alpha / 2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    return (observed_diff, ci_lower, ci_upper)


def permutation_test_difference(
    values1: np.ndarray,
    values2: np.ndarray,
    n_permutations: int = 10000,
) -> Tuple[float, float]:
    """
    Permutation test for difference in means.

    Returns:
        (observed_difference, p_value)
    """
    values1 = np.array(values1)[~np.isnan(values1)]
    values2 = np.array(values2)[~np.isnan(values2)]

    observed_diff = np.mean(values1) - np.mean(values2)
    combined = np.concatenate([values1, values2])
    n1 = len(values1)

    count_extreme = 0
    rng = np.random.default_rng(42)

    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)
    return observed_diff, p_value


def run_validation(
    group1_data: List[Dict],
    group2_data: List[Dict],
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    metrics: Optional[List[str]] = None,
) -> ValidationReport:
    """
    Run complete statistical validation comparing two groups.

    Args:
        group1_data: List of dicts with 'pattern_type' and metric values
        group2_data: List of dicts with 'pattern_type' and metric values
        group1_name: Name for first group
        group2_name: Name for second group
        metrics: List of metric names to compare

    Returns:
        ValidationReport with all test results
    """
    if metrics is None:
        metrics = [
            "conformance_score",
            "ki_sho_smoothness",
            "ten_strength",
            "ketsu_compression",
            "ten_position",
        ]

    report = ValidationReport(
        comparison_name=f"{group1_name} vs {group2_name}",
        n_group1=len(group1_data),
        n_group2=len(group2_data),
        group1_name=group1_name,
        group2_name=group2_name,
    )

    # 1. Test shape distribution
    shapes1 = [d.get("pattern_type", "unknown") for d in group1_data]
    shapes2 = [d.get("pattern_type", "unknown") for d in group2_data]
    report.shape_distribution_test = test_shape_distribution(shapes1, shapes2)

    # 2. Test each metric
    for metric in metrics:
        values1 = np.array([d.get(metric, np.nan) for d in group1_data])
        values2 = np.array([d.get(metric, np.nan) for d in group2_data])

        # Skip if all NaN
        if np.all(np.isnan(values1)) or np.all(np.isnan(values2)):
            continue

        test_result = test_metric_difference(values1, values2, metric)
        report.metric_tests[metric] = test_result

        # Bootstrap CI for the difference
        diff, ci_lo, ci_hi = bootstrap_difference_ci(values1, values2)
        report.bootstrap_cis[f"{metric}_difference"] = (diff, ci_lo, ci_hi)

    # 3. Generate summary
    sig_tests = [
        name for name, test in report.metric_tests.items()
        if test.significant_at_05
    ]

    large_effects = [
        name for name, test in report.metric_tests.items()
        if test.effect_size and test.effect_size.interpretation in ("medium", "large")
    ]

    summary_lines = [
        f"Comparison: {group1_name} (n={report.n_group1}) vs {group2_name} (n={report.n_group2})",
        "",
        f"Shape distribution: p = {report.shape_distribution_test.p_value:.4f}"
        if report.shape_distribution_test else "Shape distribution: not tested",
    ]

    if sig_tests:
        summary_lines.append(f"Significant differences (p < 0.05): {', '.join(sig_tests)}")
    else:
        summary_lines.append("No significant differences at p < 0.05")

    if large_effects:
        summary_lines.append(f"Medium/large effect sizes: {', '.join(large_effects)}")

    report.summary = "\n".join(summary_lines)

    return report


def validate_cross_cultural(
    japanese_results: List[Dict],
    western_results: List[Dict],
    output_path: Optional[Path] = None,
) -> ValidationReport:
    """
    Run cross-cultural validation between Japanese and Western corpora.

    Args:
        japanese_results: Kishōtenketsu match results for Japanese works
        western_results: Kishōtenketsu match results for Western works
        output_path: Optional path to save JSON report

    Returns:
        ValidationReport
    """
    report = run_validation(
        japanese_results,
        western_results,
        group1_name="Japanese (Aozora)",
        group2_name="Western (Gutenberg)",
        metrics=[
            "conformance_score",
            "ki_sho_smoothness",
            "ten_strength",
            "ketsu_compression",
            "ten_position",
        ],
    )

    # Add kishōtenketsu rate comparison
    jp_k_count = sum(1 for d in japanese_results if "kishotenketsu" in d.get("pattern_type", ""))
    west_k_count = sum(1 for d in western_results if "kishotenketsu" in d.get("pattern_type", ""))
    jp_k_rate = jp_k_count / len(japanese_results) if japanese_results else 0
    west_k_rate = west_k_count / len(western_results) if western_results else 0

    # Test proportion difference using Fisher exact test (more robust for small samples)
    from scipy.stats import fisher_exact
    contingency = np.array([
        [jp_k_count, len(japanese_results) - jp_k_count],
        [west_k_count, len(western_results) - west_k_count],
    ])

    if contingency.sum() > 0:
        odds_ratio, p_prop = fisher_exact(contingency)
        report.metric_tests["kishotenketsu_rate"] = HypothesisTestResult(
            test_name="Fisher exact test (kishōtenketsu rate)",
            statistic=odds_ratio,
            p_value=p_prop,
            notes=f"Japanese: {jp_k_rate:.1%} ({jp_k_count}/{len(japanese_results)}), Western: {west_k_rate:.1%} ({west_k_count}/{len(western_results)})",
        )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    return report


def print_report(report: ValidationReport) -> None:
    """Print a formatted validation report."""
    print("=" * 70)
    print(f"STATISTICAL VALIDATION REPORT")
    print("=" * 70)
    print(f"\n{report.comparison_name}")
    print(f"  {report.group1_name}: n = {report.n_group1}")
    print(f"  {report.group2_name}: n = {report.n_group2}")

    print("\n" + "-" * 70)
    print("SHAPE DISTRIBUTION TEST")
    print("-" * 70)
    if report.shape_distribution_test:
        test = report.shape_distribution_test
        sig = "***" if test.significant_at_01 else ("**" if test.significant_at_05 else "")
        print(f"  {test.test_name}")
        print(f"  Statistic: {test.statistic:.4f}")
        print(f"  p-value: {test.p_value:.4f} {sig}")
        if test.effect_size:
            print(f"  Effect size: {test.effect_size.name} = {test.effect_size.value:.4f} ({test.effect_size.interpretation})")
        if test.notes:
            print(f"  Notes: {test.notes}")

    print("\n" + "-" * 70)
    print("METRIC COMPARISONS")
    print("-" * 70)
    for metric, test in report.metric_tests.items():
        sig = "***" if test.significant_at_01 else ("**" if test.significant_at_05 else "")
        print(f"\n  {metric}:")
        print(f"    {test.test_name}")
        print(f"    p-value: {test.p_value:.4f} {sig}")
        if test.effect_size:
            print(f"    Effect: {test.effect_size.name} = {test.effect_size.value:.4f} ({test.effect_size.interpretation})")
        if test.notes:
            print(f"    Notes: {test.notes}")

        # Bootstrap CI
        ci_key = f"{metric}_difference"
        if ci_key in report.bootstrap_cis:
            diff, lo, hi = report.bootstrap_cis[ci_key]
            print(f"    95% CI for difference: [{lo:.4f}, {hi:.4f}]")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(report.summary)
    print("\n  *** p < 0.01, ** p < 0.05")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from geometry.surprisal import SurprisalExtractor
    from detectors.kishotenketsu import KishotenketsuDetector

    print("Running cross-cultural statistical validation...")

    extractor = SurprisalExtractor(method='entropy', window_size=200)
    detector = KishotenketsuDetector()

    # Load Japanese works
    jp_results = []
    aozora_dir = Path("data/raw/aozora_extended/texts")
    if aozora_dir.exists():
        for f in list(aozora_dir.glob("*.json"))[:20]:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                text = data.get('text', '')
                if len(text) < 3000:
                    continue
                trajectory = extractor.extract(text[:80000])
                match = detector.detect(trajectory.values, f.stem, data.get('title', f.stem))
                jp_results.append(match.to_dict())
            except Exception:
                pass

    # Load Western works
    west_results = []
    gutenberg_dir = Path("data/raw/gutenberg_extended/texts")
    if gutenberg_dir.exists():
        for f in list(gutenberg_dir.glob("*.json"))[:20]:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                text = data.get('text', '')
                if len(text) < 3000:
                    continue
                trajectory = extractor.extract(text[:80000])
                match = detector.detect(trajectory.values, f.stem, data.get('title', f.stem))
                west_results.append(match.to_dict())
            except Exception:
                pass

    # Run validation
    report = validate_cross_cultural(
        jp_results,
        west_results,
        output_path=Path("data/results/statistical_validation.json"),
    )

    print_report(report)
