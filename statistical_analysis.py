"""
Statistical Analysis Module for IEEE TMI Publication
====================================================

Provides:
1. DeLong test for AUC comparison
2. McNemar's test for accuracy comparison
3. Calibration metrics (ECE, reliability diagram)
4. Bootstrap confidence intervals
5. Multiple comparison correction (Bonferroni)

Usage:
    from statistical_analysis import (
        delong_test, mcnemar_test, calibration_metrics,
        bootstrap_ci, compare_all_models
    )
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import warnings


# ============================================================
# 1. DeLong Test for AUC Comparison
# ============================================================

def _compute_midrank(x):
    """Compute midranks for DeLong test."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong covariance computation."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r] = _compute_midrank(positive_examples[r])
        ty[r] = _compute_midrank(negative_examples[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    return aucs, delongcov


def delong_test(y_true: np.ndarray,
                pred1: np.ndarray,
                pred2: np.ndarray) -> Tuple[float, float, float, float]:
    """
    DeLong test for comparing two ROC AUC scores.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    pred1 : array-like
        Predicted probabilities from model 1
    pred2 : array-like
        Predicted probabilities from model 2

    Returns
    -------
    auc1 : float
        AUC of model 1
    auc2 : float
        AUC of model 2
    z_stat : float
        Z-statistic
    p_value : float
        Two-sided p-value
    """
    y_true = np.asarray(y_true)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)

    # Sort by labels
    order = np.argsort(y_true)[::-1]  # Positive first
    y_sorted = y_true[order]
    pred1_sorted = pred1[order]
    pred2_sorted = pred2[order]

    label_1_count = int(y_sorted.sum())
    predictions = np.vstack([pred1_sorted, pred2_sorted])

    aucs, cov = _fast_delong(predictions, label_1_count)

    auc1, auc2 = aucs[0], aucs[1]

    # Z-statistic
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return auc1, auc2, 0.0, 1.0

    z = (auc1 - auc2) / np.sqrt(var)
    p_value = 2 * stats.norm.sf(abs(z))

    return auc1, auc2, z, p_value


# ============================================================
# 2. McNemar's Test for Accuracy Comparison
# ============================================================

def mcnemar_test(y_true: np.ndarray,
                 pred1: np.ndarray,
                 pred2: np.ndarray,
                 threshold: float = 0.5) -> Tuple[float, float]:
    """
    McNemar's test for comparing two classifiers.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    pred1 : array-like
        Predicted probabilities from model 1
    pred2 : array-like
        Predicted probabilities from model 2
    threshold : float
        Classification threshold

    Returns
    -------
    chi2 : float
        Chi-square statistic
    p_value : float
        p-value
    """
    y_true = np.asarray(y_true)
    pred1_binary = (np.asarray(pred1) >= threshold).astype(int)
    pred2_binary = (np.asarray(pred2) >= threshold).astype(int)

    correct1 = (pred1_binary == y_true)
    correct2 = (pred2_binary == y_true)

    # Contingency table
    # b = model1 correct, model2 wrong
    # c = model1 wrong, model2 correct
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)

    if b + c == 0:
        return 0.0, 1.0

    # McNemar's test with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value


# ============================================================
# 3. Calibration Metrics
# ============================================================

def calibration_metrics(y_true: np.ndarray,
                       y_prob: np.ndarray,
                       n_bins: int = 10) -> Dict:
    """
    Compute calibration metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration curve

    Returns
    -------
    dict with:
        - ece: Expected Calibration Error
        - mce: Maximum Calibration Error
        - brier: Brier score
        - prob_true: Fraction of positives per bin
        - prob_pred: Mean predicted probability per bin
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # ECE: weighted average of calibration error
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    bin_weights = bin_counts / len(y_prob)

    # Align weights with calibration curve bins
    non_empty_bins = bin_counts > 0
    non_empty_weights = bin_weights[non_empty_bins]

    if len(prob_true) != len(non_empty_weights):
        # Fallback: simple average
        ece = np.mean(np.abs(prob_true - prob_pred))
    else:
        ece = np.sum(non_empty_weights * np.abs(prob_true - prob_pred))

    # MCE: maximum calibration error
    mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0

    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'prob_true': prob_true,
        'prob_pred': prob_pred
    }


def plot_reliability_diagram(calibration_results: Dict[str, Dict],
                            save_path: Optional[str] = None):
    """
    Plot reliability diagram for multiple models.

    Parameters
    ----------
    calibration_results : dict
        {model_name: calibration_metrics output}
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    colors = plt.cm.tab10.colors
    for i, (name, results) in enumerate(calibration_results.items()):
        prob_true = results['prob_true']
        prob_pred = results['prob_pred']
        ece = results['ece']

        ax.plot(prob_pred, prob_true, 's-', color=colors[i % len(colors)],
                label=f'{name} (ECE={ece:.3f})')

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# 4. Bootstrap Confidence Intervals
# ============================================================

def bootstrap_ci(y_true: np.ndarray,
                y_prob: np.ndarray,
                metric_fn,
                n_bootstrap: int = 1000,
                ci: float = 0.95,
                random_state: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    metric_fn : callable
        Function that takes (y_true, y_prob) and returns a scalar
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int
        Random seed

    Returns
    -------
    point_estimate : float
        Metric on original data
    ci_lower : float
        Lower bound of CI
    ci_upper : float
        Upper bound of CI
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    rng = np.random.RandomState(random_state)

    # Point estimate
    point_estimate = metric_fn(y_true, y_prob)

    # Bootstrap samples
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            bootstrap_scores.append(score)
        except:
            continue

    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

    return point_estimate, ci_lower, ci_upper


# ============================================================
# 5. Compare All Models
# ============================================================

def compare_all_models(y_true: np.ndarray,
                      predictions: Dict[str, np.ndarray],
                      baseline_model: str = None) -> pd.DataFrame:
    """
    Compare all models against each other (or a baseline).

    Parameters
    ----------
    y_true : array-like
        True labels
    predictions : dict
        {model_name: predicted_probabilities}
    baseline_model : str, optional
        Name of baseline model. If None, compares all pairs.

    Returns
    -------
    DataFrame with comparison results
    """
    y_true = np.asarray(y_true)
    model_names = list(predictions.keys())

    results = []

    if baseline_model:
        # Compare each model to baseline
        baseline_pred = predictions[baseline_model]
        for name in model_names:
            if name == baseline_model:
                continue

            auc1, auc2, z, p = delong_test(y_true,
                                          predictions[name],
                                          baseline_pred)
            chi2, p_mcnemar = mcnemar_test(y_true,
                                           predictions[name],
                                           baseline_pred)

            results.append({
                'Model 1': name,
                'Model 2': baseline_model,
                'AUC 1': auc1,
                'AUC 2': auc2,
                'DeLong Z': z,
                'DeLong p': p,
                'McNemar chi2': chi2,
                'McNemar p': p_mcnemar,
            })
    else:
        # Compare all pairs
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                auc1, auc2, z, p = delong_test(y_true,
                                              predictions[name1],
                                              predictions[name2])
                chi2, p_mcnemar = mcnemar_test(y_true,
                                               predictions[name1],
                                               predictions[name2])

                results.append({
                    'Model 1': name1,
                    'Model 2': name2,
                    'AUC 1': auc1,
                    'AUC 2': auc2,
                    'DeLong Z': z,
                    'DeLong p': p,
                    'McNemar chi2': chi2,
                    'McNemar p': p_mcnemar,
                })

    df = pd.DataFrame(results)

    # Apply Bonferroni correction
    n_comparisons = len(df)
    df['DeLong p (Bonf.)'] = np.minimum(df['DeLong p'] * n_comparisons, 1.0)
    df['McNemar p (Bonf.)'] = np.minimum(df['McNemar p'] * n_comparisons, 1.0)
    df['Significant (0.05)'] = df['DeLong p (Bonf.)'] < 0.05

    return df


# ============================================================
# 6. Generate Publication-Ready Results Table
# ============================================================

def generate_results_table(y_true: np.ndarray,
                          predictions: Dict[str, np.ndarray],
                          model_order: List[str] = None) -> pd.DataFrame:
    """
    Generate publication-ready results table with all metrics.

    Parameters
    ----------
    y_true : array-like
        True labels
    predictions : dict
        {model_name: predicted_probabilities}
    model_order : list, optional
        Order of models in output table

    Returns
    -------
    DataFrame with metrics for each model
    """
    y_true = np.asarray(y_true)

    if model_order is None:
        model_order = list(predictions.keys())

    results = []

    for name in model_order:
        if name not in predictions:
            continue

        y_prob = predictions[name]
        y_pred = (y_prob >= 0.5).astype(int)

        # AUC with 95% CI
        auc, auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score)

        # Accuracy with 95% CI
        acc, acc_lo, acc_hi = bootstrap_ci(
            y_true, y_pred,
            lambda y, p: accuracy_score(y, p)
        )

        # Calibration
        cal = calibration_metrics(y_true, y_prob)

        # Sensitivity/Specificity
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results.append({
            'Model': name,
            'AUC': f"{auc:.3f} ({auc_lo:.3f}-{auc_hi:.3f})",
            'Accuracy': f"{acc:.3f} ({acc_lo:.3f}-{acc_hi:.3f})",
            'Sensitivity': f"{sensitivity:.3f}",
            'Specificity': f"{specificity:.3f}",
            'ECE': f"{cal['ece']:.3f}",
            'Brier': f"{cal['brier']:.3f}",
        })

    return pd.DataFrame(results)


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.4, n)

    # Simulated model predictions
    pred_model1 = np.clip(y_true * 0.7 + np.random.normal(0, 0.2, n), 0, 1)
    pred_model2 = np.clip(y_true * 0.6 + np.random.normal(0, 0.25, n), 0, 1)
    pred_model3 = np.clip(y_true * 0.8 + np.random.normal(0, 0.15, n), 0, 1)

    predictions = {
        'Image Only': pred_model1,
        'EHR Only': pred_model2,
        'Fusion': pred_model3,
    }

    print("=" * 70)
    print("STATISTICAL ANALYSIS EXAMPLE")
    print("=" * 70)

    # 1. Results table
    print("\n1. RESULTS TABLE:")
    results_df = generate_results_table(y_true, predictions)
    print(results_df.to_string(index=False))

    # 2. Model comparisons
    print("\n2. PAIRWISE COMPARISONS (with Bonferroni correction):")
    comparison_df = compare_all_models(y_true, predictions)
    print(comparison_df.to_string(index=False))

    # 3. Calibration
    print("\n3. CALIBRATION METRICS:")
    cal_results = {}
    for name, pred in predictions.items():
        cal = calibration_metrics(y_true, pred)
        cal_results[name] = cal
        print(f"  {name}: ECE={cal['ece']:.4f}, MCE={cal['mce']:.4f}, Brier={cal['brier']:.4f}")

    # 4. Plot reliability diagram
    print("\n4. Plotting reliability diagram...")
    plot_reliability_diagram(cal_results)

    print("\n" + "=" * 70)
    print("USE THIS MODULE IN YOUR CLASSIFICATION NOTEBOOK")
    print("=" * 70)
