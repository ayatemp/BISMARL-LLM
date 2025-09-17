
import os, math, json, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None
try:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
except Exception:
    r2_score = None
    mean_squared_error = None
    mean_absolute_error = None
import matplotlib.pyplot as plt


@dataclass
class IRMMetrics:
    n: int
    mae: float
    mse: float
    rmse: float
    r2: Optional[float]
    pearson_r: Optional[float]
    pearson_p: Optional[float]
    spearman_rho: Optional[float]
    spearman_p: Optional[float]
    calib_slope: float
    calib_intercept: float
    ece: float  # Expected Calibration Error for regression (binning-based)


def _safe_numpy(x) -> np.ndarray:
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float64)
    if hasattr(x, "cpu") and hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _linreg(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return slope, intercept of y_true ~ a*y_pred + b (ordinary least squares)."""
    x = np.c_[y_pred, np.ones_like(y_pred)]
    try:
        coef = np.linalg.pinv(x.T @ x) @ (x.T @ y_true)
        slope, intercept = float(coef[0]), float(coef[1])
    except Exception:
        slope, intercept = math.nan, math.nan
    return slope, intercept


def _ece_regression(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 20) -> float:
    """Regression ECE: bin by prediction, compare bin mean prediction vs bin mean target."""
    order = np.argsort(y_pred)
    y_true = y_true[order]
    y_pred = y_pred[order]
    bins = np.array_split(np.arange(len(y_pred)), n_bins)
    ece = 0.0
    total = len(y_pred)
    for idx in bins:
        if len(idx) == 0:
            continue
        bin_pred = y_pred[idx].mean()
        bin_true = y_true[idx].mean()
        w = len(idx) / total
        ece += w * abs(bin_true - bin_pred)
    return float(ece)


def summarize_metrics(y_true, y_pred, assume_range: Optional[Tuple[float, float]] = None) -> IRMMetrics:
    """
    Compute key regression metrics + simple calibration stats.
    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and model predictions. Shape (N,).
    assume_range : (low, high) or None
        If your IRM outputs are theoretically bounded (e.g., 0..1),
        pass (0,1) to clip predictions into range before scoring.
    """
    yt = _safe_numpy(y_true).reshape(-1)
    yp = _safe_numpy(y_pred).reshape(-1)
    assert yt.size == yp.size, "y_true and y_pred must have the same length"

    if assume_range is not None:
        low, high = assume_range
        yp = np.clip(yp, low, high)

    # Base metrics
    n = yt.size
    if mean_absolute_error is not None:
        mae = float(mean_absolute_error(yt, yp))
        mse = float(mean_squared_error(yt, yp))
        rmse = float(math.sqrt(mse))
        r2 = float(r2_score(yt, yp))
    else:
        diff = yt - yp
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        rmse = float(math.sqrt(mse))
        ss_res = np.sum(diff ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else math.nan

    # Correlations
    if pearsonr is not None:
        pr, pp = pearsonr(yt, yp)
        sr, sp = spearmanr(yt, yp)
        pearson_r, pearson_p = float(pr), float(pp)
        spearman_rho, spearman_p = float(sr), float(sp)
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = math.nan

    slope, intercept = _linreg(yt, yp)
    ece = _ece_regression(yt, yp, n_bins=20)

    return IRMMetrics(
        n=n,
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        calib_slope=slope,
        calib_intercept=intercept,
        ece=ece,
    )


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Basic bootstrap percentile CI for a 1D array of scalar values (e.g., per-sample losses)."""
    rng = np.random.default_rng(seed)
    vals = _safe_numpy(values).reshape(-1)
    boots = []
    n = len(vals)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(np.mean(vals[idx]))
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def save_metrics_json(metrics: IRMMetrics, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics.__dict__, f, indent=2, ensure_ascii=False)


def plot_pred_vs_true(y_true, y_pred, out_path: str, title: str = "IRM: Predicted vs True"):
    yt = _safe_numpy(y_true).reshape(-1)
    yp = _safe_numpy(y_pred).reshape(-1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.scatter(yt, yp, s=6, alpha=0.5)
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_residuals(y_true, y_pred, out_path: str, title: str = "IRM: Residuals (Pred - True)"):
    yt = _safe_numpy(y_true).reshape(-1)
    yp = _safe_numpy(y_pred).reshape(-1)
    res = yp - yt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.scatter(yp, res, s=6, alpha=0.5)
    plt.axhline(0.0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Pred - True)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_calibration_curve(y_true, y_pred, out_path: str, n_bins: int = 20, title: str = "IRM: Calibration by Prediction Bins"):
    yt = _safe_numpy(y_true).reshape(-1)
    yp = _safe_numpy(y_pred).reshape(-1)
    order = np.argsort(yp)
    yt = yt[order]
    yp = yp[order]
    bins = np.array_split(np.arange(len(yp)), n_bins)

    xs, ys = [], []
    for idx in bins:
        if len(idx) == 0:
            continue
        xs.append(yp[idx].mean())
        ys.append(yt[idx].mean())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.scatter(xs, ys)
    lo = float(min(min(xs), min(ys)))
    hi = float(max(max(xs), max(ys)))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Mean Pred (per bin)")
    plt.ylabel("Mean True (per bin)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_hist(y_true, y_pred, out_path: str, title: str = "IRM: Absolute Error Histogram", bins: int = 40):
    yt = _safe_numpy(y_true).reshape(-1)
    yp = _safe_numpy(y_pred).reshape(-1)
    ae = np.abs(yt - yp)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.hist(ae, bins=bins)
    plt.xlabel("|Pred - True|")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bland_altman(y_true, y_pred, out_path: str, title: str = "IRM: Bland–Altman Plot"):
    yt = _safe_numpy(y_true).reshape(-1)
    yp = _safe_numpy(y_pred).reshape(-1)
    avg = (yt + yp) / 2.0
    diff = yp - yt
    md = np.mean(diff)
    sd = np.std(diff)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.scatter(avg, diff, s=6, alpha=0.5)
    plt.axhline(md)
    plt.axhline(md + 1.96 * sd)
    plt.axhline(md - 1.96 * sd)
    plt.xlabel("Mean of True & Pred")
    plt.ylabel("Pred - True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_predictions_csv(y_true, y_pred, out_path: str):
    yt = _safe_numpy(y_true).reshape(-1)
    yp = _safe_numpy(y_pred).reshape(-1)
    df = pd.DataFrame(
        {
            "y_true": yt,
            "y_pred": yp,
            "error": yp - yt,
            "abs_error": np.abs(yp - yt),
        }
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def make_markdown_report(y_true, y_pred, out_dir: str, prefix: str = "irm_eval", assume_range: Optional[Tuple[float, float]] = None) -> str:
    """
    End-to-end: compute metrics, save figures, and write a simple Markdown report.
    Returns the path to the generated markdown file.
    """
    os.makedirs(out_dir, exist_ok=True)

    metrics = summarize_metrics(y_true, y_pred, assume_range=assume_range)
    metrics_path = os.path.join(out_dir, f"{prefix}_metrics.json")
    save_metrics_json(metrics, metrics_path)

    # Figures
    fig_pred_true = os.path.join(out_dir, f"{prefix}_pred_vs_true.png")
    fig_resid = os.path.join(out_dir, f"{prefix}_residuals.png")
    fig_calib = os.path.join(out_dir, f"{prefix}_calibration.png")
    fig_hist = os.path.join(out_dir, f"{prefix}_abs_error_hist.png")
    fig_ba = os.path.join(out_dir, f"{prefix}_bland_altman.png")

    plot_pred_vs_true(y_true, y_pred, fig_pred_true)
    plot_residuals(y_true, y_pred, fig_resid)
    plot_calibration_curve(y_true, y_pred, fig_calib, n_bins=20)
    plot_error_hist(y_true, y_pred, fig_hist)
    plot_bland_altman(y_true, y_pred, fig_ba)

    # CSV
    csv_path = os.path.join(out_dir, f"{prefix}_predictions.csv")
    export_predictions_csv(y_true, y_pred, csv_path)

    # Bootstrap CI for MAE (example)
    ae = np.abs(_safe_numpy(y_true).reshape(-1) - _safe_numpy(y_pred).reshape(-1))
    lo, hi = bootstrap_ci(ae)

    md_path = os.path.join(out_dir, f"{prefix}_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# IRM Evaluation Report\n\n")
        f.write("## Summary Metrics\n\n")
        f.write("| metric | value |\n|---|---:|\n")
        f.write(f"| n | {metrics.n} |\n")
        f.write(f"| MAE | {metrics.mae:.4f} |\n")
        f.write(f"| MSE | {metrics.mse:.4f} |\n")
        f.write(f"| RMSE | {metrics.rmse:.4f} |\n")
        f.write(f"| R² | {metrics.r2:.4f} |\n")
        f.write(f"| Pearson r (p) | {metrics.pearson_r:.4f} ({metrics.pearson_p:.2e}) |\n")
        f.write(f"| Spearman ρ (p) | {metrics.spearman_rho:.4f} ({metrics.spearman_p:.2e}) |\n")
        f.write(f"| Calibration slope | {metrics.calib_slope:.4f} |\n")
        f.write(f"| Calibration intercept | {metrics.calib_intercept:.4f} |\n")
        f.write(f"| ECE (regression) | {metrics.ece:.4f} |\n")
        f.write(f"\n**MAE bootstrap 95% CI (mean abs error):** [{lo:.4f}, {hi:.4f}]\n\n")

        f.write("## Figures\n\n")
        f.write(f"![Pred vs True]({os.path.basename(fig_pred_true)})\n\n")
        f.write(f"![Residuals]({os.path.basename(fig_resid)})\n\n")
        f.write(f"![Calibration]({os.path.basename(fig_calib)})\n\n")
        f.write(f"![Abs Error Hist]({os.path.basename(fig_hist)})\n\n")
        f.write(f"![Bland–Altman]({os.path.basename(fig_ba)})\n\n")

        f.write("## Artifacts\n\n")
        f.write(f"- Metrics JSON: {os.path.basename(metrics_path)}\n")
        f.write(f"- Predictions CSV: {os.path.basename(csv_path)}\n")

    return md_path
