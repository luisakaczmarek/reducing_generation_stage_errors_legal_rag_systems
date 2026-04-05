"""
evaluator.py — Load condition CSVs, compute accuracy, McNemar's test, per-subject
breakdown, and save results/summary.csv.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist

CONDITION_NAMES = {
    0: "Baseline",
    1: "Grounding",
    2: "Rule Extraction",
    3: "Chain of Logic",
    4: "Negative Elimination",
    5: "Answer Verification",
    6: "Self-Consistency (N=3)",
    7: "Rule Extraction + CoL",
}


def compute_ece(df: pd.DataFrame, n_bins: int = 10) -> tuple:
    """Compute Expected Calibration Error (Guo et al. 2017)."""
    df = df.copy()
    df = df.dropna(subset=["confidence", "is_correct"])
    df["confidence"] = df["confidence"].clip(0.0, 1.0)
    df["is_correct_int"] = df["is_correct"].astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin_idx"] = pd.cut(
        df["confidence"], bins=bins, labels=False, include_lowest=True
    )
    ece = 0.0
    n = len(df)
    bin_stats = []
    for b in range(n_bins):
        group = df[df["bin_idx"] == b]
        if len(group) == 0:
            continue
        avg_conf = group["confidence"].mean()
        avg_acc = group["is_correct_int"].mean()
        weight = len(group) / n
        gap = abs(avg_conf - avg_acc)
        ece += weight * gap
        bin_stats.append(
            {
                "bin_lower": round(bins[b], 2),
                "bin_upper": round(bins[b + 1], 2),
                "n": len(group),
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
                "gap": round(gap, 4),
                "weight": round(weight, 4),
                "overconfident": avg_conf > avg_acc,
            }
        )
    return round(ece, 4), pd.DataFrame(bin_stats)


def plot_reliability_diagram(bin_stats_df, condition_name, output_path):
    """Save a reliability diagram for one condition."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)
    if len(bin_stats_df) > 0:
        x = bin_stats_df["avg_confidence"].values
        y = bin_stats_df["avg_accuracy"].values
        sizes = (bin_stats_df["n"].values / bin_stats_df["n"].max()) * 300 + 30
        ax.scatter(x, y, s=sizes, alpha=0.7, color="steelblue", zorder=3)
        ax.fill_between(
            [0, 1], [0, 1], [1, 1],
            alpha=0.05, color="red", label="Overconfident region",
        )
    ax.set_xlabel("Mean Confidence", fontsize=12)
    ax.set_ylabel("Mean Accuracy", fontsize=12)
    ax.set_title(f"Reliability Diagram\n{condition_name}", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _mcnemar(y_true, pred_a, pred_b):
    """McNemar's test: A vs B, returns (chi2, p)."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int((correct_a & ~correct_b).sum())   # A right, B wrong
    c = int((~correct_a & correct_b).sum())   # A wrong, B right
    if b + c == 0:
        return float("nan"), float("nan")
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1.0 - chi2_dist.cdf(chi2, df=1)
    return chi2, p


def run_evaluator(results_dir="results"):
    """Compute summary stats for all condition CSVs in results_dir."""
    dfs = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("condition_") and fname.endswith(".csv"):
            try:
                cond_id = int(fname.split("_")[1])
                dfs[cond_id] = pd.read_csv(os.path.join(results_dir, fname))
            except (ValueError, IndexError):
                pass

    if not dfs:
        print("No condition CSVs found in results/")
        return None

    rows = []
    ece_bin_data = {}  # cond_id -> bin_stats DataFrame
    for cond_id, df in sorted(dfs.items()):
        # Fill missing ECE columns gracefully (old CSVs without logprob/confidence)
        for col in ["logprob_A", "logprob_B", "logprob_C", "logprob_D", "confidence"]:
            if col not in df.columns:
                df[col] = None

        n = len(df)
        n_correct = int(df["is_correct"].sum())
        accuracy = n_correct / n if n > 0 else 0.0
        parse_errors = int((df["predicted_answer"] == "PARSE_ERROR").sum())
        total_tokens = int(df["tokens_input"].sum() + df["tokens_output"].sum())
        cost = (
            df["tokens_input"].sum() * 0.00000015
            + df["tokens_output"].sum() * 0.0000006
        )

        ece, bin_stats = compute_ece(df)
        ece_bin_data[cond_id] = bin_stats
        conf_valid = df["confidence"].dropna()
        mean_conf = float(conf_valid.mean()) if len(conf_valid) > 0 else float("nan")
        overconf_gap = mean_conf - accuracy if not np.isnan(mean_conf) else float("nan")

        rows.append(
            {
                "condition_id": cond_id,
                "condition_name": CONDITION_NAMES.get(cond_id, f"condition_{cond_id}"),
                "n_questions": n,
                "n_correct": n_correct,
                "accuracy": accuracy,
                "parse_errors": parse_errors,
                "total_tokens": total_tokens,
                "estimated_cost_usd": cost,
                "ece": ece,
                "mean_confidence": round(mean_conf, 4),
                "overconfidence_gap": round(overconf_gap, 4),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(
        f"{'ID':<4} {'Condition':<26} {'N':>5} {'Correct':>8} "
        f"{'Accuracy':>9} {'Errors':>7} {'Cost':>10}"
    )
    print("-" * 80)
    for _, r in summary.iterrows():
        print(
            f"{int(r['condition_id']):<4} {r['condition_name']:<26} "
            f"{int(r['n_questions']):>5} {int(r['n_correct']):>8} "
            f"{r['accuracy']:>8.1%} {int(r['parse_errors']):>7} "
            f"${r['estimated_cost_usd']:>9.4f}"
        )
    total_cost = summary["estimated_cost_usd"].sum()
    print("=" * 80)
    print(f"Total estimated cost: ${total_cost:.4f}")

    # ── ECE table ────────────────────────────────────────────────────────────
    print(
        "\nECE TABLE (lower = better calibrated | "
        "Dahl et al. 2024 baseline ECE = 0.453)"
    )
    print("─" * 70)
    for _, r in summary.iterrows():
        name = r["condition_name"]
        ece_val = r["ece"]
        mc = r["mean_confidence"]
        og = r["overconfidence_gap"]
        ece_str = f"{ece_val:.3f}" if not np.isnan(ece_val) else "N/A"
        mc_str = f"{mc:.3f}" if not np.isnan(mc) else "N/A"
        og_str = (
            f"{og:+.3f}" if not np.isnan(og) else "N/A"
        )
        print(
            f"  Condition {int(r['condition_id'])} ({name:<24}): "
            f"ECE = {ece_str} | mean conf = {mc_str} | overconf = {og_str}"
        )
    print("─" * 70)

    # ── Save bin stats and reliability diagrams ───────────────────────────────
    for cond_id, bin_stats in ece_bin_data.items():
        name = CONDITION_NAMES.get(cond_id, f"condition_{cond_id}")
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if len(bin_stats) > 0:
            bin_stats.to_csv(
                os.path.join(results_dir, f"ece_bins_{safe_name}.csv"), index=False
            )
            plot_reliability_diagram(
                bin_stats,
                name,
                os.path.join(results_dir, f"reliability_{safe_name}.png"),
            )

    # ── McNemar's test vs baseline ───────────────────────────────────────────
    if 0 in dfs:
        print("\nMcNemar's test vs Condition 0 (Baseline):")
        print(f"{'Cond':<6} {'Name':<26} {'chi2':>8} {'p-value':>10} {'Sig':>5}")
        print("-" * 60)
        baseline = dfs[0]
        for cond_id, df in sorted(dfs.items()):
            if cond_id == 0:
                continue
            merged = baseline[["idx", "correct_answer", "predicted_answer"]].merge(
                df[["idx", "predicted_answer"]], on="idx", suffixes=("_base", "_cond")
            )
            chi2, p = _mcnemar(
                merged["correct_answer"],
                merged["predicted_answer_base"],
                merged["predicted_answer_cond"],
            )
            sig = "*" if (not np.isnan(p) and p < 0.05) else ""
            p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
            chi2_str = f"{chi2:.3f}" if not np.isnan(chi2) else "N/A"
            name = CONDITION_NAMES.get(cond_id, f"condition_{cond_id}")
            print(f"{cond_id:<6} {name:<26} {chi2_str:>8} {p_str:>10} {sig:>5}")

    # ── Per-subject accuracy ─────────────────────────────────────────────────
    print("\nPer-Subject Accuracy by Condition:")
    subj_data = {}
    for cond_id, df in sorted(dfs.items()):
        name = CONDITION_NAMES.get(cond_id, f"cond_{cond_id}")
        subj_data[f"{cond_id}_{name[:10]}"] = df.groupby("subject")["is_correct"].mean()

    if subj_data:
        subj_df = pd.DataFrame(subj_data).round(3)
        print(subj_df.to_string())
        subj_df.to_csv(os.path.join(results_dir, "per_subject_accuracy.csv"))

    return summary


def run_comparison(results_dir="results"):
    """
    Load zero_shot and few_shot summaries, merge them, compute per-condition
    McNemar test (zero-shot vs few-shot), and save comparison_summary.csv.
    """
    zs_dir = os.path.join(results_dir, "zero_shot")
    fs_dir = os.path.join(results_dir, "few_shot")

    zs_summary_path = os.path.join(zs_dir, "summary.csv")
    fs_summary_path = os.path.join(fs_dir, "summary.csv")

    if not os.path.exists(zs_summary_path):
        print(f"Zero-shot summary not found at {zs_summary_path}. Run --mode zero_shot first.")
        return None
    if not os.path.exists(fs_summary_path):
        print(f"Few-shot summary not found at {fs_summary_path}. Run --mode few_shot first.")
        return None

    zs_summary = pd.read_csv(zs_summary_path)
    fs_summary = pd.read_csv(fs_summary_path)

    # Load individual condition CSVs for McNemar
    def _load_cond_dfs(d):
        dfs = {}
        for fname in sorted(os.listdir(d)):
            if fname.startswith("condition_") and fname.endswith(".csv"):
                try:
                    cond_id = int(fname.split("_")[1])
                    dfs[cond_id] = pd.read_csv(os.path.join(d, fname))
                except (ValueError, IndexError):
                    pass
        return dfs

    zs_dfs = _load_cond_dfs(zs_dir)
    fs_dfs = _load_cond_dfs(fs_dir)

    rows = []
    for cond_id in sorted(set(zs_dfs.keys()) & set(fs_dfs.keys())):
        zs_row = zs_summary[zs_summary["condition_id"] == cond_id]
        fs_row = fs_summary[fs_summary["condition_id"] == cond_id]
        if zs_row.empty or fs_row.empty:
            continue

        zs_acc = float(zs_row["accuracy"].values[0])
        fs_acc = float(fs_row["accuracy"].values[0])
        diff = fs_acc - zs_acc

        # McNemar: zero-shot correct vs few-shot correct
        merged = zs_dfs[cond_id][["idx", "correct_answer", "predicted_answer"]].merge(
            fs_dfs[cond_id][["idx", "predicted_answer"]], on="idx", suffixes=("_zs", "_fs")
        )
        chi2, p = _mcnemar(
            merged["correct_answer"],
            merged["predicted_answer_zs"],
            merged["predicted_answer_fs"],
        )

        rows.append({
            "condition_id": cond_id,
            "condition_name": CONDITION_NAMES.get(cond_id, f"condition_{cond_id}"),
            "zero_shot_accuracy": round(zs_acc, 4),
            "few_shot_accuracy": round(fs_acc, 4),
            "difference": round(diff, 4),
            "mcnemar_chi2": round(chi2, 3) if not np.isnan(chi2) else float("nan"),
            "mcnemar_p": round(p, 4) if not np.isnan(p) else float("nan"),
            "significant_p05": "yes" if (not np.isnan(p) and p < 0.05) else "no",
        })

    if not rows:
        print("No conditions found in both zero_shot and few_shot directories.")
        return None

    comparison = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "comparison_summary.csv")
    comparison.to_csv(out_path, index=False)

    # ── Print comparison table ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ZERO-SHOT vs FEW-SHOT COMPARISON")
    print("=" * 80)
    print(
        f"{'ID':<4} {'Condition':<26} {'Zero-Shot':>10} {'Few-Shot':>10} "
        f"{'Diff':>8} {'p-value':>10} {'Sig':>5}"
    )
    print("-" * 80)
    for _, r in comparison.iterrows():
        p_str = f"{r['mcnemar_p']:.4f}" if not np.isnan(r["mcnemar_p"]) else "N/A"
        sig = "yes" if r["significant_p05"] == "yes" else ""
        print(
            f"{int(r['condition_id']):<4} {r['condition_name']:<26} "
            f"{r['zero_shot_accuracy']:>9.1%} {r['few_shot_accuracy']:>9.1%} "
            f"{r['difference']:>+7.1%} {p_str:>10} {sig:>5}"
        )
    print("=" * 80)
    print(f"Saved: {out_path}")
    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate MBE experiment results"
    )
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot", "both"],
        default="both",
        help=(
            "Which mode(s) to evaluate. "
            "'zero_shot' reads results/zero_shot/, "
            "'few_shot' reads results/few_shot/, "
            "'both' runs both and adds a comparison table."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root results directory (default: results/)",
    )
    args = parser.parse_args()

    if args.mode == "zero_shot":
        d = os.path.join(args.results_dir, "zero_shot")
        print(f"Evaluating zero-shot results from: {d}")
        run_evaluator(results_dir=d)

    elif args.mode == "few_shot":
        d = os.path.join(args.results_dir, "few_shot")
        print(f"Evaluating few-shot results from: {d}")
        run_evaluator(results_dir=d)

    elif args.mode == "both":
        zs_dir = os.path.join(args.results_dir, "zero_shot")
        fs_dir = os.path.join(args.results_dir, "few_shot")
        print(f"Evaluating zero-shot results from: {zs_dir}")
        run_evaluator(results_dir=zs_dir)
        print(f"\nEvaluating few-shot results from: {fs_dir}")
        run_evaluator(results_dir=fs_dir)
        run_comparison(results_dir=args.results_dir)
