#!/usr/bin/env python3
"""
thesis_figures.py — Publication-quality figures for thesis.

Outputs 300 dpi PNG to results/thesis_figures/.
Run: python3 thesis_figures.py
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
from scipy.stats import chi2 as chi2_dist, norm as norm_dist

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_ZS  = 'results/zero_shot'
RESULTS_FS  = 'results/few_shot'
FM_PATH     = 'results/failure_modes_baseline.csv'
THESIS_DIR  = 'results/thesis_figures'

# ── Condition metadata ────────────────────────────────────────────────────────
COND_NAMES = {
    0: 'Baseline',
    1: 'Grounding',
    2: 'Rule Extraction',
    3: 'Chain of Logic',
    4: 'Neg. Elimination',
    5: 'Answer Verification',
    6: 'Self-Consistency',
    7: 'Rule Ext. + CoL',
}
COND_FILES = {
    0: 'condition_0_baseline.csv',
    1: 'condition_1_grounding.csv',
    2: 'condition_2_rule_extraction.csv',
    3: 'condition_3_chain_of_logic.csv',
    4: 'condition_4_negative_elimination.csv',
    5: 'condition_5_answer_verification.csv',
    6: 'condition_6_self_consistency.csv',
    7: 'condition_7_rule_col.csv',
}
SUBJECTS = ['CONST. LAW', 'CONTRACTS', 'CRIM. LAW', 'EVIDENCE', 'REAL PROP.', 'TORTS']

# ── Thesis style ──────────────────────────────────────────────────────────────
# Colorblind-friendly palette (Wong 2011)
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7',
           '#56B4E9', '#D55E00', '#F0E442', '#999999']

def set_thesis_style():
    sns.set_theme(style='ticks', font_scale=1.0)
    plt.rcParams.update({
        'figure.dpi':         300,
        'savefig.dpi':        300,
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          True,
        'axes.grid.axis':     'y',
        'grid.color':         '#e0e0e0',
        'grid.linewidth':     0.6,
        'font.family':        'sans-serif',
        'font.size':          10,
        'axes.titlesize':     12,
        'axes.labelsize':     11,
        'xtick.labelsize':    9,
        'ytick.labelsize':    9,
        'legend.fontsize':    9,
        'legend.frameon':     False,
        'figure.titlesize':   13,
    })

def savefig(name):
    path = os.path.join(THESIS_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {path}')

# ── Helper functions ──────────────────────────────────────────────────────────
def wilson_ci(n_correct, n_total, alpha=0.05):
    if n_total == 0:
        return 0., 0.
    z = norm_dist.ppf(1 - alpha / 2)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0., center - margin), min(1., center + margin)

def mcnemar_test(correct_a, correct_b):
    b = int(( correct_a & ~correct_b).sum())
    c = int((~correct_a &  correct_b).sum())
    if b + c == 0:
        return np.nan, np.nan, np.nan
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p    = 1 - chi2_dist.cdf(stat, df=1)
    phi  = np.sqrt(stat / (b + c))
    return stat, p, phi

def load_mode(results_dir):
    dfs = []
    for cid, fname in COND_FILES.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f'  WARNING: missing {path}')
            continue
        df = pd.read_csv(path)
        df['condition_id']   = cid
        df['condition_name'] = COND_NAMES[cid]
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined['is_correct'] = combined['is_correct'].astype(bool)
    return combined

def filter_complete(df):
    counts = df.groupby('idx')['condition_id'].nunique()
    complete_idx = counts[counts == df['condition_id'].nunique()].index
    return df[df['idx'].isin(complete_idx)].copy()

def accuracy_table(df):
    pivot = df.pivot_table(index='idx', columns='condition_id',
                           values='is_correct', aggfunc='first')
    n = len(pivot)
    baseline_correct = pivot[0].astype(bool)
    rows = []
    for cid in sorted(COND_NAMES):
        if cid not in pivot.columns:
            continue
        col = pivot[cid].astype(bool)
        n_correct = int(col.sum())
        acc = n_correct / n
        lo, hi = wilson_ci(n_correct, n)
        delta = acc - (int(baseline_correct.sum()) / n)
        if cid == 0:
            p_val, phi_val, sig = np.nan, np.nan, ''
        else:
            _, p_val, phi_val = mcnemar_test(baseline_correct, col)
            if not np.isnan(p_val):
                sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
            else:
                sig = ''
        rows.append({'Cond': cid, 'Name': COND_NAMES[cid], 'N': n,
                     'Accuracy': acc, 'CI_lo': lo, 'CI_hi': hi,
                     'Δ vs Baseline': delta, 'McNemar p': p_val,
                     'φ': phi_val, 'Sig': sig})
    return pd.DataFrame(rows)


def main():
    os.makedirs(THESIS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print('Loading data...')
    zs = filter_complete(load_mode(RESULTS_ZS))
    fs = filter_complete(load_mode(RESULTS_FS))
    fm = pd.read_csv(FM_PATH) if os.path.exists(FM_PATH) else None
    tbl_zs = accuracy_table(zs)
    tbl_fs = accuracy_table(fs)
    print(f'  ZS: {zs["idx"].nunique()} questions | FS: {fs["idx"].nunique()} questions')
    if fm is not None:
        print(f'  FM: {len(fm)} classified errors')

    set_thesis_style()
    print('\nGenerating figures...')


    # ═════════════════════════════════════════════════════════════════════════
    # FIG 1 — Accuracy per condition with 95% Wilson CIs
    # ═════════════════════════════════════════════════════════════════════════
    print('\n[1/8] Accuracy per condition')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    for ax, tbl, title in zip(axes, [tbl_zs, tbl_fs], ['(a) Zero-Shot', '(b) Few-Shot']):
        x = np.arange(len(tbl))
        bars = ax.bar(x, tbl['Accuracy'],
                      color=[PALETTE[i % len(PALETTE)] for i in tbl['Cond']],
                      alpha=0.85, width=0.6, zorder=3)
        ax.errorbar(x, tbl['Accuracy'],
                    yerr=[tbl['Accuracy'] - tbl['CI_lo'], tbl['CI_hi'] - tbl['Accuracy']],
                    fmt='none', color='#333333', capsize=3.5, linewidth=1.1, zorder=4)
        baseline_acc = tbl.loc[tbl['Cond'] == 0, 'Accuracy'].values[0]
        ax.axhline(baseline_acc, color='#555555', linestyle='--', linewidth=1,
                   label=f'Baseline ({baseline_acc:.3f})', zorder=2)
        # Significance stars
        for xi, (_, row) in zip(x, tbl.iterrows()):
            if row['Sig']:
                ax.text(xi, row['CI_hi'] + 0.012, row['Sig'],
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tbl['Name'], rotation=38, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title(title, pad=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.legend(loc='lower right')
        ax.set_ylim(0, tbl['CI_hi'].max() + 0.08)
        sns.despine(ax=ax)

    fig.suptitle('Accuracy per Prompting Condition (95% Wilson CI)\n'
                 '* p < 0.05, ** p < 0.01, *** p < 0.001', y=1.02)
    plt.tight_layout()
    savefig('fig1_accuracy_per_condition.png')


    # ═════════════════════════════════════════════════════════════════════════
    # FIG 2 — Zero-shot vs. few-shot × condition
    # ═════════════════════════════════════════════════════════════════════════
    print('[2/8] Zero-shot vs few-shot')
    common_idx = set(zs['idx'].unique()) & set(fs['idx'].unique())
    zs_c = zs[zs['idx'].isin(common_idx)]
    fs_c = fs[fs['idx'].isin(common_idx)]
    acc_zs = zs_c.groupby('condition_id')['is_correct'].mean()
    acc_fs = fs_c.groupby('condition_id')['is_correct'].mean()
    mode_df = pd.DataFrame({'zero_shot': acc_zs, 'few_shot': acc_fs}).reset_index()
    mode_df['condition_name'] = mode_df['condition_id'].map(COND_NAMES)
    mode_df['delta'] = mode_df['few_shot'] - mode_df['zero_shot']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    ax = axes[0]
    x = np.arange(len(mode_df))
    w = 0.38
    ax.bar(x - w/2, mode_df['zero_shot'], w, label='Zero-Shot',
           color=PALETTE[0], alpha=0.85, zorder=3)
    ax.bar(x + w/2, mode_df['few_shot'],  w, label='Few-Shot',
           color=PALETTE[1], alpha=0.85, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_df['condition_name'], rotation=38, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('(a) Accuracy by Mode and Condition')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend()
    sns.despine(ax=ax)

    ax = axes[1]
    colors = [PALETTE[0] if d >= 0 else PALETTE[5] for d in mode_df['delta']]
    ax.bar(x, mode_df['delta'], color=colors, alpha=0.85, width=0.6, zorder=3)
    ax.axhline(0, color='black', linewidth=0.8, zorder=4)
    for xi, val in zip(x, mode_df['delta']):
        ax.text(xi, val + (0.003 if val >= 0 else -0.003),
                f'{val:+.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_df['condition_name'], rotation=38, ha='right')
    ax.set_ylabel('Few-Shot − Zero-Shot Accuracy')
    ax.set_title('(b) Few-Shot Gain per Condition')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    sns.despine(ax=ax)

    fig.suptitle('Zero-Shot vs. Few-Shot Accuracy Across All Conditions', y=1.01)
    plt.tight_layout()
    savefig('fig2_zs_vs_fs_by_condition.png')


    # ═════════════════════════════════════════════════════════════════════════
    # FIG 3 — Subject × condition heatmap (FS — more interesting)
    # ═════════════════════════════════════════════════════════════════════════
    print('[3/8] Subject × condition heatmap')
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

    for ax, df, label in zip(axes, [zs, fs], ['(a) Zero-Shot', '(b) Few-Shot']):
        df_s = df[df['subject'].isin(SUBJECTS)]
        pivot = df_s.pivot_table(index='subject', columns='condition_id',
                                 values='is_correct', aggfunc='mean')
        pivot.columns = [COND_NAMES[c] for c in pivot.columns]
        delta = pivot.sub(pivot['Baseline'], axis=0)
        sns.heatmap(delta, annot=True, fmt='+.2f', cmap='RdYlGn',
                    center=0, vmin=-0.15, vmax=0.25, linewidths=0.4,
                    ax=ax, cbar_kws={'label': 'Δ vs Baseline', 'shrink': 0.8},
                    annot_kws={'size': 8})
        ax.set_title(label, pad=8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=38, ha='right')

    fig.suptitle('Per-Subject Accuracy Δ vs. Baseline per Condition', y=1.01)
    plt.tight_layout()
    savefig('fig3_subject_condition_heatmap.png')


    # ═════════════════════════════════════════════════════════════════════════
    # FIG 4 — Failure mode fix rate per condition (zero-shot)
    # ═════════════════════════════════════════════════════════════════════════
    print('[4/8] Failure mode fix rate')
    if fm is None:
        print('  Skipped — no failure modes CSV.')
    else:
        zs_fm = zs.merge(fm[['idx', 'failure_mode']], on='idx', how='left')
        baseline_errors = zs_fm[zs_fm['condition_id'] == 0][['idx', 'failure_mode']]
        baseline_errors = baseline_errors[baseline_errors['failure_mode'].isin(['FM1', 'FM2'])]

        rows = []
        for cid in sorted(COND_NAMES):
            cond_df = zs_fm[zs_fm['condition_id'] == cid]
            for fm_type in ['FM1', 'FM2']:
                fm_idx = baseline_errors[baseline_errors['failure_mode'] == fm_type]['idx']
                subset = cond_df[cond_df['idx'].isin(fm_idx)]
                if len(subset):
                    rows.append({'condition_id': cid, 'failure_mode': fm_type,
                                 'fix_rate': subset['is_correct'].mean()})
        fm_fix = pd.DataFrame(rows)
        fm_piv = fm_fix.pivot_table(index='condition_id', columns='failure_mode', values='fix_rate')

        fig, ax = plt.subplots(figsize=(9, 4.2))
        x = np.arange(len(fm_piv))
        w = 0.38
        ax.bar(x - w/2, fm_piv.get('FM1', 0), w,
               label='FM1 – Parametric Override', color=PALETTE[5], alpha=0.85, zorder=3)
        ax.bar(x + w/2, fm_piv.get('FM2', 0), w,
               label='FM2 – Reasoning Failure', color=PALETTE[0], alpha=0.85, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_NAMES[i] for i in fm_piv.index], rotation=38, ha='right')
        ax.set_ylabel('Fix Rate')
        ax.set_title('Fraction of Baseline Errors Resolved per Condition (Zero-Shot)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.legend()
        sns.despine(ax=ax)
        plt.tight_layout()
        savefig('fig4_fm_fix_rate_by_condition.png')


    # ═════════════════════════════════════════════════════════════════════════
    # FIG 5 — FM distribution per subject (stacked bar)
    # ═════════════════════════════════════════════════════════════════════════
    print('[5/8] FM per subject')
    if fm is None:
        print('  Skipped — no failure modes CSV.')
    else:
        bl_zs = zs[zs['condition_id'] == 0][['idx', 'subject', 'is_correct']].copy()
        bl_zs = bl_zs.merge(fm[['idx', 'failure_mode']], on='idx', how='left')
        wrong = bl_zs[~bl_zs['is_correct']].copy()
        wrong['subject'] = wrong['subject'].fillna('UNKNOWN')

        subj_fm = (wrong[wrong['subject'].isin(SUBJECTS)]
                   .groupby('subject')['failure_mode']
                   .value_counts(normalize=True)
                   .unstack(fill_value=0)
                   .reindex(columns=['FM1', 'FM2'], fill_value=0))
        subj_n  = wrong[wrong['subject'].isin(SUBJECTS)].groupby('subject').size()
        subj_fm = subj_fm.loc[subj_fm.sort_values('FM1', ascending=False).index]

        overall_fm1 = (wrong['failure_mode'] == 'FM1').mean()

        fig, ax = plt.subplots(figsize=(8, 4.2))
        x = np.arange(len(subj_fm))
        ax.bar(x, subj_fm['FM1'], label='FM1 – Parametric Override', color=PALETTE[5], alpha=0.85, zorder=3)
        ax.bar(x, subj_fm['FM2'], bottom=subj_fm['FM1'],
               label='FM2 – Reasoning Failure', color=PALETTE[0], alpha=0.85, zorder=3)
        ax.axhline(overall_fm1, color='#333333', linestyle='--', linewidth=1.1,
                   label=f'Overall FM1 rate ({overall_fm1:.1%})', zorder=4)
        for xi, subj in zip(x, subj_fm.index):
            ax.text(xi, 1.02, f'n={subj_n[subj]}', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(subj_fm.index, rotation=20, ha='right')
        ax.set_ylabel('Proportion of Baseline Errors')
        ax.set_title('Failure Mode Distribution per MBE Subject (Zero-Shot Baseline Errors)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.12)
        ax.legend()
        sns.despine(ax=ax)

        # Chi-sq annotation
        ct = pd.crosstab(wrong[wrong['subject'].isin(SUBJECTS)]['subject'],
                         wrong[wrong['subject'].isin(SUBJECTS)]['failure_mode'])
        chi2_val, p_chi, dof, _ = stats.chi2_contingency(ct)
        ax.text(0.98, 0.02, f'χ²({dof}) = {chi2_val:.1f}, p = {p_chi:.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                color='#555555')
        plt.tight_layout()
        savefig('fig5_fm_per_subject.png')


    # ═════════════════════════════════════════════════════════════════════════
    # FIG 7 — Cross-mode hardcore failures
    # ═════════════════════════════════════════════════════════════════════════
    print('[6/8] Cross-mode hardcore failures')
    def get_hardcore(df):
        pivot = df.pivot_table(index='idx', columns='condition_id',
                               values='is_correct', aggfunc='first')
        return set(pivot[pivot.apply(lambda r: not r.any(), axis=1)].index)

    hc_zs   = get_hardcore(zs)
    hc_fs   = get_hardcore(fs)
    hc_both = hc_zs & hc_fs

    both_meta = zs[zs['idx'].isin(hc_both) & (zs['condition_id'] == 0)][
        ['idx', 'subject']].copy()

    # HC rate per subject: fraction of each subject's questions that are both-mode hardcore
    meta_all_subj = zs[zs['condition_id'] == 0][['idx', 'subject']].copy()
    meta_all_subj['is_hc'] = meta_all_subj['idx'].isin(hc_both)
    hc_rate_subj = (meta_all_subj[meta_all_subj['subject'].isin(SUBJECTS)]
                    .groupby('subject')
                    .agg(n_total=('idx', 'count'), n_hc=('is_hc', 'sum'))
                    .assign(hc_rate=lambda d: d['n_hc'] / d['n_total'])
                    .sort_values('hc_rate', ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel a: overlap bar
    ax = axes[0]
    labels = ['ZS only', 'FS only', 'Both modes']
    vals   = [len(hc_zs - hc_fs), len(hc_fs - hc_zs), len(hc_both)]
    colors = [PALETTE[0], PALETTE[1], PALETTE[5]]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}\n({val/1195:.1%})', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Number of Questions')
    ax.set_title('(a) Hardcore Failure Overlap')
    ax.set_ylim(0, max(vals) * 1.25)
    sns.despine(ax=ax)

    # Panel b: HC rate per subject
    ax = axes[1]
    overall_hc_rate = len(hc_both) / zs['idx'].nunique()
    bar_colors = [PALETTE[5] if r > overall_hc_rate else PALETTE[0]
                  for r in hc_rate_subj['hc_rate']]
    ax.bar(range(len(hc_rate_subj)), hc_rate_subj['hc_rate'],
           color=bar_colors, alpha=0.85, width=0.55, zorder=3)
    ax.axhline(overall_hc_rate, color='#333333', linestyle='--', linewidth=1,
               label=f'Overall HC rate ({overall_hc_rate:.1%})', zorder=4)
    for xi, (subj, row) in enumerate(hc_rate_subj.iterrows()):
        ax.text(xi, row['hc_rate'] + 0.008,
                f"{row['n_hc']}/{row['n_total']}", ha='center', fontsize=8)
    ax.set_xticks(range(len(hc_rate_subj)))
    ax.set_xticklabels(hc_rate_subj.index, rotation=20, ha='right')
    ax.set_ylabel('HC failure rate\n(fraction of subject questions)')
    ax.set_title('(b) HC Rate per Subject\n(red = above average)')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend()
    sns.despine(ax=ax)

    fig.suptitle('Cross-Mode Hardcore Failures — Questions Wrong Across All 16 Conditions', y=1.01)
    plt.tight_layout()
    savefig('fig7_cross_mode_hardcore.png')


    # ═════════════════════════════════════════════════════════════════════════
    # FIG 8 — Cost-accuracy Pareto
    # ═════════════════════════════════════════════════════════════════════════
    print('[7/8] Cost-accuracy Pareto')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, df, label in zip(axes, [zs, fs], ['(a) Zero-Shot', '(b) Few-Shot']):
        df2 = df.copy()
        df2['total_tokens'] = df2['tokens_input'].fillna(0) + df2['tokens_output'].fillna(0)
        ca = df2.groupby('condition_id').agg(
            accuracy=('is_correct', 'mean'),
            avg_tokens=('total_tokens', 'mean')
        ).reset_index()
        for _, row in ca.iterrows():
            cid = int(row['condition_id'])
            ax.scatter(row['avg_tokens'], row['accuracy'], s=90,
                       color=PALETTE[cid % len(PALETTE)], zorder=5,
                       edgecolors='white', linewidths=0.6)
            ax.annotate(COND_NAMES[cid], (row['avg_tokens'], row['accuracy']),
                        textcoords='offset points', xytext=(6, 4), fontsize=8)
        ax.set_xlabel('Average Tokens per Question (input + output)')
        ax.set_ylabel('Accuracy')
        ax.set_title(label)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        # Mark desirable region (top-left: high accuracy, low cost)
        ax.annotate('Desirable\n(high accuracy,\nlow cost)',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    xytext=(0.22, 0.78), textcoords='axes fraction',
                    fontsize=8, color='#2ca02c', va='top', ha='left',
                    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.4,
                                    connectionstyle='arc3,rad=0.1'))
        sns.despine(ax=ax)

    fig.suptitle('Cost-Accuracy Pareto Frontier', y=1.01)
    plt.tight_layout()
    savefig('fig8_cost_accuracy_pareto.png')


    # ═════════════════════════════════════════════════════════════════════════
    # FIG 9 — Regression matrix (gains vs losses)
    # ═════════════════════════════════════════════════════════════════════════
    print('[8/8] Regression matrix')
    def regression_matrix(df):
        pivot = df.pivot_table(index='idx', columns='condition_id',
                               values='is_correct', aggfunc='first')
        baseline = pivot[0]
        rows = []
        for cid in sorted(COND_NAMES):
            if cid == 0 or cid not in pivot.columns:
                continue
            col = pivot[cid]
            gained = ((baseline == False) & (col == True)).sum()
            lost   = ((baseline == True)  & (col == False)).sum()
            rows.append({'Cond': cid, 'Name': COND_NAMES[cid],
                         'Gained': gained, 'Lost': lost, 'Net': gained - lost})
        return pd.DataFrame(rows)

    rm_zs = regression_matrix(zs)
    rm_fs = regression_matrix(fs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, rm, title in zip(axes, [rm_zs, rm_fs], ['(a) Zero-Shot', '(b) Few-Shot']):
        y = np.arange(len(rm))
        ax.barh(y,  rm['Gained'], color=PALETTE[2], alpha=0.85, label='Gained (wrong→right)', zorder=3)
        ax.barh(y, -rm['Lost'],   color=PALETTE[5], alpha=0.85, label='Lost (right→wrong)',   zorder=3)
        ax.axvline(0, color='black', linewidth=0.8, zorder=4)
        for yi, (_, row) in zip(y, rm.iterrows()):
            ax.text(row['Gained'] + 0.5, yi, f'+{row["Gained"]}',
                    va='center', fontsize=8, color=PALETTE[2])
            ax.text(-row['Lost'] - 0.5, yi, f'−{row["Lost"]}',
                    va='center', ha='right', fontsize=8, color=PALETTE[5])
            ax.text(0, yi + 0.42, f'net: {row["Net"]:+d}',
                    va='bottom', ha='center', fontsize=7.5)
        ax.set_yticks(y)
        ax.set_yticklabels(rm['Name'])
        ax.set_xlabel('Number of Questions')
        ax.set_title(title)
        ax.legend(loc='lower right')
        sns.despine(ax=ax)

    fig.suptitle('Questions Gained vs. Lost Relative to Baseline per Condition', y=1.01)
    plt.tight_layout()
    savefig('fig9_regression_matrix.png')

    print(f'\nDone. All figures saved to {THESIS_DIR}/')


if __name__ == "__main__":
    main()
