"""
test_fraud_inference.py

Sanity-check script to confirm that the standalone `fraud_inference.py`
pipeline reproduces the same predictions as workbook_train_eval.ipynb

Usage (from within the notebook, after calling export_training_artifacts):

    import sys
    sys.path.append('.')  # wherever fraud_inference.py lives
    from test_fraud_inference import run_consistency_check

    run_consistency_check(
        X_validate_raw=X_validate,      # the RAW dataframe, pre category-cast, with isFraud/TransactionID etc
        notebook_preds=preds,           # predict_proba output you already computed in-notebook
        artifact_dir='artifacts',
        n_sample=2000,
    )

Or run standalone via CLI (expects a CSV export of a raw validation slice
and a CSV/NPY of the matching notebook predictions):

    python test_fraud_inference.py --raw_csv val_raw.csv --preds_file val_preds.npy --artifacts artifacts
"""

import argparse

import numpy as np
import pandas as pd

from fraud_inference import load_artifacts, score_transactions


def run_consistency_check(X_validate_raw, notebook_preds, artifact_dir='artifacts',
                           n_sample=None, atol=1e-6, random_state=42):
    """
    Compares fraud_inference.py's scoring output against predictions already
    computed in the training notebook, on the same rows.

    Parameters
    ----------
    X_validate_raw : pd.DataFrame
        The RAW validation dataframe as it looked right after the
        train/validate time-split — i.e. BEFORE UID construction, UID stats
        merge, or categorical dtype casting. This is important: the whole
        point is to check that fraud_inference.py's own feature engineering
        reproduces the same result, so don't pre-engineer it here.
    notebook_preds : array-like
        The predict_proba (or Booster.predict, for LightGBM) probabilities
        you already computed in-notebook for X_validate, in the same row
        order as X_validate_raw.
    artifact_dir : str
        Directory where export_training_artifacts() wrote its files.
    n_sample : int or None
        If set, checks only a random sample of rows (faster for large
        validation sets). None checks all rows.
    atol : float
        Absolute tolerance for probability comparison. Tiny floating point
        differences (~1e-7) are expected and fine; anything larger suggests
        a real pipeline mismatch.

    Returns
    -------
    dict summary of the check, and raises an AssertionError with details
    if predictions diverge beyond tolerance.
    """
    notebook_preds = np.asarray(notebook_preds)
    assert len(X_validate_raw) == len(notebook_preds), (
        f"Row count mismatch: X_validate_raw has {len(X_validate_raw)} rows, "
        f"notebook_preds has {len(notebook_preds)}."
    )

    df = X_validate_raw.reset_index(drop=True).copy()
    ref_preds = pd.Series(notebook_preds).reset_index(drop=True)

    if n_sample is not None and n_sample < len(df):
        sample_idx = df.sample(n=n_sample, random_state=random_state).index
        df = df.loc[sample_idx].reset_index(drop=True)
        ref_preds = ref_preds.loc[sample_idx].reset_index(drop=True)

    artifacts = load_artifacts(artifact_dir)
    scored = score_transactions(df, artifacts)
    script_preds = scored['fraud_probability'].reset_index(drop=True)

    diffs = (script_preds - ref_preds).abs()
    max_diff = diffs.max()
    mean_diff = diffs.mean()
    n_mismatched = (diffs > atol).sum()

    summary = {
        'n_rows_checked': len(df),
        'max_abs_diff': float(max_diff),
        'mean_abs_diff': float(mean_diff),
        'n_rows_beyond_tolerance': int(n_mismatched),
        'tolerance': atol,
        'passed': bool(n_mismatched == 0),
    }

    print(f"Checked {summary['n_rows_checked']:,} rows")
    print(f"Max abs diff:  {summary['max_abs_diff']:.2e}")
    print(f"Mean abs diff: {summary['mean_abs_diff']:.2e}")
    print(f"Rows beyond tolerance ({atol}): {summary['n_rows_beyond_tolerance']:,}")

    if summary['passed']:
        print("\n✅ PASSED — fraud_inference.py reproduces notebook predictions.")
    else:
        worst_idx = diffs.sort_values(ascending=False).head(10).index
        print("\n❌ FAILED — pipeline mismatch detected. Worst offending rows:")
        comparison = pd.DataFrame({
            'notebook_pred': ref_preds.loc[worst_idx],
            'script_pred': script_preds.loc[worst_idx],
            'abs_diff': diffs.loc[worst_idx],
        })
        print(comparison)
        print(
            "\nCommon causes: UID_COLS in fraud_inference.py doesn't match "
            "the notebook's uid_cols, exported uid_stats/global_mean are stale "
            "(re-run export_training_artifacts after retraining), or the raw "
            "input still has pre-engineered columns (UID, UID_mean, etc.) left "
            "over from the notebook, confusing the merge step."
        )
        raise AssertionError(
            f"{summary['n_rows_beyond_tolerance']} of {summary['n_rows_checked']} "
            f"rows diverged beyond tolerance {atol}. See comparison above."
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Check fraud_inference.py predictions against notebook predictions."
    )
    parser.add_argument('--raw_csv', required=True,
                         help="CSV of the raw (pre-feature-engineering) validation slice.")
    parser.add_argument('--preds_file', required=True,
                         help="Path to notebook predictions, .npy or single-column .csv.")
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--n_sample', type=int, default=None)
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.raw_csv)

    if args.preds_file.endswith('.npy'):
        preds = np.load(args.preds_file)
    else:
        preds = pd.read_csv(args.preds_file).iloc[:, 0].values

    run_consistency_check(
        X_validate_raw=raw_df,
        notebook_preds=preds,
        artifact_dir=args.artifacts,
        n_sample=args.n_sample,
        atol=args.atol,
    )


if __name__ == '__main__':
    main()
