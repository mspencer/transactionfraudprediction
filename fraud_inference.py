"""
inference py 

Produce the exact feature engineering pipeline used during training
(UID construction, UID-level aggregates, TransactionAmt_ratio)
So that the new/unseen transactions can be scored the same way.

Usage:
    python fraud_interence.py -- input new_transaction.csv --output scored.csv

Prerequisite: 
    exported models and training artifacts (export_training_artifacts)
"""

import argparse
import json
from pathlib import Path
import pandas as pd

# ---------------------
# config: must match the training config
# ---------------------
UID_COLS = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'D1n', 'P_emaildomain']
DROP_COLS = ['isFraud', 'TransactionID', 'TransactionDT', 'UID']

# default operating threshold: use best threshold from lightGBM = 0.253
# can be changed depending on business decision
DEFAULT_THRESHOLD = 0.253

# ---------------------
# load artifacts
# ---------------------
def load_artifacts(artifact_dir='artifacts'):
    artifact_dir = Path(artifact_dir)

    uid_stats = pd.read_pickle(artifact_dir / 'uid_stats.pkl')

    with open(artifact_dir / 'global_mean.json') as f:
        global_mean = json.load(f)['global_mean']

    with open(artifact_dir / 'feature_config.json') as f:
        config = json.load(f)

    with open(artifact_dir / 'train_categories.json') as f:
        train_categories = json.load(f)

    model_type = config['model_type']
    if model_type == 'lightgbm':
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(artifact_dir / 'model_lightgbm.txt'))
    elif model_type == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(enable_categorical=True)
        model.load_model(str(artifact_dir / 'model_xgboost.json'))
    elif model_type == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(artifact_dir / 'model_catboost.cbm'))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return {
        'model': model,
        'model_type': model_type,
        'uid_stats': uid_stats,
        'global_mean': global_mean,
        'features': config['features'],
        'cat_features': config['cat_features'],
        'threshold': config.get('threshold', DEFAULT_THRESHOLD),
        'train_categories': train_categories,
    }

# ---------------------
# feature engineering (mirror the training steps exactly)
# ---------------------
def prepare_features(raw_df, uid_stats, global_mean, model_type, cat_features, train_categories, uid_cols=UID_COLS):
    # reproduce the same feature engineering on new/raw transaction as during the training
    df = raw_df.copy()

    # prepare the feature engineering
    df['TransactionDay'] = (df['TransactionDT'] / 86400).astype(int)
    df['D1n'] = df['TransactionDay'] - df['D1']

    # UID construction
    df['UID'] = df[uid_cols].fillna('NA').astype(str).agg('_'.join, axis=1)

    # merge UID stats computed at training time
    df = df.drop(columns=[c for c in ['UID_mean', 'UID_std', 'UID_count'] if c in df.columns], errors='ignore')
    df = df.merge(uid_stats, on='UID', how='left')

    df['TransactionAmt_ratio'] = df['TransactionAmt'] / df['UID_mean'].fillna(global_mean)
    df['UID_std'] = df['UID_std'].fillna(0)
    df['UID_count'] = df['UID_count'].fillna(0)

    # categorical handling must meet the model requirements
    if model_type in ('lightgbm', 'xgboost'):
        # unseen categories -> NaN, same as during the training
        for col in cat_features:
            allowed = train_categories[col]
            df[col] = pd.Categorical(df[col], categories=allowed)
    elif model_type == 'catboost':
        # CatBoost wants strings, NaN filled explicitly
        for col in cat_features:
            df[col] = df[col].astype(str).fillna('NA')

    return df

# ---------------------
# scoring
# ---------------------
def score_transactions(raw_df, artifacts):
    # full pipeline: raw transactions in -> fraud probabilities + flags out
    model = artifacts['model']
    model_type = artifacts['model_type']
    features = artifacts['features']
    cat_features = artifacts['cat_features']
    threshold = artifacts['threshold']

    df = prepare_features(
        raw_df,
        uid_stats=artifacts['uid_stats'],
        global_mean=artifacts['global_mean'],
        model_type=model_type,
        cat_features=cat_features,
        train_categories=artifacts['train_categories'],
    )

    X = df[features]

    if model_type == 'lightgbm':
        # .predict() returns probability of the positive class directly
        proba = model.predict(X)
    else:
        # .predict_proba() returns probability of the negative and positive class
        # (XGBoost, CatBoost)
        proba = model.predict_proba(X)[:, 1]

    result = raw_df.copy()
    result['fraud_probability'] = proba
    result['is_fraud_predicted'] = (proba >= threshold).astype(int)

    return result

# ---------------------
# CLI entry point
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Score transactions for fraud probability.")
    parser.add_argument('--input', required=True, help="Path to CSV of raw new transactions.")
    parser.add_argument('--output', required=True, help="Path to write scored CSV.")
    parser.add_argument('--artifacts', default='artifacts', help="Directory containing exported training artifacts.")
    parser.add_argument('--threshold', type=float, default=None, help="Override the saved decision threshold.")
    print('1') 
    args = parser.parse_args()
    print('2')
    artifacts = load_artifacts(args.artifacts)
    print('3')
    if args.threshold is not None:
        artifacts['threshold'] = args.threshold
    print('4')
    raw_df = pd.read_csv(args.input)
    scored = score_transactions(raw_df, artifacts)

    scored.to_csv(args.output, index=False)
    n_flagged = scored['is_fraud_predicted'].sum()
    print(f"Scored {len(scored):,} transactions. "
          f"{n_flagged:,} flagged as fraud (threshold={artifacts['threshold']:.4f}).")
    print(f"Output written to {args.output}")

if __name__ == '__main__':
    main()    