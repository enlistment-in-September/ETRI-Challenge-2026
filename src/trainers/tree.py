import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score, classification_report
from scipy.optimize import minimize

from config import TRAIN_CSV, TEST_CSV, TARGETS, SEEDS, N_FOLDS, OUTPUT_DIR
from src.features.tree import build_tree_features

OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_CONFIG = {
    "Q1": {
        "top_k": 60,
        "lgb": dict(n_estimators=1200, learning_rate=0.012, max_depth=3, num_leaves=7,
                    min_child_samples=30, lambda_l1=5.0, lambda_l2=8.0),
        "cat": dict(iterations=1200, learning_rate=0.012, depth=3, l2_leaf_reg=9.0),
        "xgb": dict(n_estimators=1200, learning_rate=0.012, max_depth=3, min_child_weight=7,
                    reg_alpha=5.0, reg_lambda=8.0),
        "calib_C": 5.0,
    },
    "S1": {
        "top_k": 110,
        "lgb": dict(n_estimators=1200, learning_rate=0.015, max_depth=4, num_leaves=15,
                    min_child_samples=20, lambda_l1=3.0, lambda_l2=5.0),
        "cat": dict(iterations=1200, learning_rate=0.015, depth=4, l2_leaf_reg=6.0),
        "xgb": dict(n_estimators=1200, learning_rate=0.015, max_depth=4, min_child_weight=5,
                    reg_alpha=3.0, reg_lambda=5.0),
        "calib_C": 1e6,
    },
    "S2": {
        "top_k": 100,
        "lgb": dict(n_estimators=1200, learning_rate=0.015, max_depth=4, num_leaves=15,
                    min_child_samples=20, lambda_l1=2.0, lambda_l2=4.0),
        "cat": dict(iterations=1200, learning_rate=0.015, depth=4, l2_leaf_reg=5.0),
        "xgb": dict(n_estimators=1200, learning_rate=0.015, max_depth=4, min_child_weight=4,
                    reg_alpha=2.0, reg_lambda=4.0),
        "calib_C": 1e6,
    },
    "S4": {
        "top_k": 100,
        "lgb": dict(n_estimators=1200, learning_rate=0.015, max_depth=4, num_leaves=15,
                    min_child_samples=20, lambda_l1=2.0, lambda_l2=4.0),
        "cat": dict(iterations=1200, learning_rate=0.015, depth=4, l2_leaf_reg=5.0),
        "xgb": dict(n_estimators=1200, learning_rate=0.015, max_depth=4, min_child_weight=4,
                    reg_alpha=2.0, reg_lambda=4.0),
        "calib_C": 1e6,
    },
    "S3": {
        "top_k": 120,
        "lgb": dict(n_estimators=1200, learning_rate=0.018, max_depth=4, num_leaves=15,
                    min_child_samples=20, lambda_l1=2.0, lambda_l2=4.0),
        "cat": dict(iterations=1200, learning_rate=0.018, depth=4, l2_leaf_reg=5.0),
        "xgb": dict(n_estimators=1200, learning_rate=0.018, max_depth=4, min_child_weight=4,
                    reg_alpha=2.0, reg_lambda=4.0),
        "calib_C": 1e6,
    },
}

DEFAULT_Q = {
    "top_k": 40,
    "lgb": dict(n_estimators=1000, learning_rate=0.015, max_depth=4, num_leaves=15,
                min_child_samples=25, lambda_l1=3.0, lambda_l2=5.0),
    "cat": dict(iterations=1000, learning_rate=0.015, depth=4, l2_leaf_reg=7.0),
    "xgb": dict(n_estimators=1000, learning_rate=0.015, max_depth=4, min_child_weight=5,
                reg_alpha=3.0, reg_lambda=5.0),
    "calib_C": 1e6,
}

DEFAULT_S = {
    "top_k": 100,
    "lgb": dict(n_estimators=1000, learning_rate=0.02, max_depth=5, num_leaves=31,
                min_child_samples=15, lambda_l1=1.0, lambda_l2=2.0),
    "cat": dict(iterations=1000, learning_rate=0.02, depth=5, l2_leaf_reg=3.0),
    "xgb": dict(n_estimators=1000, learning_rate=0.02, max_depth=5, min_child_weight=3,
                reg_alpha=1.0, reg_lambda=2.0),
    "calib_C": 1e6,
}

CAUSAL_FEATURES = {
    "Q1": ["slp_screen_ratio", "slp_screen_on"],
    "S1": ["slp_screen_ratio", "slp_screen_on", "pedo_total_steps"],
    "S2": ["act_still_ratio_lag1", "pedo_total_steps"],
    "S3": ["screen_late_ratio", "screen_presleep_ratio", "screen_late_on", "ac_night_charging"],
    "S4": ["act_still_ratio_lag1", "pedo_total_steps"],
}


def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def objective_func(weights, preds, y_true):
    weights = np.clip(weights, 1e-5, 1)
    weights /= weights.sum()
    blend = sum(w * p for w, p in zip(weights, preds))
    blend = np.clip(blend, 1e-6, 1 - 1e-6)
    return log_loss(y_true, blend)


def run():
    print('Loading raw data...')
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for df in [train_df, test_df]:
        df['lifelog_date'] = pd.to_datetime(df['lifelog_date'])
        df['sleep_date'] = pd.to_datetime(df['sleep_date'])

    print('Extracting features...')
    feat_all, sleep_feats = build_tree_features(train_df, test_df)

    # Target encoding
    train_full = train_df.merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
    train_full = train_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
    test_full = test_df[['subject_id', 'lifelog_date', 'sleep_date']].merge(
        feat_all, on=['subject_id', 'lifelog_date'], how='left')
    test_full = test_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')

    all_with_labels = pd.concat([
        train_full[['subject_id', 'lifelog_date'] + TARGETS],
        test_full[['subject_id', 'lifelog_date']].assign(**{t: np.nan for t in TARGETS}),
    ], ignore_index=True).sort_values(['subject_id', 'lifelog_date'])

    enc_cols = []
    for t in TARGETS:
        for w in [3, 7, 14, 21]:
            col = f'{t}_enc{w}'
            all_with_labels[col] = all_with_labels.groupby('subject_id')[t].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            enc_cols.append(col)
        col_lag = f'{t}_lag1'
        all_with_labels[col_lag] = all_with_labels.groupby('subject_id')[t].shift(1)
        enc_cols.append(col_lag)

    enc_df = all_with_labels[['subject_id', 'lifelog_date'] + enc_cols]
    train_full = train_full.merge(enc_df, on=['subject_id', 'lifelog_date'], how='left')
    test_full = test_full.merge(enc_df, on=['subject_id', 'lifelog_date'], how='left')

    feature_cols = [c for c in train_full.columns
                    if c not in ['subject_id', 'lifelog_date', 'sleep_date'] + TARGETS]
    X_train = train_full[feature_cols].fillna(0)
    X_test = test_full[feature_cols].fillna(0)

    test_preds = np.zeros((len(X_test), len(TARGETS)))
    oof_preds = np.zeros((len(X_train), len(TARGETS)))

    print(f"\n===== CLEAN TARGET-SPECIFIC ENSEMBLE =====")

    for ti, target in enumerate(TARGETS):
        y = train_full[target].values
        print(f"\n===== {target} =====")

        cfg = TARGET_CONFIG.get(target, DEFAULT_Q if target.startswith("Q") else DEFAULT_S)
        top_k = cfg["top_k"]
        print(f"Feature Selection Top-{top_k}")

        selector = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=3,
                                          random_state=42, n_jobs=-1)
        selector.fit(X_train, y)
        imp = pd.Series(selector.feature_importances_, index=X_train.columns)

        causal_must = [f for f in CAUSAL_FEATURES.get(target, []) if f in X_train.columns]
        remaining_k = top_k - len(causal_must)
        imp_filtered = imp.drop(index=[f for f in causal_must if f in imp.index], errors='ignore')
        selected_features = causal_must + imp_filtered.nlargest(remaining_k).index.tolist()

        X_train_t = X_train[selected_features]
        X_test_t = X_test[selected_features]

        oof_lgb_all = np.zeros(len(X_train_t))
        oof_cat_all = np.zeros(len(X_train_t))
        oof_xgb_all = np.zeros(len(X_train_t))
        test_lgb_all = np.zeros(len(X_test_t))
        test_cat_all = np.zeros(len(X_test_t))
        test_xgb_all = np.zeros(len(X_test_t))

        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof_lgb = np.zeros(len(X_train_t))
            oof_cat = np.zeros(len(X_train_t))
            oof_xgb = np.zeros(len(X_train_t))
            test_lgb = np.zeros(len(X_test_t))
            test_cat = np.zeros(len(X_test_t))
            test_xgb = np.zeros(len(X_test_t))

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_t, y)):
                X_tr, y_tr = X_train_t.iloc[tr_idx], y[tr_idx]
                X_val, y_val = X_train_t.iloc[val_idx], y[val_idx]

                pos_count = y_tr.sum()
                neg_count = len(y_tr) - pos_count
                raw_spw = neg_count / pos_count if pos_count > 0 else 1.0
                clip_lo = 0.5 if target == "S3" else 1.0
                dynamic_spw = np.clip(raw_spw, clip_lo, 3.0)

                lgb_m = lgb.LGBMClassifier(**cfg["lgb"], scale_pos_weight=dynamic_spw, random_state=seed, verbosity=-1)
                cat_m = CatBoostClassifier(**cfg["cat"], random_seed=seed, verbose=0,
                                           early_stopping_rounds=50, allow_writing_files=False)
                xgb_m = XGBClassifier(**cfg["xgb"], scale_pos_weight=dynamic_spw, random_state=seed,
                                       eval_metric='logloss', early_stopping_rounds=50)

                lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
                cat_m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
                xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                oof_lgb[val_idx] = lgb_m.predict_proba(X_val)[:, 1]
                oof_cat[val_idx] = cat_m.predict_proba(X_val)[:, 1]
                oof_xgb[val_idx] = xgb_m.predict_proba(X_val)[:, 1]
                test_lgb += lgb_m.predict_proba(X_test_t)[:, 1] / N_FOLDS
                test_cat += cat_m.predict_proba(X_test_t)[:, 1] / N_FOLDS
                test_xgb += xgb_m.predict_proba(X_test_t)[:, 1] / N_FOLDS

            oof_lgb_all += oof_lgb / len(SEEDS)
            oof_cat_all += oof_cat / len(SEEDS)
            oof_xgb_all += oof_xgb / len(SEEDS)
            test_lgb_all += test_lgb / len(SEEDS)
            test_cat_all += test_cat / len(SEEDS)
            test_xgb_all += test_xgb / len(SEEDS)

        preds_list = [oof_lgb_all, oof_cat_all, oof_xgb_all]
        res = minimize(objective_func, [1/3, 1/3, 1/3], args=(preds_list, y),
                       method='L-BFGS-B', bounds=[(1e-5, 1)] * 3)
        best_weights = np.clip(res.x, 1e-5, 1)
        best_weights /= best_weights.sum()
        print("Weights:", best_weights)

        meta_oof = best_weights[0] * oof_lgb_all + best_weights[1] * oof_cat_all + best_weights[2] * oof_xgb_all
        meta_test = best_weights[0] * test_lgb_all + best_weights[1] * test_cat_all + best_weights[2] * test_xgb_all
        oof_preds[:, ti] = np.clip(meta_oof, 1e-6, 1 - 1e-6)

        meta_loss = log_loss(y, np.clip(meta_oof, 1e-6, 1 - 1e-6))
        print("Blend Loss:", meta_loss)

        calibrator = LogisticRegression(C=cfg["calib_C"], max_iter=1000, solver="lbfgs")
        calib_X = safe_logit(meta_oof).reshape(-1, 1)
        calib_test_X = safe_logit(meta_test).reshape(-1, 1)
        calibrator.fit(calib_X, y)
        calibrated_oof = calibrator.predict_proba(calib_X)[:, 1]
        calibrated_test = calibrator.predict_proba(calib_test_X)[:, 1]
        calib_loss = log_loss(y, calibrated_oof)
        print("Calibrated Loss:", calib_loss)

        pred_binary = (calibrated_oof > 0.5).astype(int)
        print(f"F1 Score (threshold=0.5): {f1_score(y, pred_binary):.4f}")
        print(classification_report(y, pred_binary))

        final_test = calibrated_test if calib_loss < meta_loss else meta_test
        print("Use calibrated" if calib_loss < meta_loss else "Use raw blend")

        clip_low = max(0.01, y.mean() * 0.10)
        clip_high = min(0.99, 1 - (1 - y.mean()) * 0.10)
        test_preds[:, ti] = np.clip(final_test, clip_low, clip_high)
        print(f"   -> Dynamic clipping: [{clip_low:.4f}, {clip_high:.4f}]")

    # Save OOF
    oof_df = train_full[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for i, t in enumerate(TARGETS):
        oof_df[t] = oof_preds[:, i]
    oof_path = OUTPUT_DIR / 'oof_lgbcatxgb.csv'
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF saved: {oof_path}")

    # Save submission
    submission = test_df[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for i, t in enumerate(TARGETS):
        submission[t] = test_preds[:, i]
    output_path = OUTPUT_DIR / 'submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved: {output_path}")


if __name__ == "__main__":
    run()

