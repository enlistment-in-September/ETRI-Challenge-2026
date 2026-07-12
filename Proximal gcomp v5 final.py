"""
Proximal_gcomp_final_full.py
============================
논문용 최종본.

포함:
- Liu et al. (2024) Procedure 3 기반 Logit-Logit proximal regression
- 주 추정량: Stage 2의 beta_A (proximal log-odds ratio)
- OR = exp(beta_A)
- subject-level cluster bootstrap 95% CI
- 무정규화 MLE 우선, quasi-separation 안정성 검사
- 불안정 시 weak ridge fallback
- Standard-X / Standard-XWZ 비교
- Z proxy-association 진단
- subject fixed-effect 사후 민감도 분석

제외:
- counterfactual g-computation ACE
- bootstrap p-value
- BH-FDR
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("default", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================
# 0. 데이터 로드
# =====================================================
DATA_PATH = (
    "/Users/bagjaeyong/Desktop/대학교/2026-1/"
    "ETRI 휴먼이해 인공지능/data 2/train_full.csv"
)

data = pd.read_csv(DATA_PATH)

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
Q_TARGETS = ["Q1", "Q2", "Q3"]
S_TARGETS = ["S1", "S2", "S3", "S4"]

N_BOOT = 500
BOOT_SEED = 42
MIN_BOOT_SUCCESS = 100
MAX_ITER = 3000


# =====================================================
# 1. Exposure 후보 정의
# =====================================================
TREATMENTS = {
    "screen_late": "screen_late_ratio",
    "slp_screen": "slp_screen_ratio",
    "hr_rmssd": "hr_rmssd",
    "hr_resting": "hr_resting",
    "act_presleep": "act_presleep_active",
    "ac_presleep": "ac_presleep_charging",
}


def make_treatment(data, col, min_treat_rate=0.20):
    """
    개인별 중앙값 기준으로 exposure 이진화.
    exposure 비율이 너무 낮으면 전체 75th percentile,
    그래도 낮으면 전체 중앙값 기준으로 완화.
    """
    s = data[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    subject_median = (
        data.groupby("subject_id")[col]
        .transform("median")
        .fillna(0)
    )

    A = (s > subject_median).astype(int)

    if A.mean() < min_treat_rate:
        A = (s > s.quantile(0.75)).astype(int)

    if A.mean() < min_treat_rate or A.nunique() < 2:
        A = (s > s.quantile(0.50)).astype(int)

    return A


# =====================================================
# 2. Q/S 그룹별 X, Z, W 정의
# =====================================================
X_Q = [
    c for c in [
        "dow",
        "is_weekend",
        "month",
        "act_active_ratio_lag1",
        "act_still_ratio_lag1",
        "screen_on_ratio_lag1",
        "mlight_all_mean_lag1",
        "gps_moving_ratio_lag1",
        "pedo_total_steps_lag1",
    ]
    if c in data.columns
]

Z_Q = [
    c for c in [
        "screen_late_ratio_lag1",
        "screen_late_ratio_roll3",
        "screen_late_on_lag1",
    ]
    if c in data.columns
]

W_Q_COL = next(
    (
        c for c in [
            "hr_rmssd_lag1",
            "slp_hr_awake_ratio",
        ]
        if c in data.columns
    ),
    None,
)

X_S = [
    c for c in [
        "dow",
        "is_weekend",
        "month",
        "pedo_total_steps_lag1",
        "pedo_total_steps_roll7",
        "act_active_ratio_lag1",
        "act_still_ratio_lag1",
        "wlight_all_mean_lag1",
        "ac_presleep_charging_lag1",
        "hr_resting_lag1",
    ]
    if c in data.columns
]

Z_S = [
    c for c in [
        "screen_late_ratio_lag1",
        "screen_late_ratio_roll3",
        "ac_presleep_charging_lag1",
    ]
    if c in data.columns
]

W_S_COL = next(
    (
        c for c in [
            "hr_mean_lag1",
            "hr_resting_lag1",
        ]
        if c in data.columns
    ),
    None,
)

print("=== Q structure ===")
print("X_Q:", X_Q)
print("Z_Q:", Z_Q)
print("W_Q:", W_Q_COL)

print("\n=== S structure ===")
print("X_S:", X_S)
print("Z_S:", Z_S)
print("W_S:", W_S_COL)


# =====================================================
# 3. 공통 함수
# =====================================================
def clean_df(df):
    """결측 indicator 생성 후 중앙값 대치."""
    df = df.copy()

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    original_cols = df.columns.tolist()

    for col in original_cols:
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(int)

    medians = df.median(numeric_only=True).fillna(0)

    return df.fillna(medians).fillna(0)


def align_design_matrix(df, reference_cols):
    """
    Stage 1의 Y=1 고정 예측 단계에서
    학습 시점과 동일한 설계행렬 열 구조를 유지.
    """
    df = clean_df(df)

    for col in reference_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[reference_cols].copy()


def binarize_w(data, w_col):
    """W를 참가자별 중앙값 기준으로 이진화."""
    w = (
        data[w_col]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(data[w_col].median())
    )

    subject_median = (
        data.groupby("subject_id")[w_col]
        .transform("median")
        .fillna(w.median())
    )

    return (w > subject_median).astype(int)


def get_structure(target):
    if target in Q_TARGETS:
        return X_Q, Z_Q, W_Q_COL

    return X_S, Z_S, W_S_COL


def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def safe_log_loss(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


# =====================================================
# 4. Logistic regression fitting
# =====================================================
def fit_logit(
    X,
    y,
    max_iter=MAX_ITER,
    coef_limit=15.0,
    extreme_prob_threshold=1e-8,
    max_extreme_prob_rate=0.05,
):
    """
    무정규화 MLE를 우선 시도.

    안정성 조건:
    1. max_iter 전에 수렴
    2. 계수가 유한
    3. 표준화된 계수 절댓값이 coef_limit 이하
    4. 예측확률이 0/1 근방에 과도하게 몰리지 않음

    불안정하면 weak ridge(C=1e4) fallback.
    """

    def is_stable(model):
        clf = model.named_steps["logisticregression"]

        n_iter = int(np.max(clf.n_iter_))
        coef = clf.coef_.ravel()

        if n_iter >= max_iter:
            return False, n_iter

        if not np.all(np.isfinite(coef)):
            return False, n_iter

        if np.max(np.abs(coef)) > coef_limit:
            return False, n_iter

        probabilities = model.predict_proba(X)[:, 1]

        if not np.all(np.isfinite(probabilities)):
            return False, n_iter

        extreme_rate = np.mean(
            (probabilities < extreme_prob_threshold)
            | (probabilities > 1 - extreme_prob_threshold)
        )

        if extreme_rate > max_extreme_prob_rate:
            return False, n_iter

        return True, n_iter

    # ------------------------------
    # 1) Unpenalized MLE
    # ------------------------------
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=42,
        ),
    )

    model.fit(X, y)

    stable, n_iter = is_stable(model)

    if stable:
        return model, "unpenalized", n_iter, True

    # ------------------------------
    # 2) Weak ridge fallback
    # ------------------------------
    fallback_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=1e4,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=42,
        ),
    )

    fallback_model.fit(X, y)

    stable_fb, fb_n_iter = is_stable(fallback_model)

    if not stable_fb:
        raise RuntimeError(
            "Both unpenalized and weak-ridge logistic regression "
            "were numerically unstable or failed to converge."
        )

    return fallback_model, "weak_ridge_fallback", fb_n_iter, True


def original_scale_beta(fitted_pipeline, feature_cols, feature_name):
    """StandardScaler 적용 계수를 원래 변수 단위 log-OR로 환산."""
    scaler = fitted_pipeline.named_steps["standardscaler"]
    clf = fitted_pipeline.named_steps["logisticregression"]

    idx = feature_cols.index(feature_name)

    return float(clf.coef_[0, idx] / scaler.scale_[idx])


# =====================================================
# 5. Regression-Based PCI: Procedure 3
# =====================================================
def proximal_logit_logit(
    data,
    target,
    A_col="A",
    extra_X_cols=None,
):
    """
    Liu et al. Procedure 3, binary Y/W Logit-Logit.

    Stage 1:
        W ~ A + Z + X + Y

    S:
        logit(P(W=1 | A, Z, X, Y=1))

    Stage 2:
        Y ~ A + S + W + X

    반환:
        beta_A, OR, observed fitted log-loss, diagnostics
    """
    X_COLS, Z_COLS, W_COL = get_structure(target)

    if extra_X_cols:
        X_COLS = list(X_COLS) + list(extra_X_cols)

    if A_col not in data.columns:
        return np.nan, np.nan, np.nan, None

    if target not in data.columns:
        return np.nan, np.nan, np.nan, None

    if W_COL is None or W_COL not in data.columns:
        return np.nan, np.nan, np.nan, None

    y = data[target].astype(int).values
    a = data[A_col].astype(int).values

    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, None

    if len(np.unique(a)) < 2:
        return np.nan, np.nan, np.nan, None

    W_bin = binarize_w(data, W_COL).values

    if len(np.unique(W_bin)) < 2:
        return np.nan, np.nan, np.nan, None

    # ------------------------------
    # Stage 1: W ~ A + Z + X + Y
    # ------------------------------
    stage1_cols = list(
        dict.fromkeys(
            [A_col] + Z_COLS + X_COLS + [target]
        )
    )

    stage1_cols = [
        col for col in stage1_cols
        if col in data.columns
    ]

    stage1_df = data[stage1_cols].copy()
    stage1_df[target] = y

    stage1_X = clean_df(stage1_df)
    stage1_features = stage1_X.columns.tolist()

    w_model, w_fit_type, w_n_iter, w_converged = fit_logit(
        stage1_X,
        W_bin,
    )

    # Y=1 고정
    stage1_y1 = stage1_df.copy()
    stage1_y1[target] = 1

    stage1_y1_X = align_design_matrix(
        stage1_y1,
        stage1_features,
    )

    prob_w = w_model.predict_proba(stage1_y1_X)[:, 1]
    S = safe_logit(prob_w)

    # ------------------------------
    # Stage 2: Y ~ A + S + W + X
    # ------------------------------
    stage2_cols = list(
        dict.fromkeys(
            [A_col] + X_COLS
        )
    )

    stage2_cols = [
        col for col in stage2_cols
        if col in data.columns
    ]

    stage2_df = data[stage2_cols].copy()
    stage2_df["S"] = S
    stage2_df["W"] = W_bin

    stage2_X = clean_df(stage2_df)
    stage2_features = stage2_X.columns.tolist()

    outcome_model, y_fit_type, y_n_iter, y_converged = fit_logit(
        stage2_X,
        y,
    )

    beta_a = original_scale_beta(
        outcome_model,
        stage2_features,
        A_col,
    )

    proximal_or = float(np.exp(beta_a))

    observed_prob = outcome_model.predict_proba(stage2_X)[:, 1]
    observed_loss = safe_log_loss(y, observed_prob)

    diagnostics = {
        "stage1_fit": w_fit_type,
        "stage1_converged": w_converged,
        "stage1_n_iter": w_n_iter,
        "stage2_fit": y_fit_type,
        "stage2_converged": y_converged,
        "stage2_n_iter": y_n_iter,
        "W_prevalence": float(W_bin.mean()),
    }

    return beta_a, proximal_or, observed_loss, diagnostics


# =====================================================
# 6. Standard logistic regression comparison
# =====================================================
def standard_logit(
    data,
    target,
    A_col="A",
    mode="XWZ",
):
    X_COLS, Z_COLS, W_COL = get_structure(target)

    if A_col not in data.columns:
        return np.nan, np.nan, np.nan, None

    if target not in data.columns:
        return np.nan, np.nan, np.nan, None

    y = data[target].astype(int).values

    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, None

    df = data.copy()

    if W_COL is not None and W_COL in df.columns:
        df["W_bin"] = binarize_w(df, W_COL).values

    if mode == "X":
        cols = [A_col] + X_COLS

    elif mode == "XW":
        cols = [A_col] + X_COLS + ["W_bin"]

    elif mode == "XWZ":
        cols = [A_col] + X_COLS + Z_COLS + ["W_bin"]

    else:
        raise ValueError("mode는 X, XW, XWZ 중 하나여야 합니다.")

    cols = list(dict.fromkeys(cols))

    cols = [
        col for col in cols
        if col in df.columns
    ]

    X_model = clean_df(df[cols])
    feature_cols = X_model.columns.tolist()

    model, fit_type, n_iter, converged = fit_logit(
        X_model,
        y,
    )

    beta_a = original_scale_beta(
        model,
        feature_cols,
        A_col,
    )

    odds_ratio = float(np.exp(beta_a))

    prediction = model.predict_proba(X_model)[:, 1]
    loss = safe_log_loss(y, prediction)

    return beta_a, odds_ratio, loss, fit_type


# =====================================================
# 7. Subject-level cluster bootstrap
# =====================================================
def bootstrap_ci(
    data,
    target,
    A_col="A",
    n_boot=N_BOOT,
    seed=BOOT_SEED,
    min_success=MIN_BOOT_SUCCESS,
    extra_X_cols=None,
):
    """
    참가자를 복원추출 단위로 사용하는 cluster bootstrap.
    """
    rng = np.random.default_rng(seed)

    all_subjects = data["subject_id"].dropna().unique()

    if len(all_subjects) < 5:
        return np.nan, np.nan, 0

    subject_to_idx = {
        subject: data.index[
            data["subject_id"] == subject
        ].to_numpy()
        for subject in all_subjects
    }

    betas = []

    for _ in range(n_boot):
        sampled_subjects = rng.choice(
            all_subjects,
            size=len(all_subjects),
            replace=True,
        )

        boot_idx = np.concatenate(
            [
                subject_to_idx[subject]
                for subject in sampled_subjects
            ]
        )

        boot_data = (
            data.loc[boot_idx]
            .reset_index(drop=True)
        )

        try:
            beta_a, _, _, _ = proximal_logit_logit(
                boot_data,
                target,
                A_col=A_col,
                extra_X_cols=extra_X_cols,
            )

            if np.isfinite(beta_a):
                betas.append(beta_a)

        except Exception:
            continue

    n_success = len(betas)

    if n_success < min_success:
        return np.nan, np.nan, n_success

    betas = np.asarray(betas)

    ci_lower = float(np.percentile(betas, 2.5))
    ci_upper = float(np.percentile(betas, 97.5))

    return ci_lower, ci_upper, n_success


# =====================================================
# 8. Z proxy-association diagnostic
# =====================================================
def compute_z_relevance(
    data,
    target,
    A_col="A",
):
    """
    W ~ A + Z + X + Y 와 W ~ A + X + Y 비교.

    Z 포함 시:
    - W 예측 log-loss 감소
    - W 예측 AUC 증가

    이면 Z가 W 예측에 추가 정보를 제공하는
    제한적 경험적 근거로 해석.
    """
    X_COLS, Z_COLS, W_COL = get_structure(target)

    if A_col not in data.columns:
        return None

    if target not in data.columns:
        return None

    if W_COL is None or W_COL not in data.columns:
        return None

    W_bin = binarize_w(data, W_COL).values

    if len(np.unique(W_bin)) < 2:
        return None

    cols_with_z = list(
        dict.fromkeys(
            [A_col] + X_COLS + Z_COLS + [target]
        )
    )

    cols_without_z = list(
        dict.fromkeys(
            [A_col] + X_COLS + [target]
        )
    )

    cols_with_z = [
        col for col in cols_with_z
        if col in data.columns
    ]

    cols_without_z = [
        col for col in cols_without_z
        if col in data.columns
    ]

    X_with_z = clean_df(data[cols_with_z])
    X_without_z = clean_df(data[cols_without_z])

    model_with_z, _, _, _ = fit_logit(
        X_with_z,
        W_bin,
    )

    model_without_z, _, _, _ = fit_logit(
        X_without_z,
        W_bin,
    )

    pred_with_z = model_with_z.predict_proba(X_with_z)[:, 1]
    pred_without_z = model_without_z.predict_proba(X_without_z)[:, 1]

    loss_with_z = safe_log_loss(W_bin, pred_with_z)
    loss_without_z = safe_log_loss(W_bin, pred_without_z)

    try:
        auc_with_z = roc_auc_score(W_bin, pred_with_z)
    except ValueError:
        auc_with_z = np.nan

    try:
        auc_without_z = roc_auc_score(W_bin, pred_without_z)
    except ValueError:
        auc_without_z = np.nan

    return {
        "n_Z_features": len(Z_COLS),
        "w_logloss_with_Z": loss_with_z,
        "w_logloss_without_Z": loss_without_z,
        "w_logloss_improvement": loss_without_z - loss_with_z,
        "w_auc_with_Z": auc_with_z,
        "w_auc_without_Z": auc_without_z,
        "w_auc_improvement": (
            auc_with_z - auc_without_z
            if np.isfinite(auc_with_z)
            and np.isfinite(auc_without_z)
            else np.nan
        ),
    }


# =====================================================
# 9. Subject fixed-effect helper
# =====================================================
def add_subject_dummies(df):
    """참가자 고정효과용 더미변수 생성."""
    dummies = pd.get_dummies(
        df["subject_id"],
        prefix="subj",
        drop_first=True,
    ).astype(int)

    output = pd.concat(
        [
            df.reset_index(drop=True),
            dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    return output, dummies.columns.tolist()


# =====================================================
# 10. Main analysis
# =====================================================
all_rows = []

for treat_name, treat_col in TREATMENTS.items():
    print("\n" + "=" * 70)
    print(f"Exposure: {treat_name} ({treat_col})")

    if treat_col not in data.columns:
        print("  컬럼 없음 -> skip")
        continue

    analysis_data = data.copy()

    analysis_data["A"] = make_treatment(
        analysis_data,
        treat_col,
    )

    treatment_rate = float(analysis_data["A"].mean())
    n_treated = int(analysis_data["A"].sum())
    n_control = int((analysis_data["A"] == 0).sum())

    print(
        f"  exposure rate={treatment_rate:.3f} | "
        f"exposed={n_treated} | "
        f"unexposed={n_control}"
    )

    if treatment_rate < 0.05 or treatment_rate > 0.95:
        print("  exposure rate 극단적 -> skip")
        continue

    if min(n_treated, n_control) < 20:
        print("  exposed/unexposed 표본 부족 -> skip")
        continue

    for target in TARGETS:
        if target not in analysis_data.columns:
            continue

        y = (
            analysis_data[target]
            .dropna()
            .astype(int)
            .values
        )

        if len(np.unique(y)) < 2:
            print(f"  [{target}] outcome variation 없음 -> skip")
            continue

        print(f"  [{target}] 계산 중...", end=" ")

        try:
            (
                beta_a,
                proximal_or,
                proximal_loss,
                diagnostics,
            ) = proximal_logit_logit(
                analysis_data,
                target,
                A_col="A",
            )

            if not np.isfinite(beta_a):
                print("beta_A 계산 실패 -> skip")
                continue

            (
                std_x_beta,
                std_x_or,
                std_x_loss,
                std_x_fit,
            ) = standard_logit(
                analysis_data,
                target,
                A_col="A",
                mode="X",
            )

            (
                std_xwz_beta,
                std_xwz_or,
                std_xwz_loss,
                std_xwz_fit,
            ) = standard_logit(
                analysis_data,
                target,
                A_col="A",
                mode="XWZ",
            )

            ci_lo, ci_hi, n_boot_ok = bootstrap_ci(
                analysis_data,
                target,
                A_col="A",
            )

            significant = (
                np.isfinite(ci_lo)
                and np.isfinite(ci_hi)
                and (ci_lo > 0 or ci_hi < 0)
            )

            z_diag = compute_z_relevance(
                analysis_data,
                target,
                A_col="A",
            )

            print(
                f"logOR={beta_a:+.4f} | "
                f"OR={proximal_or:.3f} | "
                f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] | "
                f"boot_ok={n_boot_ok} "
                f"{'★' if significant else ''}"
            )

            row = {
                "treatment": treat_name,
                "treat_col": treat_col,
                "target": target,
                "target_group": (
                    "Q"
                    if target in Q_TARGETS
                    else "S"
                ),
                "treatment_rate": treatment_rate,
                "n_treated": n_treated,
                "n_control": n_control,

                "standard_X_logOR": std_x_beta,
                "standard_X_OR": std_x_or,
                "standard_X_loss": std_x_loss,
                "standard_X_fit": std_x_fit,

                "standard_XWZ_logOR": std_xwz_beta,
                "standard_XWZ_OR": std_xwz_or,
                "standard_XWZ_loss": std_xwz_loss,
                "standard_XWZ_fit": std_xwz_fit,

                "proximal_logOR": beta_a,
                "proximal_OR": proximal_or,
                "proximal_loss": proximal_loss,

                "CI_lower_logOR": ci_lo,
                "CI_upper_logOR": ci_hi,
                "CI_lower_OR": (
                    np.exp(ci_lo)
                    if np.isfinite(ci_lo)
                    else np.nan
                ),
                "CI_upper_OR": (
                    np.exp(ci_hi)
                    if np.isfinite(ci_hi)
                    else np.nan
                ),

                "bootstrap_success": n_boot_ok,
                "significant": significant,

                "stage1_fit": diagnostics["stage1_fit"],
                "stage1_converged": diagnostics[
                    "stage1_converged"
                ],
                "stage1_n_iter": diagnostics[
                    "stage1_n_iter"
                ],

                "stage2_fit": diagnostics["stage2_fit"],
                "stage2_converged": diagnostics[
                    "stage2_converged"
                ],
                "stage2_n_iter": diagnostics[
                    "stage2_n_iter"
                ],

                "W_prevalence": diagnostics[
                    "W_prevalence"
                ],
            }

            if z_diag is not None:
                row.update(z_diag)

            all_rows.append(row)

        except Exception as e:
            print(
                f"오류 -> {type(e).__name__}: {e}"
            )
            continue


# =====================================================
# 11. 결과 저장 및 ridge fallback 요약
# =====================================================
results = pd.DataFrame(all_rows)

if results.empty:
    raise RuntimeError(
        "결과가 생성되지 않았습니다. "
        "exposure/outcome 컬럼과 exposure rate를 확인하세요."
    )

RESULT_CSV = "proximal_regression_final_results.csv"

results.to_csv(
    RESULT_CSV,
    index=False,
    encoding="utf-8-sig",
)

print(f"\nSaved: {RESULT_CSV}")

n_total = len(results)

n_stage1_fallback = int(
    (results["stage1_fit"] != "unpenalized").sum()
)

n_stage2_fallback = int(
    (results["stage2_fit"] != "unpenalized").sum()
)

print(
    f"\n===== Stage 1 ridge fallback: "
    f"{n_stage1_fallback}/{n_total} ====="
)

print(
    f"===== Stage 2 ridge fallback: "
    f"{n_stage2_fallback}/{n_total} ====="
)


# =====================================================
# 12. Subject fixed-effect sensitivity analysis
# =====================================================
sig_pairs = (
    results[results["significant"]][
        ["treatment", "treat_col", "target"]
    ]
    .drop_duplicates()
)

sensitivity_rows = []

if not sig_pairs.empty:
    print(
        "\n===== Subject Fixed-Effect "
        "Post-Selection Sensitivity Analysis ====="
    )

    print(
        "(CI 기준으로 후보 관계로 선별된 조합만 분석. "
        "독립적 재검증이 아닌 사후 민감도 분석.)"
    )

    for _, row in sig_pairs.iterrows():
        treat_name = row["treatment"]
        treat_col = row["treat_col"]
        target = row["target"]

        sens_data = data.copy()

        sens_data["A"] = make_treatment(
            sens_data,
            treat_col,
        )

        sens_data, subject_dummy_cols = add_subject_dummies(
            sens_data
        )

        try:
            beta_fe, or_fe, loss_fe, diagnostics_fe = (
                proximal_logit_logit(
                    sens_data,
                    target,
                    A_col="A",
                    extra_X_cols=subject_dummy_cols,
                )
            )

            ci_lo_fe, ci_hi_fe, n_boot_fe = bootstrap_ci(
                sens_data,
                target,
                A_col="A",
                n_boot=N_BOOT,
                seed=BOOT_SEED,
                extra_X_cols=subject_dummy_cols,
            )

            significant_fe = (
                np.isfinite(ci_lo_fe)
                and np.isfinite(ci_hi_fe)
                and (ci_lo_fe > 0 or ci_hi_fe < 0)
            )

            main_beta = results.loc[
                (
                    results["treatment"] == treat_name
                )
                & (
                    results["target"] == target
                ),
                "proximal_logOR",
            ].values[0]

            print(
                f"  {treat_name} -> {target} | "
                f"main={main_beta:+.4f} | "
                f"FE={beta_fe:+.4f} | "
                f"OR={or_fe:.3f} | "
                f"CI=[{ci_lo_fe:+.4f}, {ci_hi_fe:+.4f}] | "
                f"boot_ok={n_boot_fe} "
                f"{'★ 유지됨' if significant_fe else '× CI가 0 포함'}"
            )

            sensitivity_rows.append(
                {
                    "treatment": treat_name,
                    "target": target,

                    "beta_A_with_subject_FE": beta_fe,
                    "OR_with_subject_FE": or_fe,
                    "loss_with_subject_FE": loss_fe,

                    "CI_lower_logOR_FE": ci_lo_fe,
                    "CI_upper_logOR_FE": ci_hi_fe,

                    "bootstrap_success_FE": n_boot_fe,
                    "significant_with_FE": significant_fe,

                    "stage1_fit_FE": diagnostics_fe[
                        "stage1_fit"
                    ],
                    "stage2_fit_FE": diagnostics_fe[
                        "stage2_fit"
                    ],
                }
            )

        except Exception as e:
            print(
                f"  {treat_name} -> {target} | "
                f"FE 분석 실패: {type(e).__name__}: {e}"
            )

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    SENSITIVITY_CSV = (
        "proximal_regression_subject_FE_sensitivity.csv"
    )

    sensitivity_df.to_csv(
        SENSITIVITY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved: {SENSITIVITY_CSV}")

else:
    print(
        "\nCI 기준 후보 관계가 없어 "
        "subject FE 민감도 분석을 생략합니다."
    )

    sensitivity_df = pd.DataFrame()


# =====================================================
# 13. 시각화
# =====================================================
treat_names = results["treatment"].unique().tolist()
n_treats = len(treat_names)

colors = {
    "Q": "#2E86AB",
    "S": "#E84855",
}

fig, axes = plt.subplots(
    1,
    n_treats,
    figsize=(5 * n_treats, 7),
    sharey=False,
)

if n_treats == 1:
    axes = [axes]

fig.suptitle(
    "Regression-Based Proximal Causal Inference\n"
    "Logit-Logit Procedure 3 | "
    "95% Subject-Level Cluster Bootstrap CI",
    fontsize=12,
    fontweight="bold",
    y=1.03,
)

for ax, treat_name in zip(axes, treat_names):
    sub = (
        results[
            results["treatment"] == treat_name
        ]
        .sort_values("target")
        .reset_index(drop=True)
    )

    for i, row in sub.iterrows():
        est = row["proximal_logOR"]
        lo = row["CI_lower_logOR"]
        hi = row["CI_upper_logOR"]

        color = colors[row["target_group"]]

        alpha = (
            1.0
            if row["significant"]
            else 0.40
        )

        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(
                est,
                i,
                xerr=[
                    [max(est - lo, 0)],
                    [max(hi - est, 0)],
                ],
                fmt="o",
                color=color,
                alpha=alpha,
                markersize=8,
                capsize=5,
                linewidth=2,
            )

        else:
            ax.scatter(
                est,
                i,
                color=color,
                alpha=alpha,
                s=60,
            )

        if row["significant"]:
            ax.text(
                max(hi, est) + 0.03,
                i,
                "★",
                color=color,
                fontsize=12,
                va="center",
            )

    ax.axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_yticks(range(len(sub)))

    ax.set_yticklabels(
        sub["target"].tolist(),
        fontsize=11,
    )

    ax.set_xlabel(
        "Proximal log-odds ratio estimate\n"
        "under proxy assumptions",
        fontsize=10,
    )

    ax.set_title(
        treat_name,
        fontsize=11,
        fontweight="bold",
    )

    ax.grid(axis="x", alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

q_patch = mpatches.Patch(
    color=colors["Q"],
    label="Q: subjective outcome",
)

s_patch = mpatches.Patch(
    color=colors["S"],
    label="S: sleep outcome",
)

fig.legend(
    handles=[q_patch, s_patch],
    loc="lower center",
    ncol=2,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.06),
)

plt.tight_layout()

FIG_PATH = "proximal_logOR_ci_final.png"

plt.savefig(
    FIG_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.show()

print(f"Saved: {FIG_PATH}")


# =====================================================
# 14. 요약 출력
# =====================================================
print("\n===== Standard vs Proximal 추정치 비교 =====")

summary_cols = [
    "treatment",
    "target",
    "treatment_rate",

    "standard_X_logOR",
    "standard_XWZ_logOR",

    "proximal_logOR",
    "proximal_OR",

    "CI_lower_logOR",
    "CI_upper_logOR",

    "CI_lower_OR",
    "CI_upper_OR",

    "bootstrap_success",
]

print(
    results[summary_cols]
    .round(4)
    .to_string(index=False)
)

print("\n===== 95% CI가 0을 포함하지 않는 후보 관계 =====")

sig = results[
    results["significant"]
].copy()

if sig.empty:
    print("없음")

else:
    print(
        sig[
            [
                "treatment",
                "target",
                "proximal_logOR",
                "proximal_OR",
                "CI_lower_OR",
                "CI_upper_OR",
                "bootstrap_success",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

print("\n===== Z proxy-association diagnostic =====")

if not sig.empty:
    z_summary_cols = [
        "treatment",
        "target",

        "n_Z_features",

        "w_logloss_without_Z",
        "w_logloss_with_Z",
        "w_logloss_improvement",

        "w_auc_without_Z",
        "w_auc_with_Z",
        "w_auc_improvement",
    ]

    available_z_cols = [
        col for col in z_summary_cols
        if col in sig.columns
    ]

    print(
        sig[available_z_cols]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n해석: w_logloss_improvement > 0 및 "
        "w_auc_improvement > 0이면, "
        "Z가 W 예측에 추가 정보를 제공하는 "
        "제한적 경험적 근거로 해석합니다."
    )

if not sensitivity_df.empty:
    print(
        "\n===== Subject Fixed-Effect "
        "Post-Selection Sensitivity Analysis ====="
    )

    print(
        sensitivity_df
        .round(4)
        .to_string(index=False)
    )

print("\n===== Q / S 그룹별 평균 Proximal log-OR =====")

print(
    results
    .groupby(
        ["treatment", "target_group"]
    )["proximal_logOR"]
    .agg(["mean", "std", "count"])
    .round(4)
)