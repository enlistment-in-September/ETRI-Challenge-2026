import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

BASE_DIR = Path(__file__).resolve().parent / 'data'
data = pd.read_csv(BASE_DIR / 'train_full.csv')

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
Q_TARGETS = ["Q1", "Q2", "Q3"]
S_TARGETS = ["S1", "S2", "S3", "S4"]


# =====================================================
# 1. Treatment 후보 정의
# =====================================================
TREATMENTS = {
    "screen_late":       "screen_late_ratio",
    "usage_late":        "usage_late_time",       # ratio → time으로 교체
    "hr_rmssd":          "hr_rmssd",
    "slp_screen":        "slp_screen_ratio",
    "hr_resting":        "hr_resting",
    "act_presleep":      "act_presleep_active",
    "ac_presleep":       "ac_presleep_charging",
}


def make_treatment(data, col, min_treat_rate=0.20):
    """
    개인별 중앙값 기준 이진화.
    treatment rate가 min_treat_rate 미만이면
    전체 분포의 75th percentile로 threshold 상향 조정.
    """
    s = data[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 1차: 개인별 중앙값
    subj_med = data.groupby("subject_id")[col].transform("median").fillna(0)
    A = (s > subj_med).astype(int)

    # treatment rate 체크 → 너무 낮으면 전체 75th로 재시도
    if A.mean() < min_treat_rate:
        q75 = s.quantile(0.75)
        A = (s > q75).astype(int)

    # 그래도 안 되면 50th
    if A.mean() < min_treat_rate or A.nunique() < 2:
        q50 = s.quantile(0.50)
        A = (s > q50).astype(int)

    return A


# =====================================================
# 2. Q/S 별 분리된 X, Z, W 구조
# =====================================================

# Q타겟용: 주관적 감정/피로 관련 피처 강조
X_Q = [c for c in [
    "dow", "is_weekend", "month", "subject_num",
    "act_active_ratio_lag1", "act_still_ratio_lag1",
    "usage_total_time_lag1", "screen_on_ratio_lag1",
    "mlight_all_mean_lag1", "gps_moving_ratio_lag1",
    "pedo_total_steps_lag1",
] if c in data.columns]

Z_Q = [c for c in [
    "screen_late_ratio_lag1", "screen_late_ratio_roll3",
    "usage_late_ratio_lag1", "screen_late_on_lag1",
] if c in data.columns]

W_Q = [c for c in [
    "hr_rmssd_lag1", "hr_mean_lag1", "hr_resting_lag1",
    "slp_hr_rmssd_lag1", "slp_hr_awake_ratio_lag1",
] if c in data.columns]

# S타겟용: 수면 관련 피처 강조
X_S = [c for c in [
    "dow", "is_weekend", "month", "subject_num",
    "pedo_total_steps_lag1", "pedo_total_steps_roll7",
    "act_active_ratio_lag1", "act_still_ratio_lag1",
    "wlight_all_mean_lag1", "ac_presleep_charging_lag1",
    "hr_resting_lag1",
] if c in data.columns]

Z_S = [c for c in [
    "screen_late_ratio_lag1", "screen_late_ratio_roll3",
    "usage_late_ratio_lag1", "screen_presleep_ratio_lag1",
    "ac_presleep_charging_lag1",
] if c in data.columns]

# W_S 재정의 — slp_* lag 컬럼이 없으면 W_Q로 fallback
W_S = [c for c in [
    "slp_hr_mean_lag1", "slp_hr_rmssd_lag1",
    "slp_hr_deep_ratio_lag1", "slp_hr_awake_ratio_lag1",
    "slp_screen_ratio_lag1", "slp_wlight_dark_lag1",
    # fallback: 일반 HR lag
    "hr_resting_lag1", "hr_mean_lag1", "hr_rmssd_lag1",
] if c in data.columns]

# W_S가 여전히 비면 W_Q 사용
if len(W_S) == 0:
    W_S = W_Q.copy()
    print("W_S fallback to W_Q")

print("=== Q structure ===")
print("X_Q:", X_Q)
print("Z_Q:", Z_Q)
print("W_Q:", W_Q)
print("\n=== S structure ===")
print("X_S:", X_S)
print("Z_S:", Z_S)
print("W_S:", W_S)


# =====================================================
# 3. 공통 함수
# =====================================================
def clean_df(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]  # ✅ 중복 컬럼 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in [col for col in df.columns if df[col].isna().any()]:
        df[c + "_missing"] = df[c].isna().astype(int)
    df = df.fillna(df.median(numeric_only=True).fillna(0)).fillna(0)
    return df


def get_structure(target):
    if target in Q_TARGETS:
        return X_Q, Z_Q, W_Q
    return X_S, Z_S, W_S


# =====================================================
# 4. Binary Proximal g-computation (2SRI)
# =====================================================
def proximal_gcomp_binary(data, target, A_col="A"):
    X_COLS, Z_COLS, W_COLS = get_structure(target)
    y = data[target].astype(int).values

    if len(Z_COLS) == 0 or len(W_COLS) == 0:
        return np.nan, np.nan

    step1_cols = list(dict.fromkeys([A_col] + Z_COLS + X_COLS))  # ✅ 순서 유지하며 중복 제거
    step1_X = clean_df(data[step1_cols])
    w_residuals = {}
    for w in W_COLS:
        if w not in data.columns:
            continue
        w_y = clean_df(data[[w]])[w].values
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(step1_X, w_y)
        w_residuals[f"{w}_resid"] = w_y - m.predict(step1_X)

    if len(w_residuals) == 0:
        return np.nan, np.nan

    w_resid_df = pd.DataFrame(w_residuals, index=data.index)
    w_cols_avail = [w for w in W_COLS if w in data.columns]

    outcome_df = pd.concat([data[[A_col] + X_COLS + w_cols_avail], w_resid_df], axis=1)
    outcome_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    outcome_model.fit(clean_df(outcome_df), y)

    base_cf = pd.concat([data[[A_col] + X_COLS + w_cols_avail], w_resid_df], axis=1)
    df1 = base_cf.copy(); df1[A_col] = 1
    df0 = base_cf.copy(); df0[A_col] = 0

    p1 = outcome_model.predict_proba(clean_df(df1))[:, 1]
    p0 = outcome_model.predict_proba(clean_df(df0))[:, 1]
    ace = float(np.mean(p1 - p0))

    pred = outcome_model.predict_proba(clean_df(base_cf))[:, 1]
    loss = log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6))
    return ace, loss


# =====================================================
# 5. Continuous ACE (dose-response)
# =====================================================
def continuous_ace(data, target, treat_col, n_points=20):
    X_COLS, Z_COLS, W_COLS = get_structure(target)
    y = data[target].astype(int).values
    w_cols_avail = [w for w in W_COLS if w in data.columns]

    if len(w_cols_avail) == 0:
        return None, None

    feat_cols = [treat_col] + X_COLS + w_cols_avail
    feat_cols = [c for c in feat_cols if c in data.columns]

    # ✅ clean_df를 fit 전에 한 번만 적용해서 컬럼 고정
    X_fit = clean_df(data[feat_cols])
    fixed_cols = X_fit.columns.tolist()  # missing indicator 포함한 고정 컬럼

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    model.fit(X_fit, y)

    lo = data[treat_col].quantile(0.10)
    hi = data[treat_col].quantile(0.90)
    treat_vals = np.linspace(lo, hi, n_points)

    mean_probs = []
    for val in treat_vals:
        df_cf = data[feat_cols].copy()
        df_cf[treat_col] = val
        # ✅ clean 후 고정 컬럼으로 맞춤
        df_cf_clean = clean_df(df_cf)
        for col in fixed_cols:
            if col not in df_cf_clean.columns:
                df_cf_clean[col] = 0
        df_cf_clean = df_cf_clean[fixed_cols]
        prob = model.predict_proba(df_cf_clean)[:, 1].mean()
        mean_probs.append(prob)

    return treat_vals, np.array(mean_probs)

# =====================================================
# 6. 부트스트랩 CI
# =====================================================
def bootstrap_ci(data, target, A_col="A", n_boot=100, seed=42):
    rng = np.random.default_rng(seed)
    aces = []
    for _ in range(n_boot):
        boot = data.sample(n=len(data), replace=True,
                           random_state=int(rng.integers(10000))).reset_index(drop=True)
        try:
            ace, _ = proximal_gcomp_binary(boot, target, A_col)
            if not np.isnan(ace):
                aces.append(ace)
        except Exception:
            continue
    if len(aces) == 0:
        return np.nan, np.nan
    return float(np.percentile(aces, 2.5)), float(np.percentile(aces, 97.5))


# =====================================================
# 7. 메인 실행
# =====================================================
all_rows = []

for treat_name, treat_col in TREATMENTS.items():
    print(f"\n{'='*55}")
    print(f"Treatment: {treat_name} ({treat_col})")

    if treat_col not in data.columns:
        print(f"  !! 컬럼 없음, 스킵")
        continue

    data["A"] = make_treatment(data, treat_col)
    rate = data["A"].mean()
    print(f"  Treatment rate: {rate:.3f}")

    if rate < 0.05 or rate > 0.95:
        print(f"  !! Treatment rate 극단적 ({rate:.3f}), 스킵")
        continue

    for target in TARGETS:
        y = data[target].values
        if len(np.unique(y)) < 2:
            continue

        print(f"  [{target}] 계산 중...", end=" ")
        ace, loss = proximal_gcomp_binary(data, target, A_col="A")

        if np.isnan(ace):
            print("NaN 스킵")
            continue

        ci_lo, ci_hi = bootstrap_ci(data, target, A_col="A", n_boot=200)
        sig = (not np.isnan(ci_lo)) and (ci_lo > 0 or ci_hi < 0)

        print(f"ACE={ace:+.4f}  CI=[{ci_lo:+.4f},{ci_hi:+.4f}] {'*' if sig else ''}")

        all_rows.append({
            "treatment": treat_name,
            "treat_col": treat_col,
            "target": target,
            "target_group": "Q" if target in Q_TARGETS else "S",
            "ACE": ace,
            "CI_lower": ci_lo,
            "CI_upper": ci_hi,
            "loss": loss,
            "significant": sig,
        })

results = pd.DataFrame(all_rows)
results.to_csv("proximal_gcomp_v3.csv", index=False, encoding="utf-8-sig")
print("\nSaved: proximal_gcomp_v3.csv")


# =====================================================
# 8. 시각화 A: ACE with CI (binary treatment)
# =====================================================
treat_names = results["treatment"].unique().tolist()
n_treats = len(treat_names)

fig, axes = plt.subplots(1, n_treats, figsize=(6 * n_treats, 7), sharey=False)
if n_treats == 1:
    axes = [axes]

fig.suptitle("Proximal G-Computation: ACE with 95% CI\n(Q/S Separate Structure)",
             fontsize=13, fontweight="bold", y=1.02)

colors = {"Q": "#2E86AB", "S": "#E84855"}

for ax, treat_name in zip(axes, treat_names):
    sub = results[results["treatment"] == treat_name].sort_values("target")
    if sub.empty:
        ax.set_title(treat_name); continue

    targets = sub["target"].tolist()
    aces    = sub["ACE"].values
    lo      = sub["CI_lower"].values
    hi      = sub["CI_upper"].values
    groups  = sub["target_group"].values
    sigs    = sub["significant"].values

    for i, (ace, l, h, grp, sig) in enumerate(zip(aces, lo, hi, groups, sigs)):
        color = colors[grp]
        alpha = 1.0 if sig else 0.4
        ax.errorbar(ace, i, xerr=[[ace - l], [h - ace]],
                    fmt="o", color=color, alpha=alpha,
                    markersize=8, capsize=5, linewidth=2)
        if sig:
            ax.text(max(h, ace) + 0.003, i, "★", color=color,
                    fontsize=12, va="center")

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=11)
    ax.set_xlabel("ACE", fontsize=10)
    ax.set_title(f"{treat_name}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

q_patch = mpatches.Patch(color=colors["Q"], label="Q (Subjective)")
s_patch = mpatches.Patch(color=colors["S"], label="S (Sleep)")
fig.legend(handles=[q_patch, s_patch], loc="lower center",
           ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.06))
plt.tight_layout()
plt.savefig("ace_ci_v3.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ace_ci_v3.png")


# =====================================================
# 9. 시각화 B: Dose-Response (continuous ACE)
# =====================================================
dose_treatments = {
    "screen_late": "screen_late_ratio",
    "hr_resting":  "hr_resting",
}
dose_targets = ["S3", "Q1"]

fig2, axes2 = plt.subplots(len(dose_targets), len(dose_treatments),
                            figsize=(5 * len(dose_treatments), 4 * len(dose_targets)))
fig2.suptitle("Dose-Response: Continuous Treatment Effect",
              fontsize=13, fontweight="bold")

for ri, target in enumerate(dose_targets):
    for ci, (tname, tcol) in enumerate(dose_treatments.items()):
        ax = axes2[ri][ci] if len(dose_targets) > 1 else axes2[ci]

        if tcol not in data.columns:
            ax.set_title(f"{tname} → {target}\n(no data)")
            continue

        vals, probs = continuous_ace(data, target, tcol)
        if vals is None:
            ax.set_title(f"{tname} → {target}\n(failed)")
            continue

        color = colors["Q"] if target in Q_TARGETS else colors["S"]
        ax.plot(vals, probs, color=color, linewidth=2.5)
        ax.fill_between(vals, probs - 0.02, probs + 0.02,
                        alpha=0.15, color=color)
        ax.set_xlabel(f"{tname}", fontsize=10)
        ax.set_ylabel(f"P({target}=1)", fontsize=10)
        ax.set_title(f"{tname} → {target}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("dose_response_v3.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: dose_response_v3.png")


# =====================================================
# 10. 요약 출력
# =====================================================
print("\n===== Q vs S 그룹별 평균 ACE =====")
summary = results.groupby(["treatment", "target_group"])["ACE"].agg(["mean", "std"])
print(summary.round(4))

print("\n===== 유의한 결과 (★) =====")
sig = results[results["significant"]]
if sig.empty:
    print("  없음")
else:
    print(sig[["treatment", "target", "ACE", "CI_lower", "CI_upper"]].to_string(index=False))