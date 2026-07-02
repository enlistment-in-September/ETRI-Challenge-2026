import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from config import TARGETS, OUTPUT_DIR, TRAIN_CSV


def _objective(weights, p1, p2, y):
    w = np.clip(weights, 0, 1)
    w = w / w.sum()
    blend = np.clip(w[0] * p1 + w[1] * p2, 1e-6, 1 - 1e-6)
    return log_loss(y, blend)


# LSTM 파일들은 submissions/ 폴더의 상위(data/)에 저장됨
LSTM_OOF_PATH = OUTPUT_DIR.parent / 'oof_5fold_bilstm_multitask_safe.csv'
LSTM_SUB_PATH = OUTPUT_DIR.parent / 'submission_5fold_bilstm_multitask_safe.csv'


def run():
    # Load train labels
    train_df = pd.read_csv(TRAIN_CSV)
    train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date']).dt.date.astype(str)

    # Load OOF predictions
    oof_lgb = pd.read_csv(OUTPUT_DIR / 'oof_lgbcatxgb.csv')
    oof_lstm = pd.read_csv(LSTM_OOF_PATH)

    for df in [oof_lgb, oof_lstm]:
        df['lifelog_date'] = pd.to_datetime(df['lifelog_date']).dt.date.astype(str)

    # Align order to oof_lgb
    oof_lstm = oof_lgb[['subject_id', 'lifelog_date']].merge(
        oof_lstm, on=['subject_id', 'lifelog_date'], how='left'
    )
    labels = oof_lgb[['subject_id', 'lifelog_date']].merge(
        train_df[['subject_id', 'lifelog_date'] + TARGETS],
        on=['subject_id', 'lifelog_date'], how='left'
    )

    # Optimize blending weights per target
    final_weights = {}
    for t in TARGETS:
        y = labels[t].values.astype(int)
        p_lgb = oof_lgb[t].values
        p_lstm_col = f'{t}_pred'
        p_lstm = oof_lstm[p_lstm_col].values if p_lstm_col in oof_lstm.columns else oof_lstm[t].values

        lgb_loss = log_loss(y, np.clip(p_lgb, 1e-6, 1 - 1e-6))
        lstm_loss = log_loss(y, np.clip(p_lstm, 1e-6, 1 - 1e-6))

        if t == "S3":
            w = np.array([0.5, 0.5])
        else:
            res = minimize(_objective, [0.8, 0.2], args=(p_lgb, p_lstm, y), method='Nelder-Mead')
            w = np.clip(res.x, 0, 1)
            w = w / w.sum()

        final_weights[t] = w
        blend_loss = log_loss(y, np.clip(w[0] * p_lgb + w[1] * p_lstm, 1e-6, 1 - 1e-6))
        print(f"{t}: LGB={w[0]:.3f}(loss={lgb_loss:.4f}), LSTM={w[1]:.3f}(loss={lstm_loss:.4f}), Blend={blend_loss:.5f}")

    # Load test predictions
    sub_lgb = pd.read_csv(OUTPUT_DIR / 'submission.csv')
    sub_lstm = pd.read_csv(LSTM_SUB_PATH)

    sub_blend = sub_lgb[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    for t in TARGETS:
        w = final_weights[t]
        sub_blend[t] = np.clip(w[0] * sub_lgb[t].values + w[1] * sub_lstm[t].values, 0.01, 0.99)

    out_path = OUTPUT_DIR / 'submission_blend_lstm.csv'
    sub_blend.to_csv(out_path, index=False)
    print(f"\n저장됨: {out_path}")


if __name__ == "__main__":
    run()
