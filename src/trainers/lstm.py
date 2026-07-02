import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

from config import (
    TRAIN_CSV, TEST_CSV, TARGETS, SEEDS, N_FOLDS,
    LOOKBACK, BATCH_SIZE, EPOCHS, LR, WD,
    HIDDEN, NUM_LAYERS, DROPOUT, MAX_SEQ_FEATURES, NUM_WORKERS,
    OUTPUT_DIR,
)
from src.utils import seed_everything, clip_probs
from src.features.lstm import (
    build_lstm_features, merge_feature_list, add_calendar_feats,
    add_subject_stats, add_recent_stats, sanitize_numeric_df,
    select_stable_features, fit_feature_stats, build_subject_daily_arrays,
)
from src.models.bilstm import SleepSequenceDataset, MultiTaskBiLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def assign_time_folds(df, n_folds=5):
    df = df.copy()
    df["fold"] = -1
    df["date_dt"] = pd.to_datetime(df["lifelog_date"])
    for sid in sorted(df["subject_id"].unique()):
        idx = df[df["subject_id"] == sid].sort_values("date_dt").index.tolist()
        for f, chunk in enumerate(np.array_split(idx, n_folds)):
            df.loc[chunk, "fold"] = f
    return df.drop(columns=["date_dt"])


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for batch in loader:
        seq = batch["seq"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        static = batch["static"].to(DEVICE)
        subject_id = batch["subject_id"].to(DEVICE)
        target = batch["target"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(seq, mask, static, subject_id)
        loss = criterion(logits, target)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 999.0


@torch.no_grad()
def valid_one_epoch(model, loader, criterion):
    model.eval()
    losses, preds = [], []
    for batch in loader:
        seq = batch["seq"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        static = batch["static"].to(DEVICE)
        subject_id = batch["subject_id"].to(DEVICE)
        target = batch["target"].to(DEVICE)
        logits = model(seq, mask, static, subject_id)
        loss = criterion(logits, target)
        if torch.isfinite(loss):
            losses.append(loss.item())
        preds.append(clip_probs(torch.sigmoid(logits).cpu().numpy()))
    return (float(np.mean(losses)) if losses else 999.0), np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        seq = batch["seq"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        static = batch["static"].to(DEVICE)
        subject_id = batch["subject_id"].to(DEVICE)
        preds.append(clip_probs(torch.sigmoid(model(seq, mask, static, subject_id)).cpu().numpy()))
    return np.concatenate(preds, axis=0)


def multi_target_logloss(y_true, y_pred, targets):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_probs(y_pred)
    scores = {}
    vals = []
    for i, t in enumerate(targets):
        s = log_loss(y_true[:, i], y_pred[:, i], labels=[0, 1])
        scores[t] = s
        vals.append(s)
    scores["avg"] = float(np.mean(vals))
    return scores


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    for df in [train, test]:
        df["lifelog_date"] = pd.to_datetime(df["lifelog_date"]).dt.date.astype(str)
        df["sleep_date"] = pd.to_datetime(df["sleep_date"]).dt.date.astype(str)

    base_all = pd.concat([
        train[["subject_id", "lifelog_date"]],
        test[["subject_id", "lifelog_date"]],
    ]).drop_duplicates().reset_index(drop=True)

    print("train:", train.shape, "test:", test.shape)

    # Feature engineering
    feature_tables = build_lstm_features(base_all)
    daily_feat = merge_feature_list(base_all, feature_tables)
    daily_feat = add_calendar_feats(daily_feat)
    daily_feat["num_missing"] = daily_feat.isna().sum(axis=1)

    exclude_cols = ["subject_id", "lifelog_date"]
    daily_feat = add_subject_stats(daily_feat, exclude_cols=exclude_cols)
    daily_feat = add_recent_stats(daily_feat, exclude_cols=exclude_cols, max_cols=60)
    daily_feat = sanitize_numeric_df(daily_feat)
    print("daily_feat:", daily_feat.shape)

    train_df = train.merge(daily_feat, on=["subject_id", "lifelog_date"], how="left")
    test_df = test.merge(daily_feat, on=["subject_id", "lifelog_date"], how="left")

    train_df = assign_time_folds(train_df, N_FOLDS)

    DROP_COLS = ["sleep_date", "fold"] + TARGETS
    BASE_SEQ_FEATURE_COLS = [
        c for c in train_df.columns
        if c not in DROP_COLS and c not in ["subject_id", "lifelog_date"]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    print("raw num seq features:", len(BASE_SEQ_FEATURE_COLS))

    all_subjects = sorted(pd.concat([train_df["subject_id"], test_df["subject_id"]]).unique())
    subject2idx = {s: i for i, s in enumerate(all_subjects)}

    oof_pred = np.zeros((len(train_df), len(TARGETS)), dtype=np.float32)
    test_pred = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)

    for seed in SEEDS:
        seed_everything(seed)
        print(f"\n================ SEED {seed} ================")

        oof_seed = np.zeros((len(train_df), len(TARGETS)), dtype=np.float32)
        test_seed = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)

        for fold in range(N_FOLDS):
            print(f"\n---- FOLD {fold} ----")
            tr_idx = train_df[train_df["fold"] != fold].index
            va_idx = train_df[train_df["fold"] == fold].index
            tr_part = train_df.loc[tr_idx].copy()
            va_part = train_df.loc[va_idx].copy()

            selected_cols = select_stable_features(tr_part, BASE_SEQ_FEATURE_COLS, MAX_SEQ_FEATURES)
            train_subjects = tr_part["subject_id"].unique().tolist()
            daily_tr = daily_feat[daily_feat["subject_id"].isin(train_subjects)].copy()
            med, scaler = fit_feature_stats(daily_tr, selected_cols)
            subject_daily_dict = build_subject_daily_arrays(daily_feat, selected_cols, med, scaler)

            loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
            ds_kwargs = dict(subject_daily_dict=subject_daily_dict, subject2idx=subject2idx,
                             feature_cols=selected_cols, lookback=LOOKBACK, targets=TARGETS)
            tr_loader = DataLoader(SleepSequenceDataset(tr_part, **ds_kwargs, is_test=False), shuffle=True, **loader_kwargs)
            va_loader = DataLoader(SleepSequenceDataset(va_part, **ds_kwargs, is_test=False), shuffle=False, **loader_kwargs)
            te_loader = DataLoader(SleepSequenceDataset(test_df, **ds_kwargs, is_test=True), shuffle=False, **loader_kwargs)

            model = MultiTaskBiLSTM(
                input_dim=len(selected_cols), n_subjects=len(subject2idx),
                hidden_dim=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT, out_dim=len(TARGETS),
            ).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            criterion = nn.BCEWithLogitsLoss()

            best_score, best_state, patience, wait = 1e9, None, 8, 0
            y_val = va_part[TARGETS].values.astype(np.float32)

            for epoch in range(1, EPOCHS + 1):
                tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion)
                va_loss, va_pred = valid_one_epoch(model, va_loader, criterion)
                va_score = multi_target_logloss(y_val, va_pred, TARGETS)["avg"]
                print(f"Seed {seed} Fold {fold} | Epoch {epoch:02d} | train {tr_loss:.5f} | valid {va_loss:.5f} | logloss {va_score:.5f}")
                if va_score < best_score:
                    best_score = va_score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            model.load_state_dict(best_state)
            oof_seed[va_idx] = predict_loader(model, va_loader)
            test_seed += predict_loader(model, te_loader) / N_FOLDS

            fold_scores = multi_target_logloss(y_val, oof_seed[va_idx], TARGETS)
            print(f"[Seed {seed} FOLD {fold}] scores: {fold_scores}")

            del model, tr_loader, va_loader, te_loader, subject_daily_dict
            gc.collect()
            torch.cuda.empty_cache()

        oof_pred += oof_seed / len(SEEDS)
        test_pred += test_seed / len(SEEDS)

    final_scores = multi_target_logloss(train_df[TARGETS].values.astype(np.float32), oof_pred, TARGETS)
    print("\n========== LSTM OOF ==========")
    for k, v in final_scores.items():
        print(f"{k}: {v:.6f}")

    # Save OOF
    oof_df = train_df[["subject_id", "sleep_date", "lifelog_date"] + TARGETS].copy()
    for i, t in enumerate(TARGETS):
        oof_df[f"{t}_pred"] = clip_probs(oof_pred[:, i])
    oof_path = OUTPUT_DIR.parent / "oof_5fold_bilstm_multitask_safe.csv"
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
    print("saved:", oof_path)

    # Save submission
    sub = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for i, t in enumerate(TARGETS):
        sub[t] = clip_probs(test_pred[:, i])
    sub_path = OUTPUT_DIR.parent / "submission_5fold_bilstm_multitask_safe.csv"
    sub.to_csv(sub_path, index=False, encoding="utf-8-sig")
    print("saved:", sub_path)


if __name__ == "__main__":
    run()

