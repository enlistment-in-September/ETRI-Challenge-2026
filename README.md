# Life-Log Sleep Quality Prediction

ETRI 휴먼이해 AI 챌린지 2026 - 스마트폰·웨어러블 라이프로그 기반 수면 품질 예측

## 과제 개요

하루치 라이프로그 데이터(활동, 심박수, 조도, GPS 등 12가지 센서)를 사용해 수면 관련 7개 타겟(`Q1, Q2, Q3, S1, S2, S3, S4`)의 이진 확률을 예측합니다.

평가 지표: **Log-Loss** (타겟별 평균)

## 파이프라인 구조

```
STEP 1  Tree Ensemble  ─────────────────────────────────────────────────
  LGBMClassifier + CatBoostClassifier + XGBClassifier
  · 타겟별 하이퍼파라미터 설정 (TARGET_CONFIG)
  · RandomForest로 Top-K 피처 선택 + 인과 피처 강제 포함
  · 5-Fold × 5-Seed → OOF 블렌딩 후 Logistic Regression 캘리브레이션
  출력: data/submissions/oof_lgbcatxgb.csv
        data/submissions/submission.csv

STEP 2  BiLSTM  ────────────────────────────────────────────────────────
  MultiTaskBiLSTM (Bidirectional LSTM + Attention Pooling + Subject Embedding)
  · 시계열 lookback 14일 시퀀스
  · 5-Fold × 5-Seed OOF
  출력: data/oof_5fold_bilstm_multitask_safe.csv
        data/submission_5fold_bilstm_multitask_safe.csv

STEP 3  Blend  ─────────────────────────────────────────────────────────
  Nelder-Mead로 타겟별 최적 혼합 가중치 탐색 (OOF 기반)
  출력: data/submissions/submission_blend_lstm.csv  ← 최종 제출 파일
```

## 파일 구조

```
life_log/
├── main.py               # 전체 파이프라인 실행 진입점
├── config.py             # 경로 · 하이퍼파라미터 상수
│
├── src/
│   ├── utils.py          # 공통 유틸 (safe_mean/std/entropy, clip_probs, parse_hr …)
│   │
│   ├── features/
│   │   ├── tree.py       # Tree 모델용 피처 추출 (extract_*, build_tree_features)
│   │   └── lstm.py       # LSTM용 피처 추출 (agg_*, build_lstm_features)
│   │
│   ├── models/
│   │   └── bilstm.py     # MultiTaskBiLSTM, AttentionPooling, SleepSequenceDataset
│   │
│   └── trainers/
│       ├── tree.py       # Tree 앙상블 훈련 루프 → run()
│       ├── lstm.py       # BiLSTM 훈련 루프 → run()
│       └── blend.py      # OOF 기반 블렌딩 → run()
│
├── data/
│   ├── raw/              # 원본 parquet · csv (변경 금지)
│   └── submissions/      # 출력 파일 (자동 생성)
│
└── A.py / B.py / C.py    # 원본 단일 스크립트 (참고용)
```

## 사용 방법

### 전체 파이프라인 한 번에 실행

```bash
python main.py
```

### 단계별 실행

```bash
# STEP 1 only
python -c "from src.trainers import tree; tree.run()"

# STEP 2 only
python -c "from src.trainers import lstm; lstm.run()"

# STEP 3 only
python -c "from src.trainers import blend; blend.run()"
```

## 데이터 배치

`data/raw/` 아래에 다음 파일을 위치시킵니다.

| 파일 | 설명 |
|------|------|
| `ch2026_metrics_train.csv` | 학습 라벨 |
| `ch2026_submission_sample.csv` | 테스트 제출 양식 |
| `ch2025_mActivity.parquet` | 스마트폰 활동 |
| `ch2025_mACStatus.parquet` | 충전 상태 |
| `ch2025_mScreenStatus.parquet` | 화면 사용 |
| `ch2025_mLight.parquet` | 스마트폰 조도 |
| `ch2025_wLight.parquet` | 워치 조도 |
| `ch2025_wPedo.parquet` | 보행 계수 |
| `ch2025_wHr.parquet` | 심박수 |
| `ch2025_mUsageStats.parquet` | 앱 사용 통계 |
| `ch2025_mWifi.parquet` | Wi-Fi 스캔 |
| `ch2025_mBle.parquet` | Bluetooth 스캔 |
| `ch2025_mGps.parquet` | GPS |
| `ch2025_mAmbience.parquet` | 주변 소리 |

## 주요 의존성

```
pandas, numpy, scikit-learn, scipy
lightgbm, catboost, xgboost
torch
holidays
```

## 타겟 설명

| 타겟 | 의미 |
|------|------|
| Q1 | 수면 만족도 |
| Q2 | 수면 효율 |
| Q3 | 수면 지속성 |
| S1 | 수면 시작 어려움 |
| S2 | 수면 중 각성 |
| S3 | 조기 기상 |
| S4 | 주간 졸림 |
