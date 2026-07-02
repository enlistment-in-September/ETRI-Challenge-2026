from src.trainers import tree, lstm, blend

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1  Tree Ensemble  (LGB / CatBoost / XGBoost)")
    print("=" * 60)
    tree.run()

    print("\n" + "=" * 60)
    print("STEP 2  BiLSTM Multi-task")
    print("=" * 60)
    lstm.run()

    print("\n" + "=" * 60)
    print("STEP 3  Blend")
    print("=" * 60)
    blend.run()

    print("\nDone!  →  data/submissions/submission_blend_lstm.csv")
