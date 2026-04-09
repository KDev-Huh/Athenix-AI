"""
VAEP 전체 학습 파이프라인

실행 순서:
    1. StatsBomb 데이터 로드 및 SPADL 변환 → h5 저장
    2. 피처 / 레이블 생성 → h5 저장
    3. 모델 학습 및 평가
    4. 모델 저장

실행 방법:
    python -m src.scripts.train
"""

from src.preprocessing.vaep import DataPreprocessor
from src.features.vaep import FeatureBuilder
from src.modeling.vaep import VAEPModel, VAEPConfig

# ── 설정 ──────────────────────────────────────────────────────────────

# 학습할 대회 / 시즌 선택
SELECTED_COMPETITIONS = {
    "FIFA World Cup": [2018],
}

# 저장할 모델 이름 (None이면 타임스탬프 자동 생성)
MODEL_NAME = "worldcup_2018"

# 피처 생성 시 참조할 이전 액션 수 (이 값 이하로만 학습 가능)
NB_PREV_ACTIONS_FEATURES = 3

# 학습에 사용할 VAEPConfig (기본값 사용 시 None)
CONFIG = None
# CONFIG = VAEPConfig(
#     feature_names=["actiontype_onehot", "startlocation", "endlocation"],
#     xgb_params={"n_estimators": 100, "max_depth": 5},
#     nb_prev_actions=3,
# )

# 데이터 / 모델 경로
DATA_DIR  = "data/vaep"
MODEL_DIR = "models/vaep"

# ── 실행 ──────────────────────────────────────────────────────────────

def run():
    # ── Step 1. 데이터 로드 및 SPADL 변환 ────────────────────────────
    print("\n[Step 1] 데이터 로드 및 SPADL 변환")
    preprocessor = DataPreprocessor(data_dir=DATA_DIR, source="remote")

    print("사용 가능한 대회 목록:")
    competitions = preprocessor.load_competitions()
    print(competitions[["competition_name", "season_name"]].to_string(index=False))
    print()

    data = preprocessor.convert(SELECTED_COMPETITIONS)
    preprocessor.save(data, filename="spadl-statsbomb.h5")

    # ── Step 2. 피처 / 레이블 생성 ───────────────────────────────────
    print("\n[Step 2] 피처 / 레이블 생성")
    builder = FeatureBuilder(data_dir=DATA_DIR)
    builder.build_features(nb_prev_actions=NB_PREV_ACTIONS_FEATURES)
    builder.build_labels()

    # ── Step 3. 모델 학습 ─────────────────────────────────────────────
    print("\n[Step 3] 모델 학습")
    model = VAEPModel(data_dir=DATA_DIR, model_dir=MODEL_DIR, config=CONFIG)

    X, y = model.load_training_data()
    print(f"학습 데이터: {X.shape[0]:,}행 × {X.shape[1]}열")

    model.train(X, y)

    # ── Step 4. 모델 평가 ─────────────────────────────────────────────
    print("\n[Step 4] 모델 평가")
    results = model.evaluate(X, y)

    for label, metrics in results.items():
        print(f"\n  [{label}]")
        for metric_name, value in metrics.items():
            print(f"    {metric_name:<15}: {value}")

    # ── Step 5. 모델 저장 ─────────────────────────────────────────────
    print("\n[Step 5] 모델 저장")
    model.save(name=MODEL_NAME)

    print("\n학습 완료!")


if __name__ == "__main__":
    run()
