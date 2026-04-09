"""
VAEP 예측 스크립트

학습된 모델을 불러와 액션 데이터에 대한 VAEP 점수를 계산한다.
입력 데이터는 처리된 SPADL h5 파일에서 샘플을 가져오거나,
직접 DataFrame을 구성해서 넘길 수 있다.

실행 방법:
    python -m src.scripts.predict
"""

import pandas as pd
from src.recommendation.vaep import VAEPPredictor

# ── 설정 ──────────────────────────────────────────────────────────────

# 로드할 모델 파일명 (models/vaep/ 기준)
SCORES_MODEL   = "worldcup_2018_scores.json"
CONCEDES_MODEL = "worldcup_2018_concedes.json"
# config는 scores 모델명에서 자동 추론:
#   worldcup_2018_scores.json → worldcup_2018_config.json

# 모델 디렉토리
MODEL_DIR = "models/vaep"

# 예측에 사용할 SPADL 데이터 경로
SPADL_PATH = "data/vaep/processed/spadl-statsbomb.h5"

# 샘플로 가져올 게임 수 (None이면 전체)
SAMPLE_GAMES = 3

# ── 실행 ──────────────────────────────────────────────────────────────

def load_sample_actions(spadl_path: str, n_games: int = None) -> pd.DataFrame:
    """SPADL h5에서 샘플 액션 데이터를 로드한다."""
    with pd.HDFStore(spadl_path, mode="r") as store:
        games = store["games"]
        game_ids = games["game_id"].tolist()

        if n_games:
            game_ids = game_ids[:n_games]

        actions = pd.concat(
            [store[f"actions/game_{gid}"] for gid in game_ids]
        ).reset_index(drop=True)

    print(f"샘플 액션 로드: {len(game_ids)}경기 / {len(actions):,}개 액션")
    return actions


def run():
    # ── Step 1. 모델 로드 ─────────────────────────────────────────────
    print("\n[Step 1] 모델 로드")
    predictor = VAEPPredictor(model_dir=MODEL_DIR)
    predictor.load_models(SCORES_MODEL, CONCEDES_MODEL)

    # ── Step 2. 예측 데이터 준비 ──────────────────────────────────────
    print("\n[Step 2] 예측 데이터 준비")
    actions = load_sample_actions(SPADL_PATH, n_games=SAMPLE_GAMES)

    # ── Step 3. VAEP 점수 계산 ────────────────────────────────────────
    print("\n[Step 3] VAEP 점수 계산")
    result = predictor.predict(actions)

    # ── Step 4. 결과 출력 ─────────────────────────────────────────────
    print("\n[Step 4] 결과 요약")
    value_cols = ["offensive_value", "defensive_value", "vaep_value"]

    print(f"\n  총 액션 수: {len(result):,}")
    print(f"\n  VAEP 점수 통계:")
    print(result[value_cols].describe().round(4).to_string())

    print(f"\n  상위 10개 액션 (vaep_value 기준):")
    top10 = result.nlargest(10, "vaep_value")[
        ["player_id", "type_id", "start_x", "start_y", "end_x", "end_y"] + value_cols
    ]
    print(top10.to_string(index=False))

    return result


if __name__ == "__main__":
    run()
