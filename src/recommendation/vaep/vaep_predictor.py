import os

import pandas as pd
from xgboost import XGBClassifier
import socceraction.spadl as spadl
import socceraction.vaep.features as fs
import socceraction.vaep.formula as vaepformula

from src.modeling.vaep.config import VAEPConfig


class VAEPPredictor:
    """
    학습된 VAEP 모델을 불러와 액션(이벤트)의 VAEP 점수를 계산하는 클래스.

    모델 로드 시 함께 저장된 config를 자동으로 읽어
    VAEPModel과 피처/파라미터가 항상 동일하게 유지된다.

    단일 이벤트와 복수 이벤트 모두 지원한다.

    사용 예시:
        predictor = VAEPPredictor(model_dir="models/vaep")
        predictor.load_models("worldcup_2018_scores.json", "worldcup_2018_concedes.json")
        # → models/vaep/worldcup_2018_config.json 자동 로드

        result = predictor.predict(actions)
        # result 컬럼: 기존 액션 컬럼 + offensive_value, defensive_value, vaep_value
    """

    def __init__(self, model_dir: str = "models/vaep"):
        """
        Args:
            model_dir: 모델 파일이 저장된 기본 디렉토리
        """
        self.model_dir = model_dir
        self.models: dict[str, XGBClassifier] = {}
        self.config: VAEPConfig = None

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def load_models(self, scores_model: str, concedes_model: str, config_file: str = None) -> None:
        """
        모델 파일과 config를 로드한다.

        Args:
            scores_model:   scores 모델 파일명 또는 경로
                            예) "worldcup_2018_scores.json"
                                "experiments/v2/worldcup_scores.json"
            concedes_model: concedes 모델 파일명 또는 경로 (동일 규칙)
            config_file:    config 파일명 또는 경로. None이면 scores_model 기준 자동 추론.
                            예) "worldcup_2018_config.json"
        """
        scores_path   = self._resolve_path(scores_model)
        concedes_path = self._resolve_path(concedes_model)

        for label, path in [("scores", scores_path), ("concedes", concedes_path)]:
            clf = XGBClassifier()
            clf.load_model(path)
            self.models[label] = clf

        config_path = (
            self._resolve_path(config_file)
            if config_file
            else scores_path.replace("_scores.json", "_config.json")
        )
        self.config = VAEPConfig.load(config_path)

        print(f"Models loaded: {scores_model}, {concedes_model}")
        print(f"Config loaded: {os.path.basename(config_path)}")

    def predict(self, actions: pd.DataFrame) -> pd.DataFrame:
        """
        SPADL 형식의 액션 데이터에 대해 VAEP 점수를 계산한다.

        Args:
            actions: SPADL 표준 컬럼을 가진 DataFrame
                     필수 컬럼: game_id, period_id, time_seconds, team_id, player_id,
                                start_x, start_y, end_x, end_y,
                                type_id, result_id, bodypart_id

        Returns:
            입력 actions에 3개 컬럼이 추가된 DataFrame:
                - offensive_value: 공격적 기여도
                - defensive_value: 수비적 기여도
                - vaep_value:      종합 VAEP 점수 (offensive + defensive)
        """
        if not self.models or self.config is None:
            raise RuntimeError("먼저 load_models()를 호출해 주세요.")

        # type_name 등 이름 컬럼 추가 (피처 계산 및 VAEP 수식 모두 필요)
        actions = spadl.add_names(actions).reset_index(drop=True)

        features     = self._compute_features(actions)
        feature_cols = fs.feature_column_names(self.config.feature_functions, self.config.nb_prev_actions)

        scores_prob   = self.models["scores"].predict_proba(features[feature_cols])[:, 1]
        concedes_prob = self.models["concedes"].predict_proba(features[feature_cols])[:, 1]

        preds = pd.DataFrame({"scores": scores_prob, "concedes": concedes_prob})
        values = vaepformula.value(actions, preds["scores"], preds["concedes"])

        return pd.concat([actions, values], axis=1)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _resolve_path(self, file_name: str) -> str:
        """절대 경로면 그대로, 상대 경로면 model_dir를 prefix로 붙인다."""
        if os.path.isabs(file_name):
            return file_name
        return os.path.join(self.model_dir, file_name)

    def _compute_features(self, actions: pd.DataFrame) -> pd.DataFrame:
        """config의 feature_functions와 nb_prev_actions로 피처를 계산한다."""
        game_states = fs.gamestates(actions, self.config.nb_prev_actions)
        game_states = fs.play_left_to_right(game_states, actions["team_id"])

        feature_parts = [fn(game_states) for fn in self.config.feature_functions]
        return pd.concat(feature_parts, axis=1)
