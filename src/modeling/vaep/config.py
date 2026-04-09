import json
import socceraction.vaep.features as fs


# 피처 함수 이름 ↔ 함수 객체 매핑
FEATURE_REGISTRY = {
    "actiontype":        fs.actiontype,
    "actiontype_onehot": fs.actiontype_onehot,
    "bodypart":          fs.bodypart,
    "bodypart_onehot":   fs.bodypart_onehot,
    "result":            fs.result,
    "result_onehot":     fs.result_onehot,
    "goalscore":         fs.goalscore,
    "startlocation":     fs.startlocation,
    "endlocation":       fs.endlocation,
    "movement":          fs.movement,
    "space_delta":       fs.space_delta,
    "startpolar":        fs.startpolar,
    "endpolar":          fs.endpolar,
    "team":              fs.team,
    "time_delta":        fs.time_delta,
}

DEFAULT_FEATURE_NAMES = [
    "actiontype_onehot",
    "bodypart_onehot",
    "result_onehot",
    "goalscore",
    "startlocation",
    "endlocation",
    "movement",
    "space_delta",
    "startpolar",
    "endpolar",
    "team",
    "time_delta",
]

DEFAULT_XGB_PARAMS = dict(
    n_estimators=50,
    max_depth=3,
    n_jobs=-3,
    enable_categorical=True,
)

DEFAULT_NB_PREV_ACTIONS = 1


class VAEPConfig:
    """
    VAEPModel / VAEPPredictor 공유 설정 클래스.

    모델 저장 시 JSON으로 함께 저장되며, Predictor 로드 시 자동으로 읽어
    피처 구성과 파라미터가 항상 동일하게 유지된다.

    사용 예시:
        # 기본값 사용
        config = VAEPConfig()

        # 커스텀
        config = VAEPConfig(
            feature_names=["actiontype_onehot", "startlocation", "endlocation"],
            xgb_params={"n_estimators": 100, "max_depth": 5},
            nb_prev_actions=3,
        )
    """

    def __init__(
        self,
        feature_names: list[str] = None,
        xgb_params: dict = None,
        nb_prev_actions: int = None,
    ):
        """
        Args:
            feature_names:    사용할 피처 함수 이름 목록. None이면 기본값 사용.
                              사용 가능한 이름: FEATURE_REGISTRY 키 참고.
            xgb_params:       XGBoost 파라미터. None이면 기본값 사용.
                              일부만 지정하면 기본값에 덮어씌워진다.
            nb_prev_actions:  이전 액션 참조 수. None이면 기본값(1) 사용.
        """
        self.feature_names    = feature_names or DEFAULT_FEATURE_NAMES
        self.xgb_params       = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.nb_prev_actions  = nb_prev_actions if nb_prev_actions is not None else DEFAULT_NB_PREV_ACTIONS

        self._validate_feature_names()

    @property
    def feature_functions(self) -> list:
        """feature_names를 실제 함수 객체 목록으로 반환한다."""
        return [FEATURE_REGISTRY[name] for name in self.feature_names]

    def save(self, path: str) -> None:
        """설정을 JSON 파일로 저장한다."""
        data = {
            "feature_names":   self.feature_names,
            "xgb_params":      self.xgb_params,
            "nb_prev_actions": self.nb_prev_actions,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "VAEPConfig":
        """JSON 파일에서 설정을 복원한다."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def _validate_feature_names(self) -> None:
        unknown = [name for name in self.feature_names if name not in FEATURE_REGISTRY]
        if unknown:
            available = list(FEATURE_REGISTRY.keys())
            raise ValueError(f"알 수 없는 피처 이름: {unknown}\n사용 가능한 이름: {available}")
