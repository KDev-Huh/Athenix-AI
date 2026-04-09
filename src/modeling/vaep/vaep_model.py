import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import socceraction.vaep.features as fs
from .config import VAEPConfig


class VAEPModel:
    """
    VAEP 확률 추정 모델 (scores, concedes 이진 분류기).

    사용 예시:
        # 기본 설정으로 학습
        model = VAEPModel(data_dir="data/vaep", model_dir="models/vaep")
        X, y = model.load_training_data()
        model.train(X, y)
        print(model.evaluate(X, y))
        model.save(name="worldcup_2018")
        # → models/vaep/worldcup_2018_scores.json
        # → models/vaep/worldcup_2018_concedes.json
        # → models/vaep/worldcup_2018_config.json  ← 자동 저장

        # 커스텀 설정으로 학습
        from src.modeling.vaep.config import VAEPConfig
        config = VAEPConfig(
            feature_names=["actiontype_onehot", "startlocation", "endlocation"],
            xgb_params={"n_estimators": 100, "max_depth": 5},
            nb_prev_actions=3,
        )
        model = VAEPModel(config=config)
    """

    LABEL_COLUMNS = ["scores", "concedes"]

    def __init__(
        self,
        data_dir: str = "data/vaep",
        model_dir: str = "models/vaep",
        config: VAEPConfig = None,
    ):
        """
        Args:
            data_dir:  데이터 저장 루트 경로
            model_dir: 모델 저장 경로 (기본값: models/vaep)
            config:    VAEPConfig 인스턴스. None이면 기본 설정 사용.
        """
        self.processed_dir = os.path.join(data_dir, "processed")
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.config = config or VAEPConfig()
        self.models: dict[str, XGBClassifier] = {}

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def load_training_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        features.h5 / labels.h5를 읽어 X, y 행렬을 반환한다.
        config의 feature_names와 nb_prev_actions를 기준으로 컬럼을 선택한다.

        Returns:
            X: 피처 DataFrame
            y: 레이블 DataFrame (scores, concedes 컬럼)
        """
        feature_cols = fs.feature_column_names(
            self.config.feature_functions, self.config.nb_prev_actions
        )

        features_path = os.path.join(self.processed_dir, "features.h5")
        labels_path   = os.path.join(self.processed_dir, "labels.h5")

        all_features, all_labels = [], []

        with pd.HDFStore(features_path, mode="r") as f_store, \
             pd.HDFStore(labels_path,   mode="r") as l_store:

            for key in tqdm(f_store.keys(), desc="Loading training data"):
                all_features.append(f_store[key][feature_cols])
                all_labels.append(l_store[key][self.LABEL_COLUMNS])

        X = pd.concat(all_features).reset_index(drop=True)
        y = pd.concat(all_labels).reset_index(drop=True)

        return X, y

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        scores, concedes 각각 XGBClassifier를 학습한다.

        Args:
            X: 피처 DataFrame
            y: 레이블 DataFrame (scores, concedes 컬럼)
        """
        for label in tqdm(self.LABEL_COLUMNS, desc="Training models"):
            clf = XGBClassifier(**self.config.xgb_params)
            clf.fit(X, y[label])
            self.models[label] = clf

        print("Training complete.")

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict:
        """
        학습된 모델을 평가하고 지표를 반환한다.

        Returns:
            {
                "scores":   {"brier": ..., "log_loss": ..., "roc_auc": ..., "brier_skill": ...},
                "concedes": {...},
            }
        """
        if not self.models:
            raise RuntimeError("먼저 train() 또는 load()를 호출해 주세요.")

        results = {}
        for label in self.LABEL_COLUMNS:
            y_true = y[label]
            y_pred = self.models[label].predict_proba(X)[:, 1]  # (p(0), p(1)) 중 p(1) 확률

            baseline = y_true.mean()
            results[label] = {
                "brier":       round(brier_score_loss(y_true, y_pred), 5),
                "log_loss":    round(log_loss(y_true, y_pred), 5),
                "roc_auc":     round(roc_auc_score(y_true, y_pred), 4),
                "brier_skill": round(1 - brier_score_loss(y_true, y_pred) / brier_score_loss(y_true, [baseline] * len(y_true)), 4),
            }

        return results

    def save(self, name: str = None) -> None:
        """
        학습된 모델과 config를 함께 저장한다.

        Args:
            name: 파일 이름 접두어. None이면 타임스탬프로 자동 생성.
                  예) "worldcup_2018" → worldcup_2018_scores.json
                                        worldcup_2018_concedes.json
                                        worldcup_2018_config.json
        """
        if not self.models:
            raise RuntimeError("저장할 모델이 없습니다. 먼저 train()을 호출해 주세요.")

        prefix = name if name else f"vaep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for label, clf in self.models.items():
            path = os.path.join(self.model_dir, f"{prefix}_{label}.json")
            clf.save_model(path)
            print(f"Saved → {path}")

        config_path = os.path.join(self.model_dir, f"{prefix}_config.json")
        self.config.save(config_path)
        print(f"Saved → {config_path}")

    def load(self, scores_path: str, concedes_path: str, config_path: str = None) -> None:
        """
        저장된 모델과 config를 로드한다.

        Args:
            scores_path:   scores 모델 파일 경로
            concedes_path: concedes 모델 파일 경로
            config_path:   config JSON 파일 경로. None이면 scores_path 기준으로 자동 추론.
        """
        for label, path in [("scores", scores_path), ("concedes", concedes_path)]:
            clf = XGBClassifier()
            clf.load_model(path)
            self.models[label] = clf

        resolved_config_path = config_path or scores_path.replace("_scores.json", "_config.json")
        self.config = VAEPConfig.load(resolved_config_path)

        print("Models loaded.")
