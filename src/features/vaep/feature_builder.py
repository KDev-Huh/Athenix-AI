import os
import pandas as pd
from tqdm import tqdm
import socceraction.spadl as spadl
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab


class FeatureBuilder:
    """
    SPADL H5 데이터를 읽어 피처와 레이블을 생성하고 H5로 저장하는 클래스.

    사용 예시:
        builder = FeatureBuilder(data_dir="data/vaep")
        builder.build_features(nb_prev_actions=3)
        builder.build_labels()
    """

    # 노트북 2번과 동일한 16개 피처 함수
    FEATURE_FUNCTIONS = [
        fs.actiontype,
        fs.actiontype_onehot,
        fs.bodypart,
        fs.bodypart_onehot,
        fs.result,
        fs.result_onehot,
        fs.goalscore,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
        fs.startpolar,
        fs.endpolar,
        fs.team,
        fs.time_delta,
    ]

    # 노트북 2번과 동일한 3개 레이블 함수
    LABEL_FUNCTIONS = [
        lab.scores,
        lab.concedes,
        lab.goal_from_shot,
    ]

    def __init__(self, data_dir: str = "data/vaep", spadl_file: str = "spadl-statsbomb.h5"):
        """
        Args:
            data_dir:   데이터 저장 루트 경로
            spadl_file: processed/ 폴더 내 SPADL H5 파일명
        """
        self.processed_dir = os.path.join(data_dir, "processed")
        self.spadl_path = os.path.join(self.processed_dir, spadl_file)

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def build_features(self, nb_prev_actions: int = 3) -> None:
        """
        게임별 피처를 계산하고 features.h5에 저장한다.

        Args:
            nb_prev_actions: 게임 상태 구성 시 참조할 이전 액션 수 (기본값 3)
        """
        game_ids = self._load_game_ids()
        output_path = os.path.join(self.processed_dir, "features.h5")

        with pd.HDFStore(self.spadl_path, mode="r") as spadl_store, \
             pd.HDFStore(output_path, mode="w") as feature_store:

            for game_id in tqdm(game_ids, desc="Building features"):
                actions = spadl_store[f"actions/game_{game_id}"]
                features = self._compute_features(actions, nb_prev_actions)
                feature_store[f"game_{game_id}"] = self._to_storable(features)

        print(f"Saved → {output_path}")

    def build_labels(self) -> None:
        """
        게임별 레이블을 계산하고 labels.h5에 저장한다.
        레이블 컬럼: scores, concedes, goal_from_shot
        """
        game_ids = self._load_game_ids()
        output_path = os.path.join(self.processed_dir, "labels.h5")

        with pd.HDFStore(self.spadl_path, mode="r") as spadl_store, \
             pd.HDFStore(output_path, mode="w") as label_store:

            for game_id in tqdm(game_ids, desc="Building labels"):
                actions = spadl_store[f"actions/game_{game_id}"]
                labels = self._compute_labels(actions)
                label_store[f"game_{game_id}"] = labels

        print(f"Saved → {output_path}")

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _load_game_ids(self) -> list:
        """SPADL H5에서 게임 ID 목록을 읽어온다."""
        with pd.HDFStore(self.spadl_path, mode="r") as store:
            games = store["games"]
        return games["game_id"].tolist()

    def _compute_features(self, actions: pd.DataFrame, nb_prev_actions: int) -> pd.DataFrame:
        """단일 게임의 피처를 계산한다."""
        # type_id → type_name 등 이름 컬럼 추가 (goalscore 피처 함수가 type_name 필요)
        actions = spadl.add_names(actions)
        # 게임 상태 구성 (현재 + 이전 nb_prev_actions 개 액션)
        game_states = fs.gamestates(actions, nb_prev_actions)
        # 모든 팀이 왼쪽 → 오른쪽 방향으로 공격하도록 좌표 정규화
        game_states = fs.play_left_to_right(game_states, actions["team_id"])

        feature_parts = [fn(game_states) for fn in self.FEATURE_FUNCTIONS]
        return pd.concat(feature_parts, axis=1)

    def _compute_labels(self, actions: pd.DataFrame) -> pd.DataFrame:
        """단일 게임의 레이블을 계산한다."""
        actions = spadl.add_names(actions)
        label_parts = [fn(actions) for fn in self.LABEL_FUNCTIONS]
        return pd.concat(label_parts, axis=1)

    def _to_storable(self, df: pd.DataFrame) -> pd.DataFrame:
        """category 타입 컬럼을 HDF5 저장 가능한 타입으로 변환한다."""
        cat_cols = df.select_dtypes(include="category").columns
        df = df.copy()
        df[cat_cols] = df[cat_cols].astype(str)
        return df
