import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class SituationIndex:
    """
    vaep_360_merged.csv(이벤트당 1행 포맷)를 읽어 50차원 벡터를 빌드하고
    KNN 인덱스를 구성한다. 인덱스 저장/로드를 지원한다.

    벡터 구조 (50차원):
        [0:2]   → 공 상대위치 (start_x - carrier_x, start_y - carrier_y)
        [2:50]  → 선수 최대 16명 × (dx, dy, role), 6시 기준 반시계 각도 순 정렬
                  role: +1=팀메이트, -1=상대, +2=골키퍼

    사용 예시:
        idx = SituationIndex()
        idx.build("data/StatsBombGithub/processed/vaep_360_merged.csv")
        idx.save("models/knn_index")

        idx2 = SituationIndex()
        idx2.load("models/knn_index")
    """

    MAX_PLAYERS = 16
    VECTOR_DIM  = 2 + MAX_PLAYERS * 3   # 50

    _META_COLS = [
        "event_id", "match_id", "type_name", "result_name",
        "start_x", "start_y", "end_x", "end_y", "vaep_value",
    ]

    def __init__(self) -> None:
        self._nn      = None          # sklearn NearestNeighbors (피팅 후)
        self._vectors = None          # np.ndarray (N, 50), dtype=float32
        self.meta     = None          # pd.DataFrame — SituationFinder가 직접 참조

    # ── 공개 메서드 ────────────────────────────────────────────────────────

    def build(self, csv_path: str) -> None:
        """
        CSV를 읽어 벡터 행렬을 계산하고 KNN 인덱스를 빌드한다.

        Args:
            csv_path: vaep_360_merged.csv 경로 (이벤트당 1행 포맷)
        """
        df = pd.read_csv(csv_path)

        self._vectors = self._vectorize_all(df)
        self.meta     = df[self._META_COLS].reset_index(drop=True)
        self._nn      = NearestNeighbors(metric="euclidean", n_jobs=-1).fit(self._vectors)

        print(f"인덱스 빌드 완료: {len(self.meta):,}개 이벤트  벡터 shape={self._vectors.shape}")

    def save(self, dir_path: str) -> None:
        """
        빌드된 인덱스를 디렉토리에 저장한다.

        저장 파일:
            vectors.npz  — 벡터 행렬 (압축)
            meta.parquet — 메타 DataFrame
            nn.pkl       — NearestNeighbors 객체

        Args:
            dir_path: 저장 디렉토리 경로 (없으면 생성)
        """
        if self._nn is None:
            raise RuntimeError("먼저 build()를 호출해 주세요.")
        os.makedirs(dir_path, exist_ok=True)
        np.savez_compressed(os.path.join(dir_path, "vectors.npz"), vectors=self._vectors)
        self.meta.to_csv(os.path.join(dir_path, "meta.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(dir_path, "nn.pkl"), "wb") as f:
            pickle.dump(self._nn, f)
        print(f"저장 완료: {dir_path}")

    def load(self, dir_path: str) -> None:
        """
        save()가 저장한 인덱스를 로드한다.

        Args:
            dir_path: 저장 디렉토리 경로
        """
        self._vectors = np.load(os.path.join(dir_path, "vectors.npz"))["vectors"]
        self.meta     = pd.read_csv(os.path.join(dir_path, "meta.csv"))
        with open(os.path.join(dir_path, "nn.pkl"), "rb") as f:
            self._nn = pickle.load(f)
        print(f"로드 완료: {dir_path}  ({len(self.meta):,}개 이벤트)")

    def kneighbors(self, query: np.ndarray, n_neighbors: int):
        """
        쿼리 벡터에 대한 KNN 검색 결과를 반환한다.

        Args:
            query:       shape (1, 50) 또는 (50,) 쿼리 벡터
            n_neighbors: 반환할 이웃 수

        Returns:
            (distances, indices) — 각각 shape (1, n_neighbors)
        """
        if self._nn is None:
            raise RuntimeError("먼저 build() 또는 load()를 호출해 주세요.")
        return self._nn.kneighbors(query.reshape(1, -1), n_neighbors=n_neighbors)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _vectorize_all(self, df: pd.DataFrame) -> np.ndarray:
        """전체 이벤트를 (N, 50) float32 행렬로 변환한다."""
        n = len(df)
        V = np.zeros((n, self.VECTOR_DIM), dtype=np.float32)

        # 공 상대위치 (캐리어 기준)
        V[:, 0] = (df["start_x"].values   - df["carrier_x"].values).astype(np.float32)
        V[:, 1] = (df["start_y"].values   - df["carrier_y"].values).astype(np.float32)

        # 비캐리어 선수들을 freeze_frame JSON에서 파싱 → 슬롯 배정 → scatter-assign
        ei, dx, dy, slots, role = self._parse_players(df)
        base = 2 + slots * 3
        V[ei, base]     = dx
        V[ei, base + 1] = dy
        V[ei, base + 2] = role

        return V

    def _parse_players(self, df: pd.DataFrame):
        """
        freeze_frame JSON 컬럼을 파싱해 선수별 벡터 성분을 numpy 배열로 반환한다.

        Returns:
            (ei, dx, dy, slots, role) — 모두 numpy 배열, 길이 = 전체 선수 수
        """
        all_ei, all_dx, all_dy, all_angle, all_role = [], [], [], [], []

        carrier_x = df["carrier_x"].values
        carrier_y = df["carrier_y"].values
        ff_col    = df["freeze_frame"].values

        for i in range(len(df)):
            ff_str = ff_col[i]
            if pd.isna(ff_str):
                continue
            cx, cy = float(carrier_x[i]), float(carrier_y[i])
            for p in json.loads(ff_str):
                dx    = p["x"] - cx
                dy    = p["y"] - cy
                angle = np.degrees(np.arctan2(dx, -dy)) % 360
                all_ei.append(i)
                all_dx.append(dx)
                all_dy.append(dy)
                all_angle.append(angle)
                all_role.append(p["role"])

        if not all_ei:
            empty = np.array([], dtype=np.int32)
            return empty, empty, empty, empty, empty

        ei    = np.array(all_ei,    dtype=np.int32)
        dx    = np.array(all_dx,    dtype=np.float32)
        dy    = np.array(all_dy,    dtype=np.float32)
        angle = np.array(all_angle, dtype=np.float32)
        role  = np.array(all_role,  dtype=np.float32)

        # 각도 기준 슬롯 번호 계산 (이벤트 내 rank, 0-indexed)
        pf = pd.DataFrame({"ei": ei, "angle": angle})
        pf["slot"] = pf.groupby("ei")["angle"].rank(method="first").astype(int) - 1

        mask  = pf["slot"].values < self.MAX_PLAYERS
        slots = pf["slot"].values[mask].astype(np.int32)

        return ei[mask], dx[mask], dy[mask], slots, role[mask]
