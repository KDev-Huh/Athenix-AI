import numpy as np
import pandas as pd

from .situation_index import SituationIndex


class SituationFinder:
    """
    입력 좌표(선수 배열 + 공 + 캐리어)를 50차원 벡터로 변환하고,
    SituationIndex를 통해 유사한 과거 이벤트 top-k를 반환한다.

    결과는 VAEP 점수(vaep_value) 내림차순으로 정렬된다.

    사용 예시:
        finder = SituationFinder(idx)
        results = finder.find(
            players=[
                {"x": 90.0, "y": 35.0, "role": -1},
                {"x": 85.0, "y": 40.0, "role":  1},
                {"x": 110.0, "y": 36.0, "role": 2},
            ],
            ball_pos=(95.0, 38.0),
            carrier_pos=(92.0, 38.0),
            max_distance=60,
            top_k=5,
        )
    """

    _RESULT_COLS = [
        "event_id", "match_id", "type_name", "result_name",
        "start_x", "start_y", "end_x", "end_y", "vaep_value", "distance",
    ]

    ALLOWED_TYPES = {"pass", "shot", "dribble", "cross"}

    def __init__(self, index: SituationIndex) -> None:
        """
        Args:
            index: 빌드 또는 로드가 완료된 SituationIndex 인스턴스
        """
        self._idx = index

    # ── 공개 메서드 ────────────────────────────────────────────────────────

    def find(
        self,
        players:      list[dict],
        ball_pos:     tuple[float, float],
        carrier_pos:  tuple[float, float],
        max_distance: float = 60,
        top_k:        int   = 5,
    ) -> pd.DataFrame:
        """
        유사 상황 top-k를 반환한다.

        Args:
            players:      선수 배열. 각 항목은 {"x": float, "y": float, "role": int}
                          role: +1=팀메이트, -1=상대, +2=골키퍼
            ball_pos:     공 좌표 (x, y)
            carrier_pos:  볼 소유 선수 좌표 (x, y)
            max_distance: 허용 최대 유클리드 거리 (기본 60)
            top_k:        반환할 최대 결과 수 (기본 5)

        Returns:
            DataFrame — vaep_value 내림차순 정렬
            컬럼: event_id, match_id, type_name, result_name,
                  start_x, start_y, vaep_value, distance
            max_distance 이내 결과가 없으면 빈 DataFrame 반환
        """
        q = self._vectorize_query(players, ball_pos, carrier_pos)
        # type 필터링 후에도 top_k 확보를 위해 넉넉하게 후보 탐색
        n_candidates = min(top_k * 20, len(self._idx.meta))

        dists, js = self._idx.kneighbors(q, n_candidates)
        dists, js = dists[0], js[0]

        result = self._idx.meta.iloc[js].copy()
        result["distance"] = dists

        filtered = (
            result[
                (result["distance"] <= max_distance) &
                (result["type_name"].isin(self.ALLOWED_TYPES))
            ]
            .sort_values("vaep_value", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

        # 결과가 없을 경우에도 올바른 컬럼 구조 유지
        if filtered.empty:
            return pd.DataFrame(columns=self._RESULT_COLS)

        return filtered[self._RESULT_COLS]

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _vectorize_query(
        self,
        players:     list[dict],
        ball_pos:    tuple[float, float],
        carrier_pos: tuple[float, float],
    ) -> np.ndarray:
        """
        입력 좌표를 50차원 쿼리 벡터로 변환한다.

        SituationIndex의 벡터화 방식과 동일한 인코딩을 사용한다:
        - 캐리어 원점, 6시(정남) = 0°, 반시계 증가
        """
        cx, cy = carrier_pos
        v = np.zeros(SituationIndex.VECTOR_DIM, dtype=np.float32)

        # 공 상대위치
        v[0] = ball_pos[0] - cx
        v[1] = ball_pos[1] - cy

        # 선수별 각도 계산 후 정렬 → 슬롯 배정
        pts = sorted(
            [
                {
                    "dx":    p["x"] - cx,
                    "dy":    p["y"] - cy,
                    "angle": np.degrees(np.arctan2(p["x"] - cx, -(p["y"] - cy))) % 360,
                    "role":  p["role"],
                }
                for p in players
            ],
            key=lambda t: t["angle"],
        )

        for i, p in enumerate(pts[: SituationIndex.MAX_PLAYERS]):
            b = 2 + i * 3
            v[b], v[b + 1], v[b + 2] = p["dx"], p["dy"], p["role"]

        return v
