"""
SoccerSceneRecommender
======================
이미지 한 장을 입력받아 전체 파이프라인을 실행하고
가장 유사한 과거 축구 장면 top-k를 VAEP 내림차순으로 추천하는 클래스.

파이프라인:
    Image
      → DetectionPredictor  (선수/공/심판 BBox 검출)
      → KeypointsPredictor  (29개 필드 키포인트 검출)
      → TopViewAnalyzer     (호모그래피 → 경기장 좌표 + 팀 분류)
      → 좌표 변환            (FIFA meters → StatsBomb 0-120 × 0-80)
      → SituationFinder     (KNN 유사 장면 검색)
      → 시각화               (4-panel 피치 다이어그램)

사용 예시:
    from src.recommendation import SoccerSceneRecommender

    rec = SoccerSceneRecommender(
        detection_model_dir="models/detection",
        keypoints_model_dir="models/keypoints",
        knn_index_dir="models/knn_index",
        csv_path="data/StatsBombGithub/processed/vaep_360_merged.csv",
    )
    rec.load_models(
        detection_model_name="soccana_detection_v1.pt",
        keypoints_model_name="soccernet_keypoints_v1.pt",
    )

    output = rec.recommend(
        image_path="path/to/image.jpg",
        is_rtl=False,          # 공격 방향: False=왼→오, True=오→왼
        top_k=3,
        save_topview="topview_result.png",   # None이면 저장 안 함
        save_result="knn_result.png",        # None이면 저장 안 함
    )
    # output["results"]        : DataFrame — event_id, type_name, result_name, vaep_value, distance
    # output["query"]          : dict     — players, ball_pos, carrier_pos, carrier_team
    # output["topview_result"] : dict     — TopViewAnalyzer 원본 결과
"""

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .detection import DetectionPredictor
from .keypoints import KeypointsPredictor
from .knn import SituationFinder, SituationIndex
from .topview_analyzer import TopViewAnalyzer


class SoccerSceneRecommender:
    """
    이미지 → TopView → KNN 유사 장면 추천 통합 파이프라인 클래스.

    load_models() 호출 후 recommend()로 전체 파이프라인을 실행한다.
    """

    # StatsBomb 피치 크기
    _SB_W   = 120.0
    _SB_H   = 80.0
    # FIFA 피치 크기 (미터)
    _FIFA_W = 105.0
    _FIFA_H = 68.0

    # 선수 역할별 시각화 색상
    _COLOR = {1: "#1E90FF", -1: "#FF4444", 2: "#FFD700"}
    _BG    = "#1a1a2e"

    def __init__(
        self,
        detection_model_dir: str,
        keypoints_model_dir: str,
        knn_index_dir: str,
        csv_path: str,
    ) -> None:
        """
        Args:
            detection_model_dir: 검출 모델 디렉토리 경로
            keypoints_model_dir: 키포인트 모델 디렉토리 경로
            knn_index_dir:       KNN 인덱스 디렉토리 경로 (nn.pkl, vectors.npz, meta.csv)
            csv_path:            vaep_360_merged.csv 경로 (freeze_frame 시각화용)
        """
        self._det_model_dir = detection_model_dir
        self._kp_model_dir  = keypoints_model_dir
        self._index_dir     = knn_index_dir
        self._csv_path      = Path(csv_path)

        self._det        : Optional[DetectionPredictor] = None
        self._kp         : Optional[KeypointsPredictor] = None
        self._analyzer   : Optional[TopViewAnalyzer]    = None
        self._idx        : Optional[SituationIndex]     = None
        self._finder     : Optional[SituationFinder]    = None
        self._df         : Optional[pd.DataFrame]       = None
        self._zone_trees : Optional[dict]               = None  # (zone, role) → {tree, event_ids}

    # ── 공개 메서드 ────────────────────────────────────────────────────────

    def load_models(
        self,
        detection_model_name: str,
        keypoints_model_name: str,
        conf_det: float        = 0.4,
        conf_kp: float         = 0.5,
        min_kp_for_homography: int = 4,
    ) -> None:
        """
        모든 모델 및 KNN 인덱스를 로드한다.

        Args:
            detection_model_name:   검출 모델 파일명 (예: "soccana_detection_v1.pt")
            keypoints_model_name:   키포인트 모델 파일명 (예: "soccernet_keypoints_v1.pt")
            conf_det:               선수/공 검출 신뢰도 임계값 (기본 0.4)
            conf_kp:                키포인트 검출 신뢰도 임계값 (기본 0.5)
            min_kp_for_homography:  호모그래피 계산 최소 키포인트 수 (기본 4)
        """
        self._det = DetectionPredictor(model_dir=self._det_model_dir)
        self._det.load_model(detection_model_name)
        print(f"DetectionPredictor  loaded  ({detection_model_name})")

        self._kp = KeypointsPredictor(model_dir=self._kp_model_dir)
        self._kp.load_model(keypoints_model_name)
        print(f"KeypointsPredictor  loaded  ({keypoints_model_name})")

        self._analyzer = TopViewAnalyzer(
            detection_predictor=self._det,
            keypoints_predictor=self._kp,
            conf_det=conf_det,
            conf_kp=conf_kp,
            min_kp_for_homography=min_kp_for_homography,
        )
        print("TopViewAnalyzer     ready")

        self._idx    = SituationIndex()
        self._idx.load(self._index_dir)
        self._finder = SituationFinder(self._idx)
        print(f"SituationIndex      loaded  ({len(self._idx.meta):,} events)")

        self._df = pd.read_csv(self._csv_path)
        print(f"CSV                 loaded  ({len(self._df):,} rows)")

        self._zone_trees = self._build_zone_index(self._df)
        print(f"ZoneIndex           built   ({len(self._zone_trees)} zone-role trees)")

    def recommend(
        self,
        image_path: str,
        is_rtl: bool              = False,
        top_k: int                = 3,
        max_distance: float       = 80.0,
        team_color_threshold: float = 40.0,
        color_method: str         = "mean_l_norm",
        outlier_std_factor: float = 2.0,
        ball_retry_conf: float    = 0.2,
        save_topview: Optional[str] = None,
        save_result: Optional[str]  = None,
        show_plot: bool           = True,
    ) -> dict:
        """
        이미지 한 장을 분석해 유사한 과거 축구 장면 top-k를 추천하고 시각화한다.

        Args:
            image_path:             분석할 이미지 경로
            is_rtl:                 True면 RTL→LTR 좌표 플립 (오→왼 공격 시)
            top_k:                  추천할 유사 장면 수 (기본 3)
            max_distance:           KNN 최대 허용 유클리드 거리 (기본 80)
            team_color_threshold:   팀 분류 색상 임계값
            color_method:           팀 분류 색상 메서드
            outlier_std_factor:     아웃라이어 필터 기준 표준편차 계수
            ball_retry_conf:        공 미감지 시 재시도 신뢰도 임계값 (기본 0.2)
            save_topview:           TopView 시각화 저장 경로 (None이면 저장 안 함)
            save_result:            KNN 결과 시각화 저장 경로 (None이면 저장 안 함)
            show_plot:              True면 plt.show() 호출

        Returns:
            {
                "results":         pd.DataFrame   — top-k 유사 장면 (VAEP 내림차순)
                                                    컬럼: event_id, match_id, type_name,
                                                          result_name, start_x, start_y,
                                                          vaep_value, distance
                "query": {
                    "players":      list[dict],    — [{"x", "y", "role"}, ...]
                    "ball_pos":     (float, float), — StatsBomb 좌표
                    "carrier_pos":  (float, float), — StatsBomb 좌표
                    "carrier_team": str,            — "a" 또는 "b"
                }
                "topview_result":  dict            — TopViewAnalyzer 원본 결과
            }

        Raises:
            RuntimeError: load_models() 미호출 또는 호모그래피 실패
            ValueError:   선수 미감지
        """
        self._check_loaded()

        # Step 1: 이미지 분석 (TopViewAnalyzer + 공 재감지)
        topview_result = self._analyze_image(
            image_path,
            team_color_threshold=team_color_threshold,
            color_method=color_method,
            outlier_std_factor=outlier_std_factor,
            ball_retry_conf=ball_retry_conf,
        )
        fp = topview_result["field_positions"]

        if save_topview is not None:
            self._analyzer.visualize(image_path, topview_result, save_path=save_topview)
            print(f"TopView saved: {save_topview}")

        # Step 2: 좌표 변환 + 캐리어 결정 + role 배정
        players, ball_pos, carrier_pos, carrier_team = self._prepare_query(fp)

        # Step 3: 공격 방향 정규화 (RTL → LTR)
        if is_rtl:
            ball_pos, carrier_pos, players = self._flip_ltr(ball_pos, carrier_pos, players)
            print("RTL → LTR 좌표 플립 적용")
            print(f"  ball_pos    = ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
            print(f"  carrier_pos = ({carrier_pos[0]:.1f}, {carrier_pos[1]:.1f})")

        # Step 4: KNN 유사 장면 검색
        results = self._finder.find(
            players=players,
            ball_pos=ball_pos,
            carrier_pos=carrier_pos,
            max_distance=max_distance,
            top_k=top_k,
        )
        self._print_results(results, top_k)

        # Step 5: 4-panel 시각화
        query = {
            "players":      players,
            "ball_pos":     ball_pos,
            "carrier_pos":  carrier_pos,
            "carrier_team": carrier_team,
        }
        fig = self._visualize(query, results)
        if save_result is not None:
            fig.savefig(save_result, dpi=150, bbox_inches="tight", facecolor=self._BG)
            print(f"KNN result saved: {save_result}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return {
            "results":        results,
            "query":          query,
            "topview_result": topview_result,
        }

    def zone_recommend(
        self,
        image_path: str,
        is_rtl: bool                = False,
        top_k: int                  = 3,
        radius_rel: float           = 3.0,
        min_matches: int            = 3,
        team_color_threshold: float = 40.0,
        color_method: str           = "mean_l_norm",
        outlier_std_factor: float   = 2.0,
        ball_retry_conf: float      = 0.2,
        save_topview: Optional[str] = None,
        show_plot: bool             = True,
    ) -> dict:
        """
        Zone + Ball-Relative 방식으로 유사 장면을 추천한다.

        피치를 6구역으로 분할하고 공 기준 상대 좌표로 선수 패턴을 매칭하여
        같은 구역 내 절대 좌표가 유사한 이벤트를 반환한다.

        Args:
            image_path:   분석할 이미지 경로
            is_rtl:       True면 RTL→LTR 좌표 플립
            top_k:        추천 결과 수 (기본 3)
            radius_rel:   ball-relative 공간 탐색 반경 (기본 3.0 SB-units)
            min_matches:  최소 매칭 선수 수 (기본 3)
            ...           나머지는 recommend()와 동일

        Returns:
            recommend()와 동일한 구조.
            results DataFrame 컬럼: event_id, match_id, type_name, result_name,
                                    start_x, start_y, end_x, end_y, vaep_value, match_count
        """
        self._check_loaded()

        topview_result = self._analyze_image(
            image_path,
            team_color_threshold=team_color_threshold,
            color_method=color_method,
            outlier_std_factor=outlier_std_factor,
            ball_retry_conf=ball_retry_conf,
        )
        fp = topview_result["field_positions"]

        if save_topview is not None:
            self._analyzer.visualize(image_path, topview_result, save_path=save_topview)

        players, ball_pos, carrier_pos, carrier_team = self._prepare_query(fp)

        if is_rtl:
            ball_pos, carrier_pos, players = self._flip_ltr(ball_pos, carrier_pos, players)

        results = self._zone_search(players, ball_pos, top_k, radius_rel, min_matches)
        self._print_zone_results(results, top_k)

        query = {
            "players":      players,
            "ball_pos":     ball_pos,
            "carrier_pos":  carrier_pos,
            "carrier_team": carrier_team,
        }

        fig = self._visualize(query, results)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return {
            "results":        results,
            "query":          query,
            "topview_result": topview_result,
        }

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _check_loaded(self) -> None:
        if self._finder is None:
            raise RuntimeError("load_models()를 먼저 호출해 주세요.")

    # ── Zone + Ball-Relative 헬퍼 ──────────────────────────────────────────

    _ALLOWED_TYPES = {"pass", "shot", "dribble", "cross"}

    @staticmethod
    def _get_zone(x: float, y: float) -> tuple:
        """StatsBomb 좌표 → (col, row) 구역 번호. 3열×2행 = 6구역."""
        col = 0 if x < 40 else (1 if x < 80 else 2)
        row = 0 if y < 40 else 1
        return (col, row)

    @staticmethod
    def _build_zone_index(df: pd.DataFrame) -> dict:
        """
        CSV DataFrame으로 (zone, role) 별 cKDTree를 빌드한다.

        Returns:
            {(zone, role): {"tree": cKDTree, "event_ids": np.ndarray}}
        """
        print("ZoneIndex           building...")
        rows = []
        for _, row in df.iterrows():
            if pd.isna(row["freeze_frame"]) or pd.isna(row["start_x"]):
                continue
            bx, by = float(row["start_x"]), float(row["start_y"])
            zone   = SoccerSceneRecommender._get_zone(bx, by)
            eid    = row["event_id"]
            for p in json.loads(row["freeze_frame"]):
                rows.append({
                    "event_id": eid,
                    "rel_x":    p["x"] - bx,
                    "rel_y":    p["y"] - by,
                    "role":     int(p["role"]),
                    "zone":     zone,
                })

        df_rel = pd.DataFrame(rows)
        trees  = {}
        for (zone, role), sub in df_rel.groupby(["zone", "role"]):
            sub = sub.reset_index(drop=True)
            trees[(zone, role)] = {
                "tree":      cKDTree(sub[["rel_x", "rel_y"]].values),
                "event_ids": sub["event_id"].values,
            }
        return trees

    def _zone_search(
        self,
        players:     list,
        ball_pos:    tuple,
        top_k:       int,
        radius_rel:  float,
        min_matches: int,
    ) -> pd.DataFrame:
        """
        Zone + Ball-Relative 방식으로 유사 이벤트를 검색하고 결과 DataFrame을 반환한다.
        """
        query_zone   = self._get_zone(ball_pos[0], ball_pos[1])
        event_counter = Counter()

        for p in players:
            key = (query_zone, p["role"])
            if key not in self._zone_trees:
                continue
            td   = self._zone_trees[key]
            rel_x = p["x"] - ball_pos[0]
            rel_y = p["y"] - ball_pos[1]
            idxs  = td["tree"].query_ball_point([rel_x, rel_y], r=radius_rel)
            for eid in td["event_ids"][idxs]:
                event_counter[eid] += 1

        filtered_eids = {eid for eid, cnt in event_counter.items() if cnt >= min_matches}

        if not filtered_eids:
            print(f"  [ZoneSearch] 후보 없음 (radius_rel={radius_rel}, min_matches={min_matches})")
            return pd.DataFrame(columns=["event_id", "match_id", "type_name", "result_name",
                                         "start_x", "start_y", "end_x", "end_y",
                                         "vaep_value", "match_count"])

        meta = self._idx.meta[self._idx.meta["event_id"].isin(filtered_eids)].copy()
        meta["match_count"] = meta["event_id"].map(event_counter)

        return (
            meta[meta["type_name"].isin(self._ALLOWED_TYPES)]
            .sort_values(["match_count", "vaep_value"], ascending=[False, False])
            .head(top_k)
            .reset_index(drop=True)
        )

    def _print_zone_results(self, results: pd.DataFrame, top_k: int) -> None:
        if results.empty:
            print("유사 장면을 찾지 못했습니다.")
            return
        print(f"\nTop-{top_k} 추천 장면  (match_count 내림차순 → VAEP 내림차순):")
        print(f"  {'Rank':<5} {'Type':<16} {'Result':<12} {'Matches':>7} {'VAEP':>8}")
        print("  " + "-" * 55)
        for rank, (_, r) in enumerate(results.iterrows(), start=1):
            print(
                f"  {rank:<5} {r['type_name']:<16} {r['result_name']:<12} "
                f"{int(r['match_count']):>7} {r['vaep_value']:>8.4f}"
            )

    def _analyze_image(
        self,
        image_path: str,
        team_color_threshold: float,
        color_method: str,
        outlier_std_factor: float,
        ball_retry_conf: float,
    ) -> dict:
        """TopViewAnalyzer로 이미지를 분석하고, 공 미감지 시 낮은 conf로 재시도한다."""
        result = self._analyzer.analyze(
            image_path,
            team_color_threshold=team_color_threshold,
            color_method=color_method,
            outlier_std_factor=outlier_std_factor,
        )

        # 호모그래피 실패 — 필드 키포인트가 부족할 때 발생
        if result["homography"] is None:
            raise RuntimeError(
                "호모그래피 계산 실패: 필드 키포인트가 충분하지 않습니다.\n"
                f"  감지된 인라이어: {result['homography_inliers']} (최소 4 필요)\n"
                "  페널티 박스, 센터 서클 등 필드 마킹이 잘 보이는 이미지를 사용해 주세요."
            )

        fp = result["field_positions"]
        H  = result["homography"]

        # 공 재감지 — 팀 분류는 유지, 공만 낮은 conf로 재시도
        if not fp["ball"]:
            print(f"[Warning] Ball not detected at conf=0.4 — retrying at conf={ball_retry_conf} (ball only) ...")
            try:
                det_retry   = self._det.predict(image_path, conf=ball_retry_conf)
                balls_retry = [d for d in det_retry["detections"] if d["class_name"] == "Ball"]
                if balls_retry:
                    h_img = result["image_shape"]["height"]
                    w_img = result["image_shape"]["width"]
                    b     = balls_retry[0]
                    px    = b["x_center"] * w_img
                    py    = (b["y_center"] + b["height"] / 2) * h_img  # BBox 하단 = 공 바닥
                    pt    = cv2.perspectiveTransform(
                        np.array([[[px, py]]], dtype=np.float64), H
                    )[0][0]
                    fx, fy = float(pt[0]), float(pt[1])
                    if 0 <= fx <= self._FIFA_W and 0 <= fy <= self._FIFA_H:
                        fp["ball"] = [{"x": fx, "y": fy}]
                        print(f"  공 재감지 성공: ({fx:.1f}, {fy:.1f}) m")
                    else:
                        print("  공이 감지되었지만 필드 범위를 벗어남 — 무시")
                else:
                    print("  재시도 후에도 공 미감지")
            except Exception as e:
                warnings.warn(f"공 재감지 실패: {e}")

        print(f"Team A     : {len(fp['team_a'])} players")
        print(f"Team B     : {len(fp['team_b'])} players")
        print(f"Ball       : {len(fp['ball'])} detected")
        print(f"Referees   : {len(fp['referees'])}")
        print(f"HM inliers : {result['homography_inliers']}")

        return result

    def _prepare_query(self, fp: dict) -> tuple:
        """
        field_positions → StatsBomb 좌표 변환 + 캐리어 결정 + role 배정.

        Returns:
            (players, ball_pos, carrier_pos, carrier_team)
        """
        all_player_sb = []
        for team_key in ["team_a", "team_b"]:
            for p in fp[team_key]:
                all_player_sb.append(self._fifa_to_sb(p["x"], p["y"]))

        if not all_player_sb:
            raise ValueError(
                "선수가 감지되지 않았습니다. "
                "conf_det 값을 낮추거나 다른 이미지를 사용해 주세요."
            )

        # 공 위치 — 미감지 시 전체 선수 무게중심 fallback
        if fp["ball"]:
            ball_pos = self._fifa_to_sb(fp["ball"][0]["x"], fp["ball"][0]["y"])
            print(f"Ball pos   : ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})  [StatsBomb]")
        else:
            cx = sum(x for x, _ in all_player_sb) / len(all_player_sb)
            cy = sum(y for _, y in all_player_sb) / len(all_player_sb)
            ball_pos = (cx, cy)
            warnings.warn(
                "공 미감지 — 전체 선수 무게중심을 임시 공 위치로 사용합니다. "
                "정확한 추천을 위해 공이 보이는 이미지를 사용하세요."
            )

        # 캐리어: 공과 가장 가까운 선수
        min_dist, carrier_pos, carrier_team = float("inf"), None, None
        for team_key, team_label in [("team_a", "a"), ("team_b", "b")]:
            for p in fp[team_key]:
                xs, ys = self._fifa_to_sb(p["x"], p["y"])
                d = ((xs - ball_pos[0]) ** 2 + (ys - ball_pos[1]) ** 2) ** 0.5
                if d < min_dist:
                    min_dist, carrier_pos, carrier_team = d, (xs, ys), team_label

        print(
            f"Carrier    : team_{carrier_team}  "
            f"pos=({carrier_pos[0]:.1f}, {carrier_pos[1]:.1f})  "
            f"dist_to_ball={min_dist:.2f}"
        )

        # 나머지 선수 → role 배정 (캐리어 본인 제외)
        players = []
        for team_key, team_label in [("team_a", "a"), ("team_b", "b")]:
            role = 1 if team_label == carrier_team else -1
            for p in fp[team_key]:
                xs, ys = self._fifa_to_sb(p["x"], p["y"])
                if abs(xs - carrier_pos[0]) < 0.5 and abs(ys - carrier_pos[1]) < 0.5:
                    continue  # 캐리어 본인 제외
                players.append({"x": xs, "y": ys, "role": role})

        print(f"Players    : {len(players)}  (teammate=1, opponent=-1)")
        return players, ball_pos, carrier_pos, carrier_team

    def _flip_ltr(
        self,
        ball_pos:    tuple,
        carrier_pos: tuple,
        players:     list,
    ) -> tuple:
        """RTL → LTR 좌표 플립 (x = 120-x, y = 80-y)."""
        ball_pos    = (self._SB_W - ball_pos[0],    self._SB_H - ball_pos[1])
        carrier_pos = (self._SB_W - carrier_pos[0], self._SB_H - carrier_pos[1])
        players     = [
            {"x": self._SB_W - p["x"], "y": self._SB_H - p["y"], "role": p["role"]}
            for p in players
        ]
        return ball_pos, carrier_pos, players

    def _print_results(self, results: pd.DataFrame, top_k: int) -> None:
        if results.empty:
            print(
                "유사 장면을 찾지 못했습니다. "
                "max_distance를 늘리거나 is_rtl 설정을 확인해 주세요."
            )
            return
        print(f"\nTop-{top_k} 추천 장면  (VAEP 내림차순):")
        print(f"  {'Rank':<5} {'Type':<16} {'Result':<12} {'VAEP':>8} {'Dist':>8}")
        print("  " + "-" * 58)
        for rank, (_, r) in enumerate(results.iterrows(), start=1):
            print(
                f"  {rank:<5} {r['type_name']:<16} {r['result_name']:<12} "
                f"{r['vaep_value']:>8.4f} {r['distance']:>8.2f}"
            )

    # ── 시각화 ─────────────────────────────────────────────────────────────

    def _visualize(self, query: dict, results: pd.DataFrame) -> plt.Figure:
        """
        4-panel 피치 다이어그램을 생성하고 Figure를 반환한다.

        Panel 0   : Query (현재 이미지 상황, solid)
        Panel 1-3 : Top-1~3 추천 장면 (solid) + 쿼리 오버레이 (hollow)
        """
        players     = query["players"]
        ball_pos    = query["ball_pos"]
        carrier_pos = query["carrier_pos"]

        n_panels = 1 + min(len(results), 3)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
        if n_panels == 1:
            axes = [axes]
        fig.patch.set_facecolor(self._BG)

        # Panel 0 — Query
        self._draw_pitch(axes[0])
        self._draw_players(axes[0], players, ball_pos, carrier_pos, solid=True)
        axes[0].set_title("[Query]\n현재 이미지 상황", color="white", fontsize=11)

        # Panels 1-3 — 추천 장면 오버레이
        for rank, (_, r) in enumerate(results.iterrows(), start=1):
            ax = axes[rank]
            sim_players, sim_ball, sim_carrier = self._get_sim_frame(r["event_id"])
            self._draw_pitch(ax)
            self._draw_players(ax, players,     ball_pos,  carrier_pos, solid=False)  # 쿼리 (hollow)
            self._draw_players(ax, sim_players, sim_ball,  sim_carrier, solid=True)   # 추천 장면 (solid)
            subtitle = (
                f"matches={int(r['match_count'])}" if "match_count" in r.index
                else f"dist={r['distance']:.1f}"
            )
            ax.set_title(
                f"[Top-{rank}]  {subtitle}\n"
                f"{r['type_name']} / {r['result_name']}\n"
                f"VAEP = {r['vaep_value']:.4f}",
                color="white", fontsize=10,
            )

        legend_items = [
            mpatches.Patch(fc="#1E90FF", label="Teammate"),
            mpatches.Patch(fc="#FF4444", label="Opponent"),
            mpatches.Patch(fc="#FFD700", label="Goalkeeper"),
            plt.scatter([], [], c="white",   s=200, marker="*", label="Carrier"),
            plt.scatter([], [], c="#FFD700", s=80,  marker="o", label="Ball"),
            mpatches.Patch(fc="none", ec="gray", lw=2, label="Query (hollow)"),
            mpatches.Patch(fc="#888888",            label="Recommended (filled)"),
        ]
        fig.legend(
            handles=legend_items, loc="lower center", ncol=7,
            facecolor="#222", labelcolor="white", fontsize=9, framealpha=0.9,
        )
        plt.suptitle(
            f"Soccer Scene Recommendation  —  Top-{min(len(results), 3)} by VAEP",
            color="white", fontsize=13, y=1.01,
        )
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        return fig

    def _draw_pitch(self, ax) -> None:
        ax.set_facecolor("#4a7c4e")
        ax.set_xlim(0, self._SB_W)
        ax.set_ylim(0, self._SB_H)
        ax.set_aspect("equal")
        ax.axvline(60, color="white", lw=1, alpha=0.6)
        ax.add_patch(plt.Circle((60, 40), 9.15, fill=False, ec="white", lw=1, alpha=0.6))
        for x, y, w, h in [(102, 18, 18, 44), (0, 18, 18, 44)]:
            ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="white", lw=1, alpha=0.7))
        for x, y, w, h in [(118, 36, 2, 8), (0, 36, 2, 8)]:
            ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="yellow", lw=2, alpha=0.9))

    def _draw_players(
        self,
        ax,
        players:     list,
        ball_pos:    tuple,
        carrier_pos: tuple,
        solid:       bool = True,
    ) -> None:
        alpha = 1.0 if solid else 0.45
        for p in players:
            c = self._COLOR.get(p["role"], "#888888")
            if solid:
                ax.scatter(
                    p["x"], p["y"], c=c, s=150,
                    edgecolors="white", linewidths=0.7, alpha=alpha, zorder=4,
                )
            else:
                ax.scatter(
                    p["x"], p["y"], c="none", s=200,
                    edgecolors=c, linewidths=2.0, alpha=alpha, zorder=3,
                )
        # 캐리어 (별 마커)
        ax.scatter(
            *carrier_pos,
            c="white" if solid else "none", s=320, marker="*", zorder=5,
            edgecolors="black" if solid else "white", linewidths=1.2, alpha=alpha,
        )
        # 공
        ax.scatter(
            *ball_pos,
            c="#FFD700" if solid else "none", s=100, zorder=5,
            edgecolors="#FFD700", linewidths=1.5, alpha=alpha,
        )

    def _get_sim_frame(self, event_id: str) -> tuple:
        """CSV에서 추천 장면의 freeze_frame / ball_pos / carrier_pos를 조회한다."""
        row = self._df[self._df["event_id"] == event_id].iloc[0]
        return (
            json.loads(row["freeze_frame"]),
            (float(row["start_x"]),   float(row["start_y"])),
            (float(row["carrier_x"]), float(row["carrier_y"])),
        )

    # ── 좌표 변환 ─────────────────────────────────────────────────────────

    def _fifa_to_sb(self, x: float, y: float) -> tuple:
        """FIFA field coords (0-105 m, 0-68 m) → StatsBomb (0-120, 0-80)."""
        return x / self._FIFA_W * self._SB_W, y / self._FIFA_H * self._SB_H
