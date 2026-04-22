"""
Detection + Keypoints + Homography 를 결합해 Top-View 경기장 좌표를 생성하는 클래스.

DetectionPredictor / KeypointsPredictor 를 의존성으로 주입받아
Raw YOLO 없이 기존 클래스만으로 전체 파이프라인을 실행한다.
팀 구분은 유니폼 색상 K-Means 자동 군집화로 처리한다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.recommendation.detection.detection_predictor import DetectionPredictor
from src.recommendation.keypoints.keypoints_predictor import KeypointsPredictor


# ── FIFA 표준 필드 치수 ────────────────────────────────────────────
_FL, _FW       = 105.0, 68.0          # 필드 길이, 너비 (m)
_BBL, _BBW     = 16.5,  40.32         # 패널티 에어리어
_SBL, _SBW     = 5.5,   18.32         # 골 에어리어
_CCR           = 9.15                  # 센터 서클 반경
_SPOT          = 11.0                  # 패널티 스팟 거리
_CX, _CY       = _FL / 2, _FW / 2     # 필드 중심
_BBT           = (_FW - _BBW) / 2     # 패널티 에어리어 상단 y
_SBT           = (_FW - _SBW) / 2     # 골 에어리어 상단 y


class TopViewAnalyzer:
    """
    축구 경기 이미지에서 Detection + Keypoints 추론 결과를 결합해
    Top-View 경기장 좌표로 변환하고 시각화하는 클래스.

    - DetectionPredictor : 선수 · 공 · 심판 BBox 검출
    - KeypointsPredictor : 29개 필드 키포인트 검출
    - Homography         : 이미지 픽셀 → 경기장 미터(m) 좌표 변환
    - 팀 구분            : 유니폼 상체 색상 K-Means(k=2) 자동 군집화

    사용 예시:
        det = DetectionPredictor("models/detection")
        det.load_model("soccana_detection_v1.pt")

        kp = KeypointsPredictor("models/keypoints")
        kp.load_model("soccernet_keypoints_v1.pt")

        analyzer = TopViewAnalyzer(det, kp)
        result   = analyzer.analyze("match_frame.jpg")
        analyzer.visualize("match_frame.jpg", result, save_path="topview.png")
    """

    # ── 클래스 상수 ───────────────────────────────────────────────
    FIELD_LENGTH: float = _FL
    FIELD_WIDTH:  float = _FW

    # 29개 키포인트 실제 경기장 좌표 (m)
    # 원점 = 필드 왼쪽 위 코너, x: 왼골라인→오른골라인, y: 위터치라인→아래터치라인
    KP_WORLD: Dict[int, Tuple[float, float]] = {
         0: (0,             0),
         1: (0,             _BBT),
         2: (_BBL,          _BBT),
         3: (0,             _FW - _BBT),
         4: (_BBL,          _FW - _BBT),
         5: (0,             _SBT),
         6: (_SBL,          _SBT),
         7: (0,             _FW - _SBT),
         8: (_SBL,          _FW - _SBT),
         9: (0,             _FW),
        10: (_SPOT + _CCR,  _CY),
        11: (_CX,           0),
        12: (_CX,           _FW),
        13: (_CX,           _CY - _CCR),
        14: (_CX,           _CY + _CCR),
        15: (_CX,           _CY),
        16: (_FL,           0),
        17: (_FL,           _BBT),
        18: (_FL - _BBL,    _BBT),
        19: (_FL,           _FW - _BBT),
        20: (_FL - _BBL,    _FW - _BBT),
        21: (_FL,           _SBT),
        22: (_FL - _SBL,    _SBT),
        23: (_FL,           _FW - _SBT),
        24: (_FL - _SBL,    _FW - _SBT),
        25: (_FL,           _FW),
        26: (_FL - _SPOT - _CCR, _CY),
        27: (_CX - _CCR,    _CY),
        28: (_CX + _CCR,    _CY),
    }

    # 팀별 시각화 색상 (matplotlib)
    _TEAM_COLORS = ["#1E90FF", "#FF8C00"]   # team_id 0: 파랑, 1: 주황
    _TEAM_COLORS_BGR = [(215, 100, 0), (0, 140, 255)]  # OpenCV BBoxoverlay용 BGR
    _BALL_COLOR    = "#FFD700"
    _REF_COLOR     = "#FF4444"
    _KP_COLOR      = "#00FF50"

    # 잔디 마스킹용 HSV 범위 (잔디 픽셀 제거 후 유니폼 색상 추출)
    _GRASS_LOWER = np.array([35, 40,  40], dtype=np.uint8)   # HSV 하한 (녹색 계열)
    _GRASS_UPPER = np.array([85, 255, 255], dtype=np.uint8)  # HSV 상한

    def __init__(
        self,
        detection_predictor: DetectionPredictor,
        keypoints_predictor: KeypointsPredictor,
        conf_det: float = 0.4,
        conf_kp:  float = 0.5,
        min_kp_for_homography: int = 4,
    ):
        """
        Args:
            detection_predictor:    load_model() 완료된 DetectionPredictor 인스턴스.
            keypoints_predictor:    load_model() 완료된 KeypointsPredictor 인스턴스.
            conf_det:               Detection 신뢰도 임계값 (기본값 0.4).
            conf_kp:                Keypoints 신뢰도 임계값 (기본값 0.5).
            min_kp_for_homography:  호모그래피 계산에 필요한 최소 키포인트 수 (기본값 4).
        """
        self.detection_predictor   = detection_predictor
        self.keypoints_predictor   = keypoints_predictor
        self.conf_det              = conf_det
        self.conf_kp               = conf_kp
        self.min_kp_for_homography = min_kp_for_homography

    # ── 공개 메서드 ───────────────────────────────────────────────

    def analyze(
        self,
        image_path:           str,
        team_color_threshold: Optional[float] = None,
        color_method:         str = "mean_l_norm",
        outlier_std_factor:   Optional[float] = 2.0,
    ) -> Dict:
        """
        색상 지정 없이 K-Means 자동 팀 분류로 전체 파이프라인을 실행한다.

        팀 색상을 모를 때 사용. 유니폼 평균 LAB 색상을 K-Means(k=2) 로
        자동 군집화해 두 팀을 구분한다. 클러스터 내 아웃라이어(골키퍼·오탐 심판 등)는
        자동으로 제거해 team_id = -1 로 처리한다.

        Args:
            image_path:           분석할 이미지 파일 경로.
            team_color_threshold: LAB 거리 절대 임계값 (선택).
                                  클러스터 중심까지 거리가 이 값 초과 시 team_id = -1.
                                  None 이면 비활성화.
                                  LAB 거리 참고: 10=약간 다름, 30=확실히 다름, 50=매우 다름
            outlier_std_factor:   클러스터 내 상대 아웃라이어 제거 강도 (기본 2.0).
                                  각 클러스터에서 거리 > mean + factor×std 인 선수를 -1 처리.
                                  골키퍼·오탐 심판 제거에 유효. None 이면 비활성화.

        Returns:
            {
                "image_path":          str,
                "image_shape":         {"height": int, "width": int},
                "detections":          [...],
                "keypoints":           [...],
                "valid_kps":           [...],
                "homography":          np.ndarray or None,
                "homography_inliers":  int,
                "field_positions": {
                    "team_a":   [{"x": float, "y": float}, ...],
                    "team_b":   [{"x": float, "y": float}, ...],
                    "ball":     [{"x": float, "y": float}],
                    "referees": [{"x": float, "y": float}, ...],
                    "unknown":  [{"x": float, "y": float}, ...],
                },
            }
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

        H_IMG, W_IMG = image.shape[:2]
        image_shape  = (H_IMG, W_IMG)

        det_result = self.detection_predictor.predict(image_path, conf=self.conf_det)
        detections = det_result["detections"]

        kp_result  = self.keypoints_predictor.predict(image_path, conf=self.conf_kp)
        keypoints  = kp_result["keypoints"]
        valid_kps  = [
            k for k in keypoints
            if k.get("conf", 0) >= self.conf_kp and not (k["x"] == 0 and k["y"] == 0)
        ]

        H_mat, n_inliers = self._compute_homography(valid_kps, image_shape)

        if H_mat is not None:
            detections = self._transform_to_field(detections, H_mat, image_shape)
        else:
            for det in detections:
                det["fx"] = None; det["fy"] = None; det["in_field"] = False

        team_ids = self._classify_teams_auto(
            image, detections, team_color_threshold, color_method, outlier_std_factor,
        )
        for det, tid in zip(detections, team_ids):
            det["team_id"] = tid

        field_positions = self._build_field_positions(detections)

        return {
            "image_path":         image_path,
            "image_shape":        {"height": H_IMG, "width": W_IMG},
            "detections":         detections,
            "keypoints":          keypoints,
            "valid_kps":          valid_kps,
            "homography":         H_mat,
            "homography_inliers": n_inliers,
            "field_positions":    field_positions,
        }


    def visualize(
        self,
        image_path: str,
        result:     Dict,
        save_path:  Optional[str] = None,
    ) -> None:
        """
        analyze() 결과를 2-panel 시각화한다.

        왼쪽 패널:  원본 이미지 + BBox(팀 색) + Keypoints overlay
        오른쪽 패널: Top-View 경기장 + 선수/공/심판 위치 scatter

        Args:
            image_path: 원본 이미지 파일 경로.
            result:     analyze() 반환값.
            save_path:  저장 경로. None이면 저장 없이 plt.show() 만 호출.
        """
        import matplotlib.pyplot as plt

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

        H_IMG  = result["image_shape"]["height"]
        W_IMG  = result["image_shape"]["width"]
        dets   = result["detections"]
        v_kps  = result["valid_kps"]

        # ── overlay 이미지 생성 ───────────────────────────────────
        vis = image.copy()
        for det in dets:
            cx, cy, w, h = det["x_center"], det["y_center"], det["width"], det["height"]
            x1 = int((cx - w / 2) * W_IMG);  y1 = int((cy - h / 2) * H_IMG)
            x2 = int((cx + w / 2) * W_IMG);  y2 = int((cy + h / 2) * H_IMG)
            foot_x = int(cx * W_IMG);         foot_y = y2

            if det["class_id"] == 0:          # Player
                tid  = det.get("team_id", 0)
                color_bgr = self._TEAM_COLORS_BGR[tid] if tid in (0, 1) else (180, 180, 180)
            elif det["class_id"] == 1:        # Ball
                color_bgr = (0, 215, 255)     # 노란색 BGR
            else:                             # Referee
                color_bgr = (50, 50, 220)     # 빨간색 BGR

            cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.circle(vis, (foot_x, foot_y), 5, color_bgr, -1)

        for k in v_kps:
            px, py = int(k["x"] * W_IMG), int(k["y"] * H_IMG)
            cv2.circle(vis, (px, py), 7, (0, 255, 80), -1)
            cv2.putText(vis, str(k["idx"]), (px + 8, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # ── matplotlib figure ─────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(22, 8),
            gridspec_kw={"width_ratios": [1.5, 1]},
        )
        fig.patch.set_facecolor("#111")

        # 왼쪽: 원본 + overlay
        ax1.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax1.set_title("Detection + Keypoints (팀 색 구분)", color="w", fontsize=12)
        ax1.axis("off")

        # 오른쪽: Top-View
        self._draw_pitch(ax2)
        in_field = [d for d in dets if d.get("in_field")]

        for tid, color, label in [(0, self._TEAM_COLORS[0], "Team A"),
                                   (1, self._TEAM_COLORS[1], "Team B")]:
            pts = [d for d in in_field if d["class_id"] == 0 and d.get("team_id") == tid]
            if pts:
                ax2.scatter(
                    [p["fx"] for p in pts], [p["fy"] for p in pts],
                    s=60, c=color, edgecolors="w", linewidths=0.6,
                    marker="o", zorder=5, label=f"{label} ({len(pts)})",
                )

        unknown = [d for d in in_field if d["class_id"] == 0 and d.get("team_id") == -1]
        if unknown:
            ax2.scatter(
                [p["fx"] for p in unknown], [p["fy"] for p in unknown],
                s=60, c="#AAAAAA", edgecolors="w", linewidths=0.6,
                marker="o", zorder=5, label=f"Unknown ({len(unknown)})",
            )

        balls = [d for d in in_field if d["class_id"] == 1]
        if balls:
            ax2.scatter(
                [b["fx"] for b in balls], [b["fy"] for b in balls],
                s=120, c=self._BALL_COLOR, edgecolors="w", linewidths=0.6,
                marker="*", zorder=6, label=f"Ball ({len(balls)})",
            )

        refs = [d for d in in_field if d["class_id"] == 2]
        if refs:
            ax2.scatter(
                [r["fx"] for r in refs], [r["fy"] for r in refs],
                s=80, c=self._REF_COLOR, edgecolors="w", linewidths=0.6,
                marker="^", zorder=5, label=f"Referee ({len(refs)})",
            )

        for k in v_kps:
            if k["idx"] in self.KP_WORLD:
                wx, wy = self.KP_WORLD[k["idx"]]
                ax2.plot(wx, wy, "*", color="w", ms=7,
                         markeredgecolor="k", markeredgewidth=0.4, zorder=6)

        n_kp  = len(v_kps)
        n_inl = result.get("homography_inliers", 0)
        ax2.legend(loc="upper right", framealpha=0.8, facecolor="#222",
                   labelcolor="w", fontsize=10)
        ax2.set_title(
            f"Top-View  (★ 키포인트 {n_kp}개, RANSAC 인라이어 {n_inl}개)",
            color="w", fontsize=12,
        )
        ax2.set_xlabel("m", color="w")
        ax2.set_ylabel("m", color="w")
        ax2.tick_params(colors="w")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"저장 완료 → {save_path}")

        plt.show()

    # ── 비공개 메서드 ─────────────────────────────────────────────

    def _build_field_positions(self, detections: List[Dict]) -> Dict:
        """
        detections 에서 경기장 내(in_field=True) 좌표만 추출해
        클래스/팀별로 정리한 딕셔너리를 반환한다.

        Returns:
            {
                "team_a":   [{"x": float, "y": float}, ...],  # team_id == 0
                "team_b":   [{"x": float, "y": float}, ...],  # team_id == 1
                "ball":     [{"x": float, "y": float}],
                "referees": [{"x": float, "y": float}, ...],
            }
        """
        result: Dict[str, List] = {
            "team_a": [], "team_b": [], "ball": [], "referees": [], "unknown": [],
        }

        for det in detections:
            if not det.get("in_field") or det["fx"] is None:
                continue

            pos = {"x": det["fx"], "y": det["fy"]}

            if det["class_id"] == 0:          # Player
                tid = det.get("team_id", -1)
                if tid == 0:
                    result["team_a"].append(pos)
                elif tid == 1:
                    result["team_b"].append(pos)
                else:                          # -1: 임계값 초과 미분류
                    result["unknown"].append(pos)
            elif det["class_id"] == 1:        # Ball
                result["ball"].append(pos)
            elif det["class_id"] == 2:        # Referee
                result["referees"].append(pos)

        return result

    def _compute_homography(
        self,
        valid_kps:   List[Dict],
        image_shape: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        유효 키포인트에서 이미지 픽셀 → 경기장 미터(m) 호모그래피 행렬을 계산한다.

        Args:
            valid_kps:   conf 필터를 통과한 키포인트 목록.
            image_shape: (H, W) 이미지 크기 (픽셀).

        Returns:
            (H_mat, n_inliers)
            H_mat:      3×3 호모그래피 행렬. 키포인트 부족 시 None.
            n_inliers:  RANSAC 인라이어 수.
        """
        H_IMG, W_IMG = image_shape

        src, dst = [], []
        for k in valid_kps:
            if k["idx"] in self.KP_WORLD:
                src.append([k["x"] * W_IMG, k["y"] * H_IMG])
                dst.append(list(self.KP_WORLD[k["idx"]]))

        if len(src) < self.min_kp_for_homography:
            print(f"[경고] 호모그래피 계산 불가: 유효 키포인트 {len(src)}개 "
                  f"(최소 {self.min_kp_for_homography}개 필요)")
            return None, 0

        H_mat, mask = cv2.findHomography(
            np.float32(src).reshape(-1, 1, 2),
            np.float32(dst).reshape(-1, 1, 2),
            cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
        n_inliers = int(mask.sum()) if mask is not None else 0
        return H_mat, n_inliers

    @staticmethod
    def sb_to_pixel(
        x_sb: float,
        y_sb: float,
        H_mat: np.ndarray,
        image_shape: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        StatsBomb 좌표 → 원본 이미지 픽셀 좌표 역변환.

        StatsBomb (0~120, 0~80) → FIFA (0~105m, 0~68m) → 역호모그래피 → 픽셀

        Args:
            x_sb:        StatsBomb x 좌표 (0~120)
            y_sb:        StatsBomb y 좌표 (0~80)
            H_mat:       _compute_homography()가 반환한 3×3 행렬 (픽셀→FIFA 방향)
            image_shape: {"height": int, "width": int} 이미지 크기.
                         전달 시 픽셀 좌표를 이미지 범위 내로 클램핑한다.

        Returns:
            {"x": float, "y": float} 픽셀 좌표, H_mat이 None이면 None
        """
        if H_mat is None:
            return None
        x_fifa = x_sb * 105.0 / 120.0
        y_fifa = y_sb * 68.0  / 80.0
        H_inv  = np.linalg.inv(H_mat)
        pt     = cv2.perspectiveTransform(
            np.array([[[x_fifa, y_fifa]]], dtype=np.float32), H_inv
        )
        px, py = float(pt[0][0][0]), float(pt[0][0][1])

        print(f"[sb_to_pixel] sb=({x_sb}, {y_sb}) -> fifa=({x_fifa:.2f}, {y_fifa:.2f}) -> pixel=({px:.1f}, {py:.1f})")

        if image_shape is not None:
            w, h = float(image_shape["width"]), float(image_shape["height"])
            px = max(0.0, min(px, w))
            py = max(0.0, min(py, h))
            print(f"[sb_to_pixel] clamped=({px:.1f}, {py:.1f})  image=({int(w)}x{int(h)})")

        return {"x": round(px, 1), "y": round(py, 1)}

    def _transform_to_field(
        self,
        detections:  List[Dict],
        H_mat:       np.ndarray,
        image_shape: Tuple[int, int],
    ) -> List[Dict]:
        """
        각 객체의 발 좌표(BBox 하단 중심)를 호모그래피로 경기장 좌표로 변환한다.

        Args:
            detections:  analyze() 에서 수집된 detection 딕셔너리 목록.
            H_mat:       3×3 호모그래피 행렬.
            image_shape: (H, W) 이미지 크기 (픽셀).

        Returns:
            각 detection 에 fx, fy, in_field 키가 추가된 목록.
        """
        H_IMG, W_IMG = image_shape

        foot_pixels = np.float32([
            [d["x_center"] * W_IMG, (d["y_center"] + d["height"] / 2) * H_IMG]
            for d in detections
        ]).reshape(-1, 1, 2)

        field_pts = cv2.perspectiveTransform(foot_pixels, H_mat).reshape(-1, 2)

        for det, fp in zip(detections, field_pts):
            fx, fy = float(fp[0]), float(fp[1])
            det["fx"]       = round(fx, 3)
            det["fy"]       = round(fy, 3)
            det["in_field"] = (0 <= fx <= self.FIELD_LENGTH) and (0 <= fy <= self.FIELD_WIDTH)

        return detections

    def _classify_teams_auto(
        self,
        image:              np.ndarray,
        detections:         List[Dict],
        threshold:          Optional[float] = None,
        color_method:       str = "mean_l_norm",
        outlier_std_factor: Optional[float] = 2.0,
    ) -> List[int]:
        """
        K-Means(k=2) 로 유니폼 색상을 자동 군집화해 팀을 배정한다.

        각 선수의 상체 평균 LAB 색상을 특징으로 사용.
        아웃라이어 제거는 두 단계로 수행된다.

        1. threshold  : 클러스터 중심까지의 거리가 이 값을 초과하면 -1 처리 (절대 기준).
        2. outlier_std_factor : 각 클러스터 내 거리 분포에서
                                mean + factor × std 초과 시 -1 처리 (상대 기준).
                                골키퍼·오탐 심판처럼 클러스터 내에서 동떨어진 선수를
                                자동으로 제거한다. None 이면 비활성화.

        Args:
            image:              원본 BGR 이미지.
            detections:         detection 딕셔너리 목록.
            threshold:          LAB 거리 절대 임계값. None 이면 비활성화.
            color_method:       유니폼 색상 추출 방식.
            outlier_std_factor: 클러스터 내 아웃라이어 제거 강도.
                                값이 낮을수록 더 공격적으로 제거 (권장 1.5~2.5).
                                None 이면 비활성화.

        Returns:
            detections 와 동일한 순서의 team_id 목록.
            0 or 1: 팀 배정 / -1: 아웃라이어 또는 비선수
        """
        from sklearn.cluster import KMeans

        team_ids       = [-1] * len(detections)
        player_indices = [i for i, d in enumerate(detections) if d["class_id"] == 0]

        if len(player_indices) < 2:
            for i in player_indices:
                team_ids[i] = 0
            return team_ids

        lab_colors = np.array(
            [self._get_jersey_mean_lab(image, detections[i], color_method) for i in player_indices],
            dtype=np.float32,
        )

        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels  = km.fit_predict(lab_colors)
        centers = km.cluster_centers_

        # 각 선수의 클러스터 중심까지 거리
        dists = np.array([
            float(np.linalg.norm(lab_colors[idx] - centers[labels[idx]]))
            for idx in range(len(player_indices))
        ])

        is_outlier = np.zeros(len(player_indices), dtype=bool)

        # 1단계: 절대 임계값 필터
        if threshold is not None:
            is_outlier |= (dists > threshold)

        # 2단계: 클러스터 내 상대 아웃라이어 제거 (mean + factor × std)
        if outlier_std_factor is not None:
            for label_val in (0, 1):
                mask = (labels == label_val)
                if mask.sum() < 3:   # 3명 미만이면 std 계산 무의미
                    continue
                cluster_dists = dists[mask]
                cutoff = cluster_dists.mean() + outlier_std_factor * cluster_dists.std()
                is_outlier[mask] |= (cluster_dists > cutoff)

        for idx, (i, label) in enumerate(zip(player_indices, labels)):
            team_ids[i] = -1 if is_outlier[idx] else int(label)

        return team_ids

    _MIN_CROP_PX = 32     # 크롭 최소 크기 (px): 미만이면 LANCZOS4 업스케일
    _LAB_L_NORM  = 128.0  # L* 정규화 고정값 (OpenCV LAB 기준, ≈ CIE L*=50)
                          # 조명 변화 제거 — 색조(a*, b*)만으로 거리 계산

    def _get_jersey_mean_lab(
        self,
        image:        np.ndarray,
        det:          Dict,
        color_method: str = "mean",
    ) -> np.ndarray:
        """
        단일 선수 BBox 상체 영역에서 유니폼 색상을 추출해 LAB 벡터로 반환한다.

        color_method:
            'mean'          — 잔디 제거 후 LAB 채널 평균 (기본값)
            'mean_l_norm'   — LAB 평균 후 L* = _LAB_L_NORM 고정 (조명 제거)
            'kmeans'        — K-Means(k=2) 지배 클러스터 중심
            'kmeans_l_norm' — K-Means 지배 색상 후 L* = _LAB_L_NORM 고정

        공통 처리:
            1. 상체 영역 크롭 (y: 15%~60%, x: 15%~85%)
            2. 크롭이 _MIN_CROP_PX 미만이면 LANCZOS4 업스케일
            3. 잔디색(초록 HSV 범위) 픽셀 제거
            4. 유효 픽셀 부족 시 fallback 반환

        Args:
            image:        원본 BGR 이미지.
            det:          detection 딕셔너리 (x_center, y_center, width, height 모두 정규화).
            color_method: 색상 추출 방식 (위 참고).

        Returns:
            3차원 float32 LAB 벡터.
        """
        H_IMG, W_IMG = image.shape[:2]
        cx, cy, w, h = det["x_center"], det["y_center"], det["width"], det["height"]

        x1_px = int((cx - w / 2) * W_IMG)
        y1_px = int((cy - h / 2) * H_IMG)
        w_px  = max(1, int(w * W_IMG))
        h_px  = max(1, int(h * H_IMG))

        # 상체 영역: 머리(상단 15%) 제외, 하체(하단 40%) 제외
        crop_y1 = max(0,     y1_px + int(h_px * 0.15))
        crop_y2 = min(H_IMG, y1_px + int(h_px * 0.60))
        crop_x1 = max(0,     x1_px + int(w_px * 0.15))
        crop_x2 = min(W_IMG, x1_px + int(w_px * 0.85))

        fallback = np.array([self._LAB_L_NORM, 128.0, 128.0], dtype=np.float32)

        if (crop_y2 - crop_y1) < 4 or (crop_x2 - crop_x1) < 4:
            return fallback

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # 크롭이 너무 작으면 업스케일
        ch, cw = crop.shape[:2]
        if ch < self._MIN_CROP_PX or cw < self._MIN_CROP_PX:
            scale = max(self._MIN_CROP_PX / max(1, ch), self._MIN_CROP_PX / max(1, cw))
            crop  = cv2.resize(crop, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LANCZOS4)

        # 잔디 마스크 제거
        crop_hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(crop_hsv, self._GRASS_LOWER, self._GRASS_UPPER)
        valid_mask = (grass_mask == 0)

        valid_pixels = crop[valid_mask]   # shape (N, 3) BGR
        if len(valid_pixels) < 4:
            return fallback

        # 유효 픽셀 BGR → LAB 변환
        pixels_lab = cv2.cvtColor(
            valid_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB,
        ).reshape(-1, 3).astype(np.float32)

        if color_method in ("kmeans", "kmeans_l_norm"):
            from sklearn.cluster import KMeans
            n_clusters = min(2, len(pixels_lab))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            labels = km.fit_predict(pixels_lab)
            counts = np.bincount(labels)
            result_lab = km.cluster_centers_[counts.argmax()].copy()
        else:
            result_lab = pixels_lab.mean(axis=0)

        if color_method in ("mean_l_norm", "kmeans_l_norm"):
            result_lab[0] = self._LAB_L_NORM

        return result_lab.astype(np.float32)

    @staticmethod
    def _parse_color(color: "str | tuple") -> np.ndarray:
        """
        RGB 튜플 또는 hex 문자열을 1×1×3 uint8 BGR ndarray 로 변환한다.

        Args:
            color: ``(R, G, B)`` 튜플 또는 ``"#RRGGBB"`` 문자열.

        Returns:
            shape (1, 1, 3) uint8 BGR ndarray.

        Examples:
            >>> TopViewAnalyzer._parse_color((30, 144, 255))
            >>> TopViewAnalyzer._parse_color("#1E90FF")
        """
        if isinstance(color, str):
            h = color.lstrip("#")
            if len(h) != 6:
                raise ValueError(f"hex 색상은 '#RRGGBB' 형식이어야 합니다: {color!r}")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        else:
            r, g, b = int(color[0]), int(color[1]), int(color[2])
        return np.array([[[b, g, r]]], dtype=np.uint8)  # BGR

    def _draw_pitch(self, ax) -> None:
        """
        matplotlib Axes 에 FIFA 표준 축구장을 그린다.

        Args:
            ax: matplotlib Axes 인스턴스.
        """
        import matplotlib.patches as mpatches
        from matplotlib.patches import Arc, Circle

        FL, FW   = self.FIELD_LENGTH, self.FIELD_WIDTH
        BBL, BBW = _BBL, _BBW
        SBL, SBW = _SBL, _SBW
        CCR      = _CCR
        SPOT     = _SPOT
        CX, CY   = _CX, _CY
        BBT, SBT = _BBT, _SBT

        ax.set_facecolor("#2d6a1e")

        # 터치라인 + 골라인
        ax.add_patch(mpatches.Rectangle((0, 0), FL, FW, lw=2, ec="w", fc="none"))

        # 센터라인
        ax.plot([CX, CX], [0, FW], "w-", lw=1.5)

        # 센터서클 + 스팟
        ax.add_patch(Circle((CX, CY), CCR, color="w", fill=False, lw=1.5))
        ax.plot(CX, CY, "wo", ms=4)

        for side in ("left", "right"):
            if side == "left":
                bx, sx, spot_x = 0, 0, SPOT
                arc_angle = 0
                t1 = -np.degrees(np.arccos(_SBL / _CCR))
                t2 =  np.degrees(np.arccos(_SBL / _CCR))
            else:
                bx, sx, spot_x = FL - BBL, FL - SBL, FL - SPOT
                arc_angle = 180
                t1 = -np.degrees(np.arccos(_SBL / _CCR))
                t2 =  np.degrees(np.arccos(_SBL / _CCR))

            # 패널티 박스
            ax.add_patch(mpatches.Rectangle(
                (bx, BBT), BBL, BBW, lw=1.5, ec="w", fc="none",
            ))
            # 골 에어리어
            ax.add_patch(mpatches.Rectangle(
                (sx, SBT), SBL, SBW, lw=1.5, ec="w", fc="none",
            ))
            # 패널티 스팟
            ax.plot(spot_x, CY, "wo", ms=4)
            # 패널티 반원
            ax.add_patch(Arc(
                (spot_x, CY), 2 * CCR, 2 * CCR,
                angle=arc_angle, theta1=t1, theta2=t2,
                color="w", lw=1.5,
            ))
            # 골대
            gw2 = 7.32 / 2
            goal_x = -2.44 if side == "left" else FL
            ax.add_patch(mpatches.Rectangle(
                (goal_x, CY - gw2), 2.44, 7.32,
                lw=1.5, ec="w", fc="none",
            ))

        ax.set_xlim(-5, FL + 5)
        ax.set_ylim(-4, FW + 4)
        ax.set_aspect("equal")
        ax.invert_yaxis()
