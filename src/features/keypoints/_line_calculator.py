"""
SoccerNet 라인 데이터에서 교점을 계산해 29개 필드 키포인트를 추출하는 내부 모듈.
KeypointsDatasetBuilder 에서만 사용한다.
"""

import json
import math
from typing import Dict, List, Optional, Tuple


class LineIntersectionCalculator:
    """
    SoccerNet calibration JSON(라인 끝점 좌표)에서
    교점 계산을 통해 29개의 필드 키포인트를 추출하는 클래스.
    """

    def __init__(self):
        self.field_keypoints: Dict = {}
        self.lines: Dict           = {}

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def load_soccernet_data(self, json_path: str) -> Dict:
        """SoccerNet calibration JSON 파일을 로드한다."""
        with open(json_path, "r") as f:
            data = json.load(f)
        self.lines = data
        return data

    def calculate_field_keypoints(self) -> Tuple[Dict, Dict]:
        """
        라인 교점 계산으로 29개의 필드 키포인트를 추출한다.

        Returns:
            (keypoints, lines)
            keypoints: {"0_sideline_top_left": (x, y), ...}  ← 정규화(0~1)
            lines:     원본 라인 데이터
        """
        keypoints = {}

        def get_line(key):
            return self.lines.get(key, [])

        # ── 라인 데이터 로드 ─────────────────────────────────────────
        side_line_top    = get_line("Side line top")
        side_line_bottom = get_line("Side line bottom")
        side_line_left   = get_line("Side line left")
        side_line_right  = get_line("Side line right")

        big_rect_left_top    = get_line("Big rect. left top")
        big_rect_left_main   = get_line("Big rect. left main")
        big_rect_left_bottom = get_line("Big rect. left bottom")
        big_rect_right_top   = get_line("Big rect. right top")
        big_rect_right_main  = get_line("Big rect. right main")
        big_rect_right_bottom = get_line("Big rect. right bottom")

        small_rect_left_top    = get_line("Small rect. left top")
        small_rect_left_main   = get_line("Small rect. left main")
        small_rect_left_bottom = get_line("Small rect. left bottom")
        small_rect_right_top   = get_line("Small rect. right top")
        small_rect_right_main  = get_line("Small rect. right main")
        small_rect_right_bottom = get_line("Small rect. right bottom")

        middle_line    = get_line("Middle line")
        circle_central = get_line("Circle central")
        circle_left    = get_line("Circle left")
        circle_right   = get_line("Circle right")

        # ── 왼쪽 키포인트 (0~10) ─────────────────────────────────────
        self._calc_intersection(keypoints, "0_sideline_top_left",        side_line_top,    side_line_left)
        self._calc_intersection(keypoints, "1_big_rect_left_top_pt1",    side_line_left,   big_rect_left_top)
        self._calc_intersection(keypoints, "2_big_rect_left_top_pt2",    big_rect_left_top, big_rect_left_main)
        self._calc_intersection(keypoints, "3_big_rect_left_bottom_pt1", side_line_left,   big_rect_left_bottom)
        self._calc_intersection(keypoints, "4_big_rect_left_bottom_pt2", big_rect_left_bottom, big_rect_left_main)
        self._calc_intersection(keypoints, "5_small_rect_left_top_pt1",  side_line_left,   small_rect_left_top)
        self._calc_intersection(keypoints, "6_small_rect_left_top_pt2",  small_rect_left_top, small_rect_left_main)
        self._calc_intersection(keypoints, "7_small_rect_left_bottom_pt1", side_line_left, small_rect_left_bottom)
        self._calc_intersection(keypoints, "8_small_rect_left_bottom_pt2", small_rect_left_bottom, small_rect_left_main)
        self._calc_intersection(keypoints, "9_sideline_bottom_left",     side_line_bottom, side_line_left)

        # 10. 왼쪽 반원 (big_rect_left_main과 가장 먼 점)
        if circle_left and big_rect_left_main:
            pt = max(circle_left, key=lambda p: self._point_to_line_distance(p, big_rect_left_main))
            if 0.0 <= pt["x"] <= 1.0 and 0.0 <= pt["y"] <= 1.0:
                keypoints["10_left_semicircle_right"] = (pt["x"], pt["y"])

        # ── 중앙 키포인트 (11~15) ─────────────────────────────────────
        self._calc_intersection(keypoints, "11_center_line_top",    middle_line, side_line_top)
        self._calc_intersection(keypoints, "12_center_line_bottom", middle_line, side_line_bottom)

        # 13. 중앙 원 상단
        if circle_central and middle_line:
            y_values  = [p["y"] for p in circle_central]
            median_y  = sorted(y_values)[len(y_values) // 2]
            upper     = [p for p in circle_central if p["y"] <= median_y]
            if len(upper) >= 2:
                closest = sorted(upper, key=lambda p: self._point_to_line_distance(p, middle_line))[:2]
                pt = self._line_intersection(closest, middle_line)
                if pt:
                    keypoints["13_center_circle_top"] = pt

        # 14. 중앙 원 하단
        if circle_central and middle_line:
            y_values = [p["y"] for p in circle_central]
            median_y = sorted(y_values)[len(y_values) // 2]
            lower    = [p for p in circle_central if p["y"] > median_y]
            if len(lower) >= 2:
                closest = sorted(lower, key=lambda p: self._point_to_line_distance(p, middle_line))[:2]
                pt = self._line_intersection(closest, middle_line)
                if pt:
                    keypoints["14_center_circle_bottom"] = pt

        # 15. 필드 중심 (13, 14의 중점)
        if "13_center_circle_top" in keypoints and "14_center_circle_bottom" in keypoints:
            top_x, top_y    = keypoints["13_center_circle_top"]
            bot_x, bot_y    = keypoints["14_center_circle_bottom"]
            keypoints["15_field_center"] = ((top_x + bot_x) / 2, (top_y + bot_y) / 2)

        # ── 오른쪽 키포인트 (16~28) ───────────────────────────────────
        self._calc_intersection(keypoints, "16_sideline_top_right",        side_line_top,    side_line_right)
        self._calc_intersection(keypoints, "17_big_rect_right_top_pt1",    side_line_right,  big_rect_right_top)
        self._calc_intersection(keypoints, "18_big_rect_right_top_pt2",    big_rect_right_top, big_rect_right_main)
        self._calc_intersection(keypoints, "19_big_rect_right_bottom_pt1", side_line_right,  big_rect_right_bottom)
        self._calc_intersection(keypoints, "20_big_rect_right_bottom_pt2", big_rect_right_bottom, big_rect_right_main)
        self._calc_intersection(keypoints, "21_small_rect_right_top_pt1",  side_line_right,  small_rect_right_top)
        self._calc_intersection(keypoints, "22_small_rect_right_top_pt2",  small_rect_right_top, small_rect_right_main)
        self._calc_intersection(keypoints, "23_small_rect_right_bottom_pt1", side_line_right, small_rect_right_bottom)
        self._calc_intersection(keypoints, "24_small_rect_right_bottom_pt2", small_rect_right_bottom, small_rect_right_main)
        self._calc_intersection(keypoints, "25_sideline_bottom_right",     side_line_bottom, side_line_right)

        # 26. 오른쪽 반원
        if circle_right and big_rect_right_main:
            pt = max(circle_right, key=lambda p: self._point_to_line_distance(p, big_rect_right_main))
            if 0.0 <= pt["x"] <= 1.0 and 0.0 <= pt["y"] <= 1.0:
                keypoints["26_right_semicircle_left"] = (pt["x"], pt["y"])

        # 27. 중앙 원 왼쪽 (중심선 기준 가장 먼 점)
        if circle_central and middle_line:
            x_values = [p["x"] for p in circle_central]
            median_x = sorted(x_values)[len(x_values) // 2]
            left_pts = [p for p in circle_central if p["x"] <= median_x]
            if left_pts:
                pt = max(left_pts, key=lambda p: self._point_to_line_distance(p, middle_line))
                if 0.0 <= pt["x"] <= 1.0 and 0.0 <= pt["y"] <= 1.0:
                    keypoints["27_center_circle_left"] = (pt["x"], pt["y"])

        # 28. 중앙 원 오른쪽
        if circle_central and middle_line:
            x_values  = [p["x"] for p in circle_central]
            median_x  = sorted(x_values)[len(x_values) // 2]
            right_pts = [p for p in circle_central if p["x"] > median_x]
            if right_pts:
                pt = max(right_pts, key=lambda p: self._point_to_line_distance(p, middle_line))
                if 0.0 <= pt["x"] <= 1.0 and 0.0 <= pt["y"] <= 1.0:
                    keypoints["28_center_circle_right"] = (pt["x"], pt["y"])

        self.field_keypoints = keypoints
        return self.field_keypoints, self.lines

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _calc_intersection(
        self,
        keypoints: Dict,
        name: str,
        line1: List[Dict],
        line2: List[Dict],
    ) -> None:
        """교점을 계산하고 결과를 keypoints 딕셔너리에 추가한다."""
        if not line1 or not line2:
            return
        pt = self._line_intersection(line1, line2)
        if pt:
            keypoints[name] = pt

    def _line_intersection(
        self,
        line1: List[Dict],
        line2: List[Dict],
    ) -> Optional[Tuple[float, float]]:
        """두 라인의 교점을 계산한다. 범위(0~1) 밖이면 None."""
        if len(line1) < 2 or len(line2) < 2:
            return None

        x1, y1 = line1[0]["x"], line1[0]["y"]
        x2, y2 = line1[1]["x"], line1[1]["y"]
        x3, y3 = line2[0]["x"], line2[0]["y"]
        x4, y4 = line2[1]["x"], line2[1]["y"]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None

        return x, y

    def _point_to_line_distance(self, point: Dict, line: List[Dict]) -> float:
        """점에서 라인까지의 수직 거리를 반환한다."""
        if len(line) < 2:
            return float("inf")

        x0, y0 = point["x"], point["y"]
        x1, y1 = line[0]["x"], line[0]["y"]
        x2, y2 = line[1]["x"], line[1]["y"]

        a = y2 - y1
        b = x1 - x2
        c = (x2 - x1) * y1 - (y2 - y1) * x1

        if a == 0 and b == 0:
            return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

        return abs(a * x0 + b * y0 + c) / math.sqrt(a * a + b * b)
