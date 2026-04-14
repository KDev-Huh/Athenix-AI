"""
HSV 색상 분할로 축구장 피치 영역(바운딩 박스)을 검출하는 내부 모듈.
KeypointsDatasetBuilder 에서만 사용한다.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class PitchDetector:
    """
    축구장 이미지에서 그린(잔디) 영역을 색상 기반으로 검출하고
    정규화된 피치 바운딩 박스를 반환하는 클래스.
    """

    def __init__(self):
        # HSV 녹색 범위 (어두운 녹색 ~ 밝은 녹색)
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([85, 255, 255])

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def detect_pitch_from_image(self, image_path: str) -> Optional[Dict]:
        """
        이미지 파일에서 피치 바운딩 박스를 검출한다.

        Args:
            image_path: 입력 이미지 파일 경로

        Returns:
            {
                "image_path":      str,
                "image_shape":     {"height": int, "width": int},
                "pitch_detection": {
                    "class_id", "class_name",
                    "center_x", "center_y", "width", "height",
                    "x_min", "y_min", "x_max", "y_max",
                    "area", "contour_area"  ← 모두 정규화(0~1)
                }
            }
            검출 실패 시 None.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        green_mask      = self._detect_green_area(image)
        largest_contour = self._find_largest_contour(green_mask)
        if largest_contour is None:
            return None

        pitch_bbox = self._get_bounding_box(largest_contour, image.shape)
        if pitch_bbox is None:
            return None

        return {
            "image_path":      image_path,
            "image_shape":     {"height": image.shape[0], "width": image.shape[1]},
            "pitch_detection": pitch_bbox,
        }

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _detect_green_area(self, image: np.ndarray) -> np.ndarray:
        hsv    = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask   = cv2.inRange(hsv, self.lower_green, self.upper_green)
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _find_largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _get_bounding_box(
        self,
        contour: np.ndarray,
        image_shape: Tuple[int, ...],
    ) -> Optional[Dict]:
        if contour is None:
            return None

        height, width = image_shape[:2]
        x, y, w, h   = cv2.boundingRect(contour)

        x_min_n = x / width
        y_min_n = y / height
        x_max_n = (x + w) / width
        y_max_n = (y + h) / height

        center_x    = (x_min_n + x_max_n) / 2
        center_y    = (y_min_n + y_max_n) / 2
        bbox_width  = x_max_n - x_min_n
        bbox_height = y_max_n - y_min_n

        return {
            "class_id":    0,
            "class_name":  "pitch",
            "center_x":    center_x,
            "center_y":    center_y,
            "width":       bbox_width,
            "height":      bbox_height,
            "x_min":       x_min_n,
            "y_min":       y_min_n,
            "x_max":       x_max_n,
            "y_max":       y_max_n,
            "area":        bbox_width * bbox_height,
            "contour_area": cv2.contourArea(contour) / (width * height),
        }
