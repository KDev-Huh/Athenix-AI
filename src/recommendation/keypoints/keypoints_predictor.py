"""
학습된 YOLO pose 모델로 이미지에서 축구장 키포인트를 예측하는 클래스.
"""

import os
from typing import Dict, List, Optional

from src.modeling.keypoints.config import KeypointsConfig

# 29개 키포인트 이름 (인덱스 순서, dataset_builder.py 의 KEYPOINT_ORDER 와 동일)
KEYPOINT_NAMES = [
    "sideline_top_left",           # 0
    "big_rect_left_top_pt1",       # 1
    "big_rect_left_top_pt2",       # 2
    "big_rect_left_bottom_pt1",    # 3
    "big_rect_left_bottom_pt2",    # 4
    "small_rect_left_top_pt1",     # 5
    "small_rect_left_top_pt2",     # 6
    "small_rect_left_bottom_pt1",  # 7
    "small_rect_left_bottom_pt2",  # 8
    "sideline_bottom_left",        # 9
    "left_semicircle_right",       # 10
    "center_line_top",             # 11
    "center_line_bottom",          # 12
    "center_circle_top",           # 13
    "center_circle_bottom",        # 14
    "field_center",                # 15
    "sideline_top_right",          # 16
    "big_rect_right_top_pt1",      # 17
    "big_rect_right_top_pt2",      # 18
    "big_rect_right_bottom_pt1",   # 19
    "big_rect_right_bottom_pt2",   # 20
    "small_rect_right_top_pt1",    # 21
    "small_rect_right_top_pt2",    # 22
    "small_rect_right_bottom_pt1", # 23
    "small_rect_right_bottom_pt2", # 24
    "sideline_bottom_right",       # 25
    "right_semicircle_left",       # 26
    "center_circle_left",          # 27
    "center_circle_right",         # 28
]


class KeypointsPredictor:
    """
    학습된 YOLO pose 모델을 불러와 이미지에서 29개 필드 키포인트를 예측하는 클래스.

    모델 로드 시 함께 저장된 config 를 자동으로 읽어
    KeypointsModel 과 학습 파라미터가 항상 동일하게 유지된다.

    사용 예시:
        predictor = KeypointsPredictor(model_dir="models/keypoints")
        predictor.load_model("soccernet_keypoints_v1.pt")
        # → models/keypoints/soccernet_keypoints_v1_config.json 자동 로드

        result = predictor.predict("path/to/image.jpg")
        # result = {
        #     "image_path": "...",
        #     "boxes":      1,
        #     "keypoints":  [
        #         {"idx": 0,  "name": "sideline_top_left",      "x": 0.0,  "y": 0.0,  "conf": 0.0},
        #         {"idx": 16, "name": "sideline_top_right",     "x": 0.38, "y": 0.22, "conf": 0.97},
        #         {"idx": 17, "name": "big_rect_right_top_pt1", "x": 0.57, "y": 0.30, "conf": 0.99},
        #         ...
        #     ]
        # }

        results = predictor.predict_batch("path/to/image_dir/")
    """

    def __init__(self, model_dir: str = "models/keypoints"):
        """
        Args:
            model_dir: 모델 파일이 저장된 기본 디렉토리
        """
        self.model_dir = model_dir
        self.model     = None
        self.config: Optional[KeypointsConfig] = None

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def load_model(self, weights_file: str, config_file: str = None) -> None:
        """
        모델 가중치와 config 를 로드한다.

        Args:
            weights_file: .pt 파일명 또는 절대 경로
                          예) "soccernet_keypoints_v1.pt"
            config_file:  _config.json 파일명 또는 절대 경로.
                          None이면 weights_file 기준으로 자동 추론.
        """
        from ultralytics import YOLO

        weights_path = self._resolve_path(weights_file)
        self.model   = YOLO(weights_path)

        config_path = (
            self._resolve_path(config_file)
            if config_file
            else weights_path.replace(".pt", "_config.json")
        )
        if os.path.exists(config_path):
            self.config = KeypointsConfig.load(config_path)

        print(f"모델 로드: {weights_file}")

    def predict(self, image_path: str, conf: float = 0.5) -> Dict:
        """
        단일 이미지에서 피치 바운딩 박스와 키포인트를 예측한다.

        Args:
            image_path: 입력 이미지 파일 경로
            conf:       신뢰도 임계값 (기본값 0.5)

        Returns:
            {
                "image_path": str,
                "boxes":      int,
                "keypoints":  [{"idx": int, "x": float, "y": float, "conf": float}, ...]
            }
        """
        if self.model is None:
            raise RuntimeError("먼저 load_model()을 호출해 주세요.")

        imgsz   = self._get_imgsz()
        results = self.model(image_path, imgsz=imgsz, conf=conf)
        return self._parse_result(results[0])

    def predict_batch(self, image_dir: str, conf: float = 0.5) -> List[Dict]:
        """
        디렉토리 내 모든 이미지에 대해 키포인트를 예측한다.

        Args:
            image_dir: 이미지 디렉토리 경로
            conf:      신뢰도 임계값 (기본값 0.5)

        Returns:
            각 이미지에 대한 predict() 결과 딕셔너리 목록
        """
        if self.model is None:
            raise RuntimeError("먼저 load_model()을 호출해 주세요.")

        imgsz   = self._get_imgsz()
        results = self.model(image_dir, imgsz=imgsz, conf=conf, stream=True)

        return [self._parse_result(r) for r in results]

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _get_imgsz(self) -> int:
        """config 에서 imgsz 를 가져온다. config 가 없으면 기본값(960) 반환."""
        if self.config:
            return self.config.yolo_params.get("imgsz", 960)
        return 960

    def _parse_result(self, r) -> Dict:
        """YOLO result 객체를 키포인트 딕셔너리로 변환한다."""
        keypoints_data = []

        if r.keypoints is not None and len(r.keypoints.xyn) > 0:
            kpts_xy   = r.keypoints.xyn[0].cpu().numpy()
            kpts_conf = (
                r.keypoints.conf[0].cpu().numpy()
                if r.keypoints.conf is not None
                else None
            )

            for i, xy in enumerate(kpts_xy):
                kp = {
                    "idx":  i,
                    "name": KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"kpt_{i}",
                    "x":    float(xy[0]),
                    "y":    float(xy[1]),
                }
                if kpts_conf is not None:
                    kp["conf"] = float(kpts_conf[i])
                keypoints_data.append(kp)

        return {
            "image_path": r.path,
            "boxes":      len(r.boxes),
            "keypoints":  keypoints_data,
        }

    def _resolve_path(self, file_name: str) -> str:
        """절대 경로면 그대로, 상대 경로면 model_dir 를 prefix 로 붙인다."""
        if os.path.isabs(file_name):
            return file_name
        return os.path.join(self.model_dir, file_name)
