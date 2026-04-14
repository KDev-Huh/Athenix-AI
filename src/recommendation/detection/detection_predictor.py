"""
학습된 YOLO detection 모델로 선수 · 공 · 심판을 검출하는 클래스.
단일 이미지, 배치, 실시간 영상(웹캠 / 파일) 모두 지원한다.
"""

import os
from typing import Dict, List, Optional, Union

from src.modeling.detection.config import DetectionConfig


class DetectionPredictor:
    """
    학습된 YOLO detection 모델을 불러와 선수 · 공 · 심판을 검출하는 클래스.

    모델 로드 시 함께 저장된 config 를 자동으로 읽어
    DetectionModel 과 학습 파라미터가 항상 동일하게 유지된다.

    사용 예시:
        predictor = DetectionPredictor(model_dir="models/detection")
        predictor.load_model("soccana_detection_v1.pt")
        # → models/detection/soccana_detection_v1_config.json 자동 로드

        # 단일 이미지
        result = predictor.predict("path/to/image.jpg")

        # 배치
        results = predictor.predict_batch("path/to/image_dir/")

        # 실시간 영상 (웹캠: 0, 파일: "video.mp4")
        predictor.predict_video(source=0)
        predictor.predict_video(source="path/to/match.mp4")
    """

    def __init__(self, model_dir: str = "models/detection"):
        """
        Args:
            model_dir: 모델 파일이 저장된 기본 디렉토리
        """
        self.model_dir = model_dir
        self.model     = None
        self.config: Optional[DetectionConfig] = None

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def load_model(self, weights_file: str, config_file: str = None) -> None:
        """
        모델 가중치와 config 를 로드한다.

        Args:
            weights_file: .pt 파일명 또는 절대 경로
                          예) "soccana_detection_v1.pt"
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
            self.config = DetectionConfig.load(config_path)

        print(f"모델 로드: {weights_file}")

    def predict(self, image_path: str, conf: float = 0.5) -> Dict:
        """
        단일 이미지에서 선수 · 공 · 심판을 검출한다.

        Args:
            image_path: 입력 이미지 파일 경로
            conf:       신뢰도 임계값 (기본값 0.5)

        Returns:
            {
                "image_path": str,
                "detections": [
                    {"class_id": int, "class_name": str,
                     "x_center": float, "y_center": float,
                     "width": float, "height": float,
                     "conf": float},
                    ...
                ]
            }
        """
        if self.model is None:
            raise RuntimeError("먼저 load_model()을 호출해 주세요.")

        results = self.model(image_path, conf=conf)
        return self._parse_result(results[0])

    def predict_batch(self, image_dir: str, conf: float = 0.5) -> List[Dict]:
        """
        디렉토리 내 모든 이미지에 대해 검출을 수행한다.

        Args:
            image_dir: 이미지 디렉토리 경로
            conf:      신뢰도 임계값 (기본값 0.5)

        Returns:
            각 이미지에 대한 predict() 결과 딕셔너리 목록
        """
        if self.model is None:
            raise RuntimeError("먼저 load_model()을 호출해 주세요.")

        results = self.model(image_dir, conf=conf, stream=True)
        return [self._parse_result(r) for r in results]

    def predict_video(
        self,
        source: Union[int, str] = 0,
        conf: float = 0.5,
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        웹캠 또는 영상 파일에서 실시간 검출을 수행한다.

        Args:
            source:    웹캠 번호(0) 또는 영상 파일 경로
            conf:      신뢰도 임계값 (기본값 0.5)
            show:      화면 출력 여부 (기본값 True)
            save_path: 결과 영상 저장 경로. None이면 저장하지 않음.
        """
        if self.model is None:
            raise RuntimeError("먼저 load_model()을 호출해 주세요.")

        import cv2

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")

        writer = None
        if save_path:
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        print(f"실시간 검출 시작 (source={source}) — 'q' 를 눌러 종료")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results   = self.model(frame, conf=conf, verbose=False)
                annotated = results[0].plot()

                if show:
                    cv2.imshow("Detection", annotated)
                if writer:
                    writer.write(annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _parse_result(self, r) -> Dict:
        """YOLO result 객체를 검출 딕셔너리로 변환한다."""
        detections = []
        class_names = r.names  # {0: 'Player', 1: 'Ball', 2: 'Referee'}

        for box in r.boxes:
            xywhn = box.xywhn[0].cpu().numpy()  # 정규화된 (cx, cy, w, h)
            detections.append({
                "class_id":   int(box.cls[0]),
                "class_name": class_names[int(box.cls[0])],
                "x_center":   round(float(xywhn[0]), 6),
                "y_center":   round(float(xywhn[1]), 6),
                "width":      round(float(xywhn[2]), 6),
                "height":     round(float(xywhn[3]), 6),
                "conf":       round(float(box.conf[0]), 4),
            })

        return {
            "image_path": r.path,
            "detections": detections,
        }

    def _resolve_path(self, file_name: str) -> str:
        """절대 경로면 그대로, 상대 경로면 model_dir 를 prefix 로 붙인다."""
        if os.path.isabs(file_name):
            return file_name
        return os.path.join(self.model_dir, file_name)
