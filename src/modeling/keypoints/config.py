"""
KeypointsModel / KeypointsPredictor 공유 설정 클래스.
"""

import json


DEFAULT_MODEL_TYPE   = "yolo11n-pose.pt"
DEFAULT_PROJECT_NAME = "runs_pose"
DEFAULT_RUN_NAME     = "football_pitch_pose"

DEFAULT_YOLO_PARAMS = dict(
    epochs=100,
    imgsz=960,
    batch=8,
    device=0,
    workers=4,
    pretrained=True,
    optimizer="auto",
)


class KeypointsConfig:
    """
    YOLO pose 모델 학습 설정 클래스.

    모델 저장 시 JSON으로 함께 저장되며, Predictor 로드 시 자동으로 읽어
    학습 파라미터가 항상 동일하게 유지된다.

    사용 예시:
        # 기본값 사용
        config = KeypointsConfig()

        # 커스텀 설정
        config = KeypointsConfig(
            model_type="yolo11n-pose.pt",
            yolo_params={"epochs": 200, "imgsz": 1280, "batch": 4},
            project_name="runs_pose",
            run_name="pitch_keypoints_v2",
        )
    """

    def __init__(
        self,
        model_type: str = None,
        yolo_params: dict = None,
        project_name: str = None,
        run_name: str = None,
    ):
        """
        Args:
            model_type:   Ultralytics 사전학습 모델 파일명.
                          None이면 기본값("yolo11n-pose.pt") 사용.
            yolo_params:  YOLO 학습 파라미터. None이면 기본값 사용.
                          일부만 지정하면 기본값에 덮어씌워진다.
            project_name: 학습 결과 저장 디렉토리 이름.
            run_name:     학습 run 이름 (project_name/run_name/weights/best.pt).
        """
        self.model_type   = model_type   or DEFAULT_MODEL_TYPE
        self.yolo_params  = {**DEFAULT_YOLO_PARAMS, **(yolo_params or {})}
        self.project_name = project_name or DEFAULT_PROJECT_NAME
        self.run_name     = run_name     or DEFAULT_RUN_NAME

    # ── 직렬화 ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """설정을 JSON 파일로 저장한다."""
        data = {
            "model_type":   self.model_type,
            "yolo_params":  self.yolo_params,
            "project_name": self.project_name,
            "run_name":     self.run_name,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "KeypointsConfig":
        """JSON 파일에서 설정을 복원한다."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
