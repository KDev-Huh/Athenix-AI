"""
Ultralytics YOLO detection 모델 학습 / 평가 / 저장 / 로드 클래스.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

from .config import DetectionConfig


class DetectionModel:
    """
    YOLO detection 모델(선수 · 공 · 심판 검출)을 학습하고 관리하는 클래스.

    사용 예시:
        # 기본 설정으로 학습
        model = DetectionModel(model_dir="models/detection")
        model.train(dataset_yaml="data/detection/external/detection_Soccana/V1/data.yaml")
        print(model.evaluate(dataset_yaml=..., split="val"))
        model.save(name="soccana_detection_v1")
        # → models/detection/soccana_detection_v1.pt
        # → models/detection/soccana_detection_v1_config.json

        # 커스텀 설정으로 학습
        from src.modeling.detection.config import DetectionConfig
        config = DetectionConfig(
            model_type="yolov8s.pt",
            yolo_params={"epochs": 100, "batch": 8},
        )
        model = DetectionModel(config=config)
    """

    def __init__(
        self,
        model_dir: str = "models/detection",
        config: DetectionConfig = None,
    ):
        """
        Args:
            model_dir: 모델 가중치 / config 저장 경로
            config:    DetectionConfig 인스턴스. None이면 기본 설정 사용.
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.config              = config or DetectionConfig()
        self.model               = None
        self._best_weights_path: str = None

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def train(self, dataset_yaml: str) -> None:
        """
        YOLO detection 모델을 학습한다.

        Args:
            dataset_yaml: Ultralytics data.yaml 파일 경로
                          예) "data/detection/external/detection_Soccana/V1/data.yaml"
        """
        from ultralytics import YOLO

        model = YOLO(self.config.model_type)
        results = model.train(
            data=dataset_yaml,
            project=self.config.project_name,
            name=self.config.run_name,
            **self.config.yolo_params,
        )

        # YOLO 가 자동 저장한 best.pt 경로 추적 (results.save_dir 로 실제 경로 확인)
        self._best_weights_path = str(Path(results.save_dir) / "weights" / "best.pt")
        self.model = YOLO(self._best_weights_path)

        print(f"\n학습 완료 → {self._best_weights_path}")

    def evaluate(self, dataset_yaml: str, split: str = "val") -> dict:
        """
        학습된 모델을 평가하고 전체 및 클래스별 지표를 반환한다.

        Args:
            dataset_yaml: Ultralytics data.yaml 파일 경로
            split:        평가할 split ("val" 또는 "test")

        Returns:
            {
                "all":     {"precision": ..., "recall": ..., "mAP50": ..., "mAP50-95": ...},
                "Player":  {"mAP50": ..., "mAP50-95": ...},
                "Ball":    {"mAP50": ..., "mAP50-95": ...},
                "Referee": {"mAP50": ..., "mAP50-95": ...},
            }
        """
        if self.model is None:
            raise RuntimeError("먼저 train() 또는 load()를 호출해 주세요.")

        metrics = self.model.val(data=dataset_yaml, split=split)

        result = {
            "all": {
                "precision": round(metrics.box.mp,    4),
                "recall":    round(metrics.box.mr,    4),
                "mAP50":     round(metrics.box.map50, 4),
                "mAP50-95":  round(metrics.box.map,   4),
            }
        }

        # 클래스별 mAP 추출
        if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
            class_names = self.model.names
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                name = class_names[int(class_idx)]
                result[name] = {
                    "mAP50":    round(float(metrics.box.ap50[i]), 4),
                    "mAP50-95": round(float(metrics.box.ap[i]),   4),
                }

        return result

    def save(self, name: str = None) -> None:
        """
        학습된 가중치와 config 를 model_dir 에 저장한다.

        Args:
            name: 파일 이름 접두어. None이면 타임스탬프로 자동 생성.
                  예) "soccana_v1" → soccana_v1.pt
                                      soccana_v1_config.json
        """
        if self.model is None or self._best_weights_path is None:
            raise RuntimeError("저장할 모델이 없습니다. 먼저 train()을 호출해 주세요.")

        prefix = name if name else f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        weights_dest = os.path.join(self.model_dir, f"{prefix}.pt")
        shutil.copy2(self._best_weights_path, weights_dest)
        print(f"Saved → {weights_dest}")

        config_path = os.path.join(self.model_dir, f"{prefix}_config.json")
        self.config.save(config_path)
        print(f"Saved → {config_path}")

    def load(self, weights_path: str, config_path: str = None) -> None:
        """
        저장된 가중치와 config 를 로드한다.

        Args:
            weights_path: .pt 파일명 또는 절대 경로
            config_path:  _config.json 파일명 또는 절대 경로.
                          None이면 weights_path 기준으로 자동 추론.
        """
        from ultralytics import YOLO

        resolved = self._resolve_path(weights_path)
        self.model              = YOLO(resolved)
        self._best_weights_path = resolved

        resolved_config = (
            self._resolve_path(config_path)
            if config_path
            else resolved.replace(".pt", "_config.json")
        )
        if os.path.exists(resolved_config):
            self.config = DetectionConfig.load(resolved_config)

        print(f"모델 로드 완료: {resolved}")

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _resolve_path(self, file_name: str) -> str:
        """절대 경로면 그대로, 상대 경로면 model_dir 를 prefix 로 붙인다."""
        if os.path.isabs(file_name):
            return file_name
        return os.path.join(self.model_dir, file_name)
