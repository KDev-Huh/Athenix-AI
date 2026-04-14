"""
Player · Ball · Referee detection 전체 학습 파이프라인

실행 순서:
    1. 데이터셋 경로 확인
    2. YOLO detection 모델 학습
    3. 모델 평가 (클래스별 mAP 출력)
    4. 모델 저장

실행 방법:
    python -m src.scripts.detection.train
"""

import os
import sys
from pathlib import Path

# 어느 디렉토리에서 실행해도 프로젝트 루트를 인식하도록 설정
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.modeling.detection import DetectionModel, DetectionConfig

# ── 설정 ──────────────────────────────────────────────────────────────

# 데이터셋 yaml 경로 (Ultralytics 포맷)
DATASET_YAML = str(ROOT / "data/detection/external/detection_Soccana/V1/data.yaml")

# 모델 저장 이름 (None이면 타임스탬프 자동 생성)
MODEL_NAME = "soccana_detection_v1"
MODEL_DIR  = str(ROOT / "models/detection")

# ── 학습 설정 ─────────────────────────────────────────────────────────
# 동작 확인용 (epochs=5)
CONFIG = DetectionConfig(
    model_type="yolov8n.pt",
    yolo_params={"epochs": 5, "batch": 16, "device": 0},
    project_name="runs",
    run_name="football_players_Soccana_yolov8n",
)

# 정식 학습용 (주석 해제 후 위 CONFIG 주석 처리)
# CONFIG = DetectionConfig(
#     model_type="yolov8n.pt",
#     yolo_params={"epochs": 50, "batch": 16, "device": 0},
#     project_name="runs",
#     run_name="football_players_Soccana_yolov8n",
# )

# ── 실행 ──────────────────────────────────────────────────────────────

def run():
    # ── Step 1. 데이터셋 확인 ─────────────────────────────────────────
    print("\n[Step 1] 데이터셋 확인")
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(
            f"data.yaml 을 찾을 수 없습니다: {DATASET_YAML}\n"
            "DATASET_YAML 경로를 확인해 주세요."
        )
    print(f"  data.yaml : {DATASET_YAML}")

    # ── Step 2. 모델 학습 ─────────────────────────────────────────────
    print("\n[Step 2] 모델 학습")
    model = DetectionModel(model_dir=MODEL_DIR, config=CONFIG)
    model.train(dataset_yaml=DATASET_YAML)

    # ── Step 3. 모델 평가 ─────────────────────────────────────────────
    print("\n[Step 3] 모델 평가")
    results = model.evaluate(dataset_yaml=DATASET_YAML, split="val")

    for class_name, metrics in results.items():
        print(f"\n  [{class_name}]")
        for metric_name, value in metrics.items():
            print(f"    {metric_name:<12}: {value}")

    # ── Step 4. 모델 저장 ─────────────────────────────────────────────
    print("\n[Step 4] 모델 저장")
    model.save(name=MODEL_NAME)

    print("\n학습 완료!")


if __name__ == "__main__":
    run()
