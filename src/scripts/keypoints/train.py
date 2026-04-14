"""
Keypoints 전체 학습 파이프라인

실행 순서:
    1. SoccerNet calibration 데이터 다운로드
    2. 이미지 처리 및 YOLO 데이터셋 생성 (피치 검출 + 키포인트 추출)
    3. YOLO pose 모델 학습
    4. 모델 평가
    5. 모델 저장

실행 방법:
    python -m src.scripts.keypoints.train
"""

import sys
from pathlib import Path

# 어느 디렉토리에서 실행해도 프로젝트 루트를 인식하도록 설정
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.preprocessing.keypoints import SoccerNetDataDownloader
from src.features.keypoints import KeypointsDatasetBuilder
from src.modeling.keypoints import KeypointsModel, KeypointsConfig

# ── 설정 ──────────────────────────────────────────────────────────────

# SoccerNet API 비밀번호 (constants.py 참고)
SOCCERNET_PASSWORD = "<password>"

# 데이터 경로
LOCAL_DIRECTORY = r"C:\Datasets\SoccerNet\Data"
DATA_DIR        = r"C:\Datasets\SoccerNet\Data\calibration"
OUTPUT_DIR      = r"C:\Datasets\SoccerNet\Data\calibration\unified_output"

# 이미 완료된 단계는 True 로 설정해 건너뛸 수 있다
SKIP_DOWNLOAD = False
SKIP_BUILD    = False

# 모델 저장 이름 (None이면 타임스탬프 자동 생성)
MODEL_NAME = "soccernet_keypoints_v1"
MODEL_DIR  = str(ROOT / "models/keypoints")

# ── 학습 설정 ─────────────────────────────────────────────────────────
# 동작 확인용 (epochs=5)
CONFIG = KeypointsConfig(
    model_type="yolo11n-pose.pt",
    yolo_params={"epochs": 1, "imgsz": 960, "batch": 8, "device": 0},
    project_name="runs_pose",
    run_name="football_pitch_pose_v1",
)

# 정식 학습용 (주석 해제 후 위 CONFIG 주석 처리)
# CONFIG = KeypointsConfig(
#     model_type="yolo11n-pose.pt",
#     yolo_params={"epochs": 100, "imgsz": 960, "batch": 8, "device": 0},
#     project_name="runs_pose",
#     run_name="football_pitch_pose_v1",
# )

# ── 실행 ──────────────────────────────────────────────────────────────

def run():
    # ── Step 1. 데이터 다운로드 ──────────────────────────────────────
    if not SKIP_DOWNLOAD:
        print("\n[Step 1] SoccerNet 데이터 다운로드")
        downloader = SoccerNetDataDownloader(
            local_directory=LOCAL_DIRECTORY,
            password=SOCCERNET_PASSWORD,
        )
        downloader.download()
    else:
        print("\n[Step 1] 다운로드 건너뜀 (SKIP_DOWNLOAD=True)")

    # ── Step 2. YOLO 데이터셋 생성 ───────────────────────────────────
    if not SKIP_BUILD:
        print("\n[Step 2] YOLO 데이터셋 생성")
        builder = KeypointsDatasetBuilder(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
        )
        builder.build()
    else:
        print("\n[Step 2] 데이터셋 빌드 건너뜀 (SKIP_BUILD=True)")

    dataset_yaml = f"{OUTPUT_DIR}/dataset.yaml"

    # ── Step 3. 모델 학습 ─────────────────────────────────────────────
    print("\n[Step 3] 모델 학습")
    model = KeypointsModel(model_dir=MODEL_DIR, config=CONFIG)
    model.train(dataset_yaml=dataset_yaml)

    # ── Step 4. 모델 평가 ─────────────────────────────────────────────
    print("\n[Step 4] 모델 평가")
    results = model.evaluate(dataset_yaml=dataset_yaml, split="val")
    for metric_name, value in results.items():
        print(f"  {metric_name:<12}: {value}")

    # ── Step 5. 모델 저장 ─────────────────────────────────────────────
    print("\n[Step 5] 모델 저장")
    model.save(name=MODEL_NAME)

    print("\n학습 완료!")


if __name__ == "__main__":
    run()
