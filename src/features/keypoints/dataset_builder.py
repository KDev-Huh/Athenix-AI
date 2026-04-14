"""
SoccerNet calibration 데이터를 Ultralytics YOLO pose 학습 포맷으로 변환하는 모듈.

process_images.py 의 pipeline 로직을 클래스로 캡슐화한다.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from ._pitch_detector import PitchDetector
from ._line_calculator import LineIntersectionCalculator


class KeypointsDatasetBuilder:
    """
    SoccerNet calibration 데이터(JSON 라인 어노테이션 + JPG 이미지)를
    Ultralytics YOLO pose 학습 포맷으로 변환하는 클래스.

    각 이미지에 대해:
      1. 라인 교점 계산 → 29개 필드 키포인트 추출
      2. HSV 색상 분할 → 피치 바운딩 박스 검출
      3. YOLO pose 어노테이션(.txt) 생성
      4. 이미지를 output_dir/images/{split}/ 로 복사

    최종적으로 dataset.yaml 까지 생성하므로, 바로 YOLO 학습에 사용할 수 있다.

    사용 예시:
        builder = KeypointsDatasetBuilder(
            data_dir=r"C:\\Datasets\\SoccerNet\\Data\\calibration",
            output_dir=r"C:\\Datasets\\SoccerNet\\Data\\calibration\\unified_output",
        )
        builder.build()
        # → output_dir/images/{train,valid,test}/
        # → output_dir/labels/{train,valid,test}/
        # → output_dir/dataset.yaml
    """

    SPLITS = ["train", "valid", "test"]

    # 29개 키포인트 순서 (YOLO 어노테이션에서의 인덱스 순서)
    KEYPOINT_ORDER = [
        "0_sideline_top_left",       "1_big_rect_left_top_pt1",    "2_big_rect_left_top_pt2",
        "3_big_rect_left_bottom_pt1","4_big_rect_left_bottom_pt2", "5_small_rect_left_top_pt1",
        "6_small_rect_left_top_pt2", "7_small_rect_left_bottom_pt1","8_small_rect_left_bottom_pt2",
        "9_sideline_bottom_left",    "10_left_semicircle_right",   "11_center_line_top",
        "12_center_line_bottom",     "13_center_circle_top",       "14_center_circle_bottom",
        "15_field_center",           "16_sideline_top_right",      "17_big_rect_right_top_pt1",
        "18_big_rect_right_top_pt2", "19_big_rect_right_bottom_pt1","20_big_rect_right_bottom_pt2",
        "21_small_rect_right_top_pt1","22_small_rect_right_top_pt2","23_small_rect_right_bottom_pt1",
        "24_small_rect_right_bottom_pt2","25_sideline_bottom_right","26_right_semicircle_left",
        "27_center_circle_left",     "28_center_circle_right",
    ]

    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            data_dir:   SoccerNet calibration 디렉토리
                        (train/, valid/, test/ 폴더와 JSON + JPG 파일 포함)
            output_dir: YOLO 데이터셋 출력 경로.
                        None이면 data_dir/unified_output 으로 자동 설정.
        """
        self.data_dir   = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "unified_output"

        self._pitch_detector = PitchDetector()
        self._line_calc      = LineIntersectionCalculator()

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def build(self) -> None:
        """
        전체 splits(train/valid/test)에 대해 YOLO 데이터셋을 생성한다.

        출력 디렉토리 구조:
            output_dir/
            ├── images/{split}/   ← 원본 이미지 복사
            ├── labels/{split}/   ← YOLO pose 어노테이션(.txt)
            └── dataset.yaml      ← Ultralytics 학습 설정 파일
        """
        print("YOLO 데이터셋 생성 시작...")

        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"

        for split in self.SPLITS:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue

            print(f"\n[{split}] 처리 중...")
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)

            json_files = list(split_dir.glob("*.json"))
            print(f"  어노테이션 파일: {len(json_files)}개")

            success, skipped = 0, 0
            for json_file in tqdm(json_files, desc=f"  {split}"):
                image_path = split_dir / json_file.with_suffix(".jpg").name
                if not image_path.exists():
                    skipped += 1
                    continue

                try:
                    annotation = self._process_single(str(json_file), str(image_path))
                except Exception:
                    skipped += 1
                    continue

                if annotation is None:
                    skipped += 1
                    continue

                # YOLO 라벨 저장
                label_path = labels_dir / split / json_file.with_suffix(".txt").name
                label_path.write_text(annotation + "\n", encoding="utf-8")

                # 이미지 복사
                shutil.copy2(str(image_path), str(images_dir / split / image_path.name))
                success += 1

            print(f"  완료: {success}개 / 스킵: {skipped}개")

        self._create_dataset_yaml()
        print(f"\n데이터셋 생성 완료 → {self.output_dir}")

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _process_single(self, json_path: str, image_path: str) -> Optional[str]:
        """
        단일 이미지에 대한 YOLO pose 어노테이션 문자열을 생성한다.

        Returns:
            YOLO pose 포맷 문자열, 처리 실패 시 None.
        """
        # 1. 키포인트 추출
        self._line_calc.load_soccernet_data(json_path)
        keypoints, _ = self._line_calc.calculate_field_keypoints()

        # 2. 피치 바운딩 박스 검출
        pitch_result = self._pitch_detector.detect_pitch_from_image(image_path)
        if not pitch_result:
            return None

        return self._create_annotation(pitch_result["pitch_detection"], keypoints)

    def _create_annotation(self, pitch_data: Dict, keypoints: Dict) -> str:
        """
        Ultralytics YOLO pose 포맷 어노테이션 문자열을 생성한다.

        포맷: <class> <cx> <cy> <w> <h> <px1> <py1> <v1> ... <px29> <py29> <v29>
        visibility: 2 = 검출됨, 0 = 미검출
        """
        parts = [
            "0",
            f"{pitch_data['center_x']:.6f}",
            f"{pitch_data['center_y']:.6f}",
            f"{pitch_data['width']:.6f}",
            f"{pitch_data['height']:.6f}",
        ]

        for kp_name in self.KEYPOINT_ORDER:
            if kp_name in keypoints:
                x, y = keypoints[kp_name]
                parts.extend([f"{x:.6f}", f"{y:.6f}", "2"])
            else:
                parts.extend(["0.0", "0.0", "0"])

        return " ".join(parts)

    def _create_dataset_yaml(self) -> None:
        """Ultralytics YOLO 학습에 필요한 dataset.yaml 을 생성한다."""
        path_str = str(self.output_dir.absolute()).replace("\\", "/")

        yaml_content = f"""# SoccerNet Keypoints Dataset – Ultralytics YOLO pose

path:  {path_str}
train: images/train
val:   images/valid
test:  images/test

# 29개 키포인트, 각각 (x, y, visibility)
kpt_shape: [29, 3]

nc: 1
names:
  0: pitch
"""
        yaml_path = self.output_dir / "dataset.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        print(f"  dataset.yaml 저장 → {yaml_path}")
