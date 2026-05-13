"""
RecommendService
================
SoccerSceneRecommender + LLMService 를 조합하는 서비스 레이어.
이미지 분석 → KNN 추천 → LLM 텍스트 생성까지 전체 파이프라인을 담당한다.
"""

import json
import os
import tempfile

from api.service.llm import LLMService
from src.recommendation.soccer_scene_recommender import SoccerSceneRecommender
from src.recommendation.topview_analyzer import TopViewAnalyzer

# ── 환경 변수 ────────────────────────────────────────────────────────────────
DETECTION_MODEL_DIR  = os.getenv("DETECTION_MODEL_DIR",  "models/detection")
KEYPOINTS_MODEL_DIR  = os.getenv("KEYPOINTS_MODEL_DIR",  "models/keypoints")
KNN_INDEX_DIR        = os.getenv("KNN_INDEX_DIR",        "models/knn_index")
CSV_PATH             = os.getenv("CSV_PATH",             "data/StatsBombGithub/processed/vaep_360_merged.csv")
DETECTION_MODEL_NAME = os.getenv("DETECTION_MODEL_NAME", "soccana_detection_v1.pt")
KEYPOINTS_MODEL_NAME = os.getenv("KEYPOINTS_MODEL_NAME", "soccernet_keypoints_v1.pt")


class RecommendService:
    """이미지 → 추천 액션 + LLM 설명 생성 서비스."""

    def __init__(self) -> None:
        self._recommender = SoccerSceneRecommender(
            detection_model_dir=DETECTION_MODEL_DIR,
            keypoints_model_dir=KEYPOINTS_MODEL_DIR,
            knn_index_dir=KNN_INDEX_DIR,
            csv_path=CSV_PATH,
        )
        self._recommender.load_models(
            detection_model_name=DETECTION_MODEL_NAME,
            keypoints_model_name=KEYPOINTS_MODEL_NAME,
        )
        self._llm = LLMService()

    def recommend_from_bytes(self, image_bytes: bytes, suffix: str, is_rtl: bool = False) -> dict:
        """
        이미지 바이트를 받아 추천 결과 + LLM 설명을 반환한다.

        Returns:
            {
                "situation": str,
                "playGuide": {
                    "type":         str,
                    "start_x":      float,
                    "start_y":      float,
                    "end_x":        float | None,
                    "end_y":        float | None,
                    "start_pixel":  {"x": float, "y": float} | None,
                    "end_pixel":    {"x": float, "y": float} | None,
                    "message":      str,
                }
            }

        Raises:
            RuntimeError: 호모그래피 실패 등 파이프라인 오류
            ValueError:   선수 미감지
            LookupError:  유사 장면 없음
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            output = self._recommender.zone_recommend(image_path=tmp_path, show_plot=False, is_rtl=is_rtl)
        finally:
            os.unlink(tmp_path)

        results = output["results"]
        if results.empty:
            raise LookupError("유사 장면을 찾지 못했습니다.")

        top   = results.iloc[0]
        query = output["query"]

        action_type = top["type_name"]
        start = {"x": round(float(top["start_x"]), 2), "y": round(float(top["start_y"]), 2)}
        end   = {"x": round(float(top["end_x"]), 2), "y": round(float(top["end_y"]), 2)} \
                if "end_x" in top.index else None

        print(f"[RecommendService] type={action_type}")
        print(f"  start : {start}")
        print(f"  end   : {end}")

        # 픽셀 역변환 — TopViewAnalyzer.sb_to_pixel() 사용
        H            = output["topview_result"]["homography"]
        image_shape  = output["topview_result"]["image_shape"]
        start_pixel  = TopViewAnalyzer.sb_to_pixel(start["x"], start["y"], H, image_shape)
        end_pixel    = TopViewAnalyzer.sb_to_pixel(end["x"], end["y"], H, image_shape) if end is not None else None

        print(f"  start_pixel : {start_pixel}")
        print(f"  end_pixel   : {end_pixel}")

        situation = self._llm.generate_situation(query)
        message   = self._llm.generate_message(action_type, start, end)

        response = {
            "situation": situation,
            "playGuide": {
                "type":         action_type,
                "start":        start,
                "end":          end,
                "start_pixel":  start_pixel,
                "end_pixel":    end_pixel,
                "message":      message,
            },
        }

        print("[RecommendService] 최종 응답:")
        print(json.dumps(response, ensure_ascii=False, indent=2))

        return response
