"""
recommend router
================
/ai/recommend 하위 엔드포인트를 정의한다.
"""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/ai/recommend", tags=["AI Recommend"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/image", summary="AI 더 나은 플레이 추천")
async def recommend_image(
    request: Request,
    image: UploadFile = File(...),
    is_rtl: bool = False,
) -> JSONResponse:
    """
    경기 장면 이미지를 분석해 최적의 플레이 액션과 좌표를 추천한다.

    - **image**: 분석할 경기 장면 이미지 (jpg / png / webp)
    - **is_rtl**: 공격 방향이 오른쪽→왼쪽이면 true (기본값: false = 왼쪽→오른쪽)
    """
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 허용: jpg, png, webp (받은 값: {image.content_type})",
        )

    suffix = "." + image.filename.rsplit(".", 1)[-1] if "." in (image.filename or "") else ".jpg"
    image_bytes = await image.read()

    try:
        result = request.app.state.recommend_service.recommend_from_bytes(image_bytes, suffix, is_rtl=is_rtl)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return JSONResponse(content={"success": True, "data": result, "error": None})
