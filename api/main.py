"""
Athenix AI API
==============
FastAPI 앱 진입점. 서버 시작 시 모델을 로드하고 라우터를 등록한다.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from api.routers.recommend import router as recommend_router
from api.service.recommend import RecommendService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recommend_service = RecommendService()
    yield


app = FastAPI(
    title="Athenix AI API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(recommend_router)
