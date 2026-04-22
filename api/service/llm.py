"""
LLMService
==========
Ollama 또는 OpenAI API를 통해 텍스트를 생성하는 서비스.
LLM_BACKEND 환경변수로 백엔드를 선택한다: "ollama" (기본값) | "openai"

situation / message 두 가지 텍스트 생성을 담당한다.
"""

import os

import requests

# ── Ollama 설정 ─────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "EEVE-Korean-10.8B:latest")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# ── OpenAI 설정 ──────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",   "gpt-4o-mini")
OPENAI_TIMEOUT  = int(os.getenv("OPENAI_TIMEOUT", "60"))

# ── 백엔드 선택 ──────────────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")   # "ollama" | "openai"

# StatsBomb 피치 크기 (좌표 → 미터 환산용)
_SB_W, _SB_H   = 120.0, 80.0
_FIFA_W, _FIFA_H = 105.0, 68.0

# type_name → 한국어 변환
_TYPE_KO = {
    "pass":             "패스",
    "shot":             "슈팅",
    "shot_freekick":    "프리킥 슈팅",
    "shot_penalty":     "페널티킥 슈팅",
    "dribble":          "드리블",
    "take_on":          "돌파",
    "cross":            "크로스",
    "corner_crossed":   "코너킥 크로스",
    "corner_short":     "코너킥 숏패스",
    "freekick_crossed": "프리킥 크로스",
    "freekick_short":   "프리킥 숏패스",
    "throw_in":         "스로인",
    "goalkick":         "골킥",
    "clearance":        "클리어런스",
    "interception":     "인터셉트",
    "tackle":           "태클",
    "foul":             "파울",
    "bad_touch":        "볼 컨트롤 실수",
    "keeper_save":      "골키퍼 세이브",
    "keeper_claim":     "골키퍼 캐치",
    "keeper_punch":     "골키퍼 펀칭",
}


def _sb_to_pct(x: float, y: float) -> tuple[float, float]:
    """StatsBomb 좌표 → 피치 백분율 (0~100) 변환."""
    return round(x / _SB_W * 100, 1), round(y / _SB_H * 100, 1)


class LLMService:
    """Ollama 또는 OpenAI 기반 텍스트 생성 서비스.

    LLM_BACKEND 환경변수로 백엔드를 선택한다.
    - "ollama" : 로컬 Ollama 서버 (기본값)
    - "openai" : OpenAI Chat Completions API (OPENAI_API_KEY 필요)
    """

    def _generate(self, prompt: str) -> str:
        """설정된 백엔드로 텍스트를 생성해 반환한다."""
        if LLM_BACKEND == "openai":
            return self._generate_openai(prompt)
        return self._generate_ollama(prompt)

    def _generate_ollama(self, prompt: str) -> str:
        """Ollama /api/generate 엔드포인트를 호출해 텍스트를 반환한다."""
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def _generate_openai(self, prompt: str) -> str:
        """OpenAI Chat Completions API를 호출해 텍스트를 반환한다."""
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            },
            timeout=OPENAI_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def generate_situation(self, query: dict) -> str:
        """
        감지된 선수 좌표 정보를 바탕으로 현재 경기 상황을 한 문장으로 설명한다.

        Args:
            query: recommender가 반환한 query dict
                   {players, ball_pos, carrier_pos, carrier_team}
        """
        players     = query["players"]
        ball_pos    = query["ball_pos"]
        carrier_pos = query["carrier_pos"]
        carrier_team = query["carrier_team"]

        teammate_count = sum(1 for p in players if p["role"] == 1)
        opponent_count = sum(1 for p in players if p["role"] == -1)

        ball_px, ball_py     = _sb_to_pct(*ball_pos)
        carrier_px, carrier_py = _sb_to_pct(*carrier_pos)

        # 피치 구역 계산 (공 위치 기준)
        if ball_px < 33:
            zone = "자기 진영 수비 지역"
        elif ball_px < 67:
            zone = "미드필드"
        else:
            zone = "상대 진영 공격 지역"

        prompt = (
            "당신은 축구 전술 분석 전문가입니다. 아래 경기 데이터를 보고 "
            "현재 상황을 축구 선수가 이해할 수 있도록 한국어로 한 문장(30자 이내)만 작성하세요. "
            "설명, 인사, 부가 설명 없이 상황 문장만 출력하세요."
            "현재 상황에 대해서 전문적으로 설명해줘"
            "예시: 수비라인이 순간적으로 좁아지며 오른쪽 하프스페이스가 열렸어요. 지금은 압박이 느슨해지는 타이밍입니다.\n\n"
            f"- 공 위치: 피치 가로 {ball_px}%, 세로 {ball_py}% ({zone})\n"
            f"- 볼 캐리어 위치: 가로 {carrier_px}%, 세로 {carrier_py}%\n"
            f"- 캐리어 팀: {'공격팀' if carrier_team == 'a' else '수비팀'}\n"
            f"- 주변 팀원 수: {teammate_count}명\n"
            f"- 주변 상대 수: {opponent_count}명\n"
        )
        return self._generate(prompt)

    def generate_message(self, action_type: str, start: dict, end: dict | None) -> str:
        """
        추천 플레이 가이드를 바탕으로 선수에게 전달할 행동 지침을 한 문장으로 생성한다.

        Args:
            action_type: StatsBomb type_name (예: "pass")
            start: {"x": float, "y": float} 액션 시작 좌표 (StatsBomb)
            end:   {"x": float, "y": float} | None 액션 목적지 좌표
        """
        type_ko = _TYPE_KO.get(action_type, action_type)
        start_px, start_py = _sb_to_pct(start["x"], start["y"])

        location_desc = f"피치 가로 {start_px}%, 세로 {start_py}% 지점에서 {type_ko}"
        if end is not None:
            end_px, end_py = _sb_to_pct(end["x"], end["y"])
            location_desc += f"하여 가로 {end_px}%, 세로 {end_py}% 지점으로 이동"

        prompt = (
            "당신은 축구 전술 분석 전문가입니다. 아래 추천 플레이를 바탕으로 "
            "선수에게 전달할 행동 지침을 한국어로 한 문장(50자 이내)만 작성하세요. "
            "설명, 인사, 부가 설명 없이 지침 문장만 출력하세요."
            "예시: 수비가 안쪽으로 쏠린 상태라 반대 공간이 비어 있어요. 타이밍에 맞게 수비 사이로 패스를 시도해보세요.\n\n"
            f"- 추천 액션: {type_ko}\n"
            f"- 액션 설명: {location_desc}\n"
        )
        return self._generate(prompt)
