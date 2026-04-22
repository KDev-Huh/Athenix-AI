## 7) AI 피드백 API

### 7.1 AI 더 나은 플레이 추천
- Method/Path: `POST /ai/recommend/image`
- Auth: 불필요
- Content-Type: `multipart/form-data`
- Request Body:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | `file` | ✅ | 분석할 경기 장면 이미지 (jpg, png, webp) |

- Request Example (multipart/form-data):

```
POST /ai/recommend/image
Content-Type: multipart/form-data; boundary=----FormBoundary

------FormBoundary
Content-Disposition: form-data; name="image"; filename="scene.jpg"
Content-Type: image/jpeg

<binary image data>
------FormBoundary--
```

- Response: `200 OK`

```json
{
  "success": true,
  "data": {
    "situation": "수비 라인이 좁아졌고 공격팀 캐리어가 상대 진영에 위치합니다.",
    "playGuide": {
      "type": "pass",
      "start":       { "x": 42.1,  "y": 58.3 },
      "end":         { "x": 71.8,  "y": 44.6 },
      "start_pixel": { "x": 312.0, "y": 204.0 },
      "end_pixel":   { "x": 520.0, "y": 178.0 },
      "message": "수비를 끌어낸 뒤 오른쪽 하프스페이스로 스루패스를 시도하세요."
    }
  },
  "error": null
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `situation` | `string` | 이미지에서 감지한 선수 좌표를 LLM이 분석한 현재 경기 상황 설명 |
| `playGuide.type` | `string` | 추천 액션 타입 (아래 enum 참고) |
| `playGuide.start` | `{x, y}` | 액션 시작 지점 StatsBomb 좌표 (x: 0~120, y: 0~80) |
| `playGuide.end` | `{x, y} \| null` | 액션 목적지 StatsBomb 좌표 (없으면 null) |
| `playGuide.start_pixel` | `{x, y} \| null` | 시작 지점 원본 이미지 픽셀 좌표 |
| `playGuide.end_pixel` | `{x, y} \| null` | 목적지 원본 이미지 픽셀 좌표 (end가 null이면 null) |
| `playGuide.message` | `string` | 추천 액션에 대해 LLM이 생성한 행동 지침 |

- Error Response:

| Status | 설명 |
|--------|------|
| `400`  | 지원하지 않는 이미지 형식 |
| `404`  | 유사 장면을 찾지 못함 |
| `422`  | 호모그래피 계산 실패 / 선수 미감지 등 분석 오류 |
| `500`  | LLM 서버(Ollama) 연결 실패 등 내부 오류 |

- `playGuide.type` enum (StatsBomb 이벤트 타입):

| 값 | 설명 |
|----|------|
| `pass` | 패스 |
| `shot` | 슈팅 |
| `shot_freekick` | 프리킥 슈팅 |
| `shot_penalty` | 페널티킥 슈팅 |
| `dribble` | 드리블 |
| `take_on` | 돌파 시도 |
| `cross` | 크로스 |
| `corner_crossed` | 코너킥 (크로스) |
| `corner_short` | 코너킥 (숏) |
| `freekick_crossed` | 프리킥 (크로스) |
| `freekick_short` | 프리킥 (숏) |
| `throw_in` | 스로인 |
| `goalkick` | 골킥 |
| `clearance` | 클리어런스 |
| `interception` | 인터셉트 |
| `tackle` | 태클 |
| `foul` | 파울 |
| `bad_touch` | 볼 컨트롤 실수 |
| `keeper_save` | 골키퍼 세이브 |
| `keeper_claim` | 골키퍼 캐치 |
| `keeper_punch` | 골키퍼 펀칭 |

- 좌표계:
  - `start_x`, `start_y`, `end_x`, `end_y`는 StatsBomb 좌표계 기준입니다.
  - `start_x`: `0 ~ 120`, `start_y`: `0 ~ 80`
  - `end_x`, `end_y`: 해당 이벤트에 목적지 좌표가 없을 경우 `null` 반환

- LLM:
  - `situation`, `message` 생성에 Ollama `EEVE-Korean-10.8B` 모델을 사용합니다.
  - Ollama 서버 주소는 환경변수 `OLLAMA_BASE_URL` (기본값: `http://localhost:11434`)로 설정합니다.
