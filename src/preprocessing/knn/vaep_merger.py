import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl.statsbomb as sb_spadl

from src.recommendation.vaep.vaep_predictor import VAEPPredictor


class VAEPMerger:
    """
    StatsBomb 이벤트 데이터를 SPADL + VAEP + 360 freeze frame으로 병합해
    이벤트당 1행 CSV로 저장한다.

    출력 CSV 스키마 (19컬럼, 이벤트당 1행):
        event_id, match_id, period_id, time_seconds, team_id, player_id,
        type_name, result_name, bodypart_name,
        start_x, start_y, end_x, end_y,
        carrier_x, carrier_y,            ← actor 선수의 location_x/y
        offensive_value, defensive_value, vaep_value,
        freeze_frame                      ← JSON: [{"x":..., "y":..., "role":1/-1/2}, ...]

    사용 예시:
        merger = VAEPMerger(
            data_root="data/StatsBombGithub/external/knn/data",
            model_dir="models/vaep",
        )
        merger.load_models("worldcup_2018_scores.json", "worldcup_2018_concedes.json")

        # 특정 시즌 하나만
        df = merger.run(
            competitions=[(43, 106)],
            output_path="data/StatsBombGithub/processed/vaep_360_merged.csv",
        )

        # 대회 ID만 넣으면 해당 대회의 모든 시즌을 자동으로 포함
        df = merger.run(
            competitions=[43, 11],          # FIFA World Cup 전 시즌 + La Liga 전 시즌
            output_path="data/StatsBombGithub/processed/vaep_360_merged.csv",
        )

        # 혼합 사용
        df = merger.run(
            competitions=[43, (16, 4)],     # World Cup 전 시즌 + UCL 2018/2019만
            output_path="data/StatsBombGithub/processed/vaep_360_merged.csv",
        )
    """

    def __init__(self, data_root: str, model_dir: str) -> None:
        """
        Args:
            data_root: StatsBomb 로컬 데이터 루트 경로 (events/, three-sixty/ 폴더 포함)
            model_dir: VAEP 모델 파일이 저장된 디렉토리
        """
        self.data_root = Path(data_root)
        self.model_dir = model_dir
        self._loader   = StatsBombLoader(getter="local", root=str(self.data_root))
        self._predictor = VAEPPredictor(model_dir=model_dir)

    # ── 공개 메서드 ────────────────────────────────────────────────────────

    def load_models(self, scores_model: str, concedes_model: str) -> None:
        """
        VAEP 모델을 로드한다.

        Args:
            scores_model:   scores 모델 파일명 (예: "worldcup_2018_scores.json")
            concedes_model: concedes 모델 파일명 (예: "worldcup_2018_concedes.json")
        """
        self._predictor.load_models(scores_model, concedes_model)

    def run(
        self,
        competitions: list,
        output_path: str,
    ) -> pd.DataFrame:
        """
        전체 파이프라인을 실행하고 이벤트당 1행 DataFrame을 반환한다.

        Args:
            competitions: 처리할 대회/시즌 목록. 각 원소는 다음 중 하나:
                            - int          → 해당 대회의 모든 시즌 포함  (예: 43)
                            - (int, int)   → 특정 대회+시즌만 포함       (예: (43, 106))
            output_path:  출력 CSV 저장 경로

        Returns:
            이벤트당 1행 DataFrame (vaep_360_merged.csv 내용)
        """
        print("[1/5] 경기 목록 로드 중...")
        games = self._load_games(competitions)
        print(f"      경기 수: {len(games)}")

        print("[2/5] 전체 경기 VAEP 계산 중...")
        df_vaep = self._build_vaep(games)
        print(f"      완료 — 총 액션: {len(df_vaep):,}")

        print("[3/5] three-sixty 데이터 로드 중...")
        df_360 = self._load_360(games)
        print(f"      완료 — 행: {len(df_360):,}  이벤트: {df_360['event_uuid'].nunique():,}")

        print("[4/5] VAEP + 360 병합 중...")
        df_merged = self._merge(df_vaep, df_360)
        print(f"      완료 — 병합 행: {len(df_merged):,}  이벤트: {df_merged['event_id'].nunique():,}")

        print("[5/5] event_id 기준 그룹화 및 저장 중...")
        df_events = self._group_by_event(df_merged)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_events.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"      저장 완료: {out}")
        print(f"      행 수: {len(df_events):,}  컬럼: {list(df_events.columns)}")

        return df_events

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _load_games(self, competitions: list) -> pd.DataFrame:
        """
        competitions 목록을 파싱해 경기 목록을 로드한다.

        - int 원소   → competitions.json을 읽어 해당 comp_id의 모든 season_id를 자동 수집
        - tuple 원소 → (comp_id, season_id)로 직접 사용

        Args:
            competitions: [int | (int, int), ...] 형태의 대회/시즌 목록

        Returns:
            모든 대상 경기를 합친 DataFrame
        """
        # competitions.json에서 comp_id → [season_id, ...] 매핑 구축
        comp_json = self.data_root / "competitions.json"
        with open(comp_json, encoding="utf-8") as f:
            all_comps = json.load(f)
        comp_seasons: dict[int, list[int]] = {}
        for entry in all_comps:
            cid = entry["competition_id"]
            comp_seasons.setdefault(cid, []).append(entry["season_id"])

        # 요청된 (comp_id, season_id) 쌍 목록 구성
        pairs: list[tuple[int, int]] = []
        for item in competitions:
            if isinstance(item, (tuple, list)):
                pairs.append((int(item[0]), int(item[1])))
            else:
                cid = int(item)
                if cid not in comp_seasons:
                    warnings.warn(f"comp_id={cid} 를 competitions.json에서 찾을 수 없습니다. 건너뜀.")
                    continue
                for sid in comp_seasons[cid]:
                    pairs.append((cid, sid))

        if not pairs:
            raise ValueError("처리할 대회/시즌이 없습니다. competitions 목록을 확인하세요.")

        # 중복 제거 (같은 쌍이 두 번 지정될 경우 대비)
        pairs = list(dict.fromkeys(pairs))

        all_games = []
        for comp_id, season_id in pairs:
            try:
                games = self._loader.games(comp_id, season_id)
                games["comp_id"]   = comp_id
                games["season_id"] = season_id
                all_games.append(games)
                print(f"      comp={comp_id}, season={season_id} → {len(games)}경기")
            except Exception as e:
                warnings.warn(f"comp={comp_id}, season={season_id} 로드 실패: {e}")

        if not all_games:
            raise RuntimeError("경기 데이터를 하나도 로드하지 못했습니다.")

        return pd.concat(all_games, ignore_index=True)

    def _build_vaep(self, games: pd.DataFrame) -> pd.DataFrame:
        """모든 경기의 SPADL 변환 + VAEP 계산 결과를 하나의 DataFrame으로 반환한다.

        is_ltr (공격 방향 정규화):
            홈팀은 전반(period 1, 3)에 왼쪽→오른쪽(x 증가 방향)으로 공격한다고 간주.
            - 홈팀 + 홀수 period  → is_ltr = True  (왼→오)
            - 홈팀 + 짝수 period  → is_ltr = False (오→왼)
            - 원정팀은 반대
        """
        # game_id → home_team_id 매핑
        home_lookup = games.set_index("game_id")["home_team_id"].to_dict()

        all_vaep = []
        for _, game in tqdm(games.iterrows(), total=len(games), desc="VAEP 계산"):
            gid  = int(game["game_id"])
            htid = int(game["home_team_id"])
            try:
                events  = self._loader.events(gid)
                actions = sb_spadl.convert_to_actions(events, htid)
                actions["game_id"] = gid
                result  = self._predictor.predict(actions)

                # SPADL → StatsBomb 좌표 복원 (팀별로 다른 변환 필요)
                #
                # SPADL 내부 변환 두 가지가 복합 적용됨:
                #  (1) _convert_locations: y축 반전
                #        y_spadl = field_width - (sb_y / 80) * field_width
                #        → sb_y 복원: 80 - y_spadl * (80/68)
                #  (2) _fix_direction_of_play: away팀 x, y 모두 반전
                #        x_spadl_away = field_length - x_spadl_home
                #        y_spadl_away = field_width  - y_spadl_home  (= sb_y 복원)
                #
                # 결과:
                #   home팀 x: x_spadl * (120/105) → sb_x           ✅
                #   home팀 y: 80 - y_spadl * (80/68) → sb_y        ✅
                #   away팀 x: 120 - x_spadl * (120/105) → sb_x     ✅
                #   away팀 y: y_spadl * (80/68) → sb_y              ✅
                is_away = result["team_id"] != htid

                for col in ["start_x", "end_x"]:
                    result.loc[~is_away, col] = result.loc[~is_away, col] * (120.0 / 105.0)
                    result.loc[is_away,  col] = 120.0 - result.loc[is_away, col] * (120.0 / 105.0)

                for col in ["start_y", "end_y"]:
                    result.loc[~is_away, col] = 80.0 - result.loc[~is_away, col] * (80.0 / 68.0)
                    result.loc[is_away,  col] = result.loc[is_away, col] * (80.0 / 68.0)

                all_vaep.append(result)
            except Exception as e:
                warnings.warn(f"game {gid} 건너뜀: {e}")

        df_vaep = pd.concat(all_vaep, ignore_index=True)

        # is_ltr 계산 — 홈팀 + 홀수 period 또는 원정팀 + 짝수 period
        df_vaep["home_team_id"] = df_vaep["game_id"].map(home_lookup)
        is_home   = df_vaep["team_id"] == df_vaep["home_team_id"]
        odd_period = df_vaep["period_id"].isin([1, 3])
        df_vaep["is_ltr"] = (is_home & odd_period) | (~is_home & ~odd_period)
        df_vaep = df_vaep.drop(columns=["home_team_id"])

        return df_vaep

    def _load_360(self, games: pd.DataFrame) -> pd.DataFrame:
        """대상 경기의 three-sixty freeze frame 데이터를 로드한다."""
        three60_dir  = self.data_root / "three-sixty"
        game_ids_set = set(games["game_id"].astype(int))

        rows, skipped = [], []
        for path in sorted(three60_dir.glob("*.json")):
            match_id = int(path.stem)
            if match_id not in game_ids_set:
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    records = json.load(f)
            except json.JSONDecodeError:
                skipped.append(path.name)
                continue
            for rec in records:
                for player in rec["freeze_frame"]:
                    rows.append({
                        "match_id"  : match_id,
                        "event_uuid": rec["event_uuid"],
                        "teammate"  : player["teammate"],
                        "actor"     : player["actor"],
                        "keeper"    : player["keeper"],
                        "location_x": player["location"][0],
                        "location_y": player["location"][1],
                    })

        if skipped:
            warnings.warn(f"JSON 파싱 실패 파일: {skipped}")
        return pd.DataFrame(rows)

    def _merge(self, df_vaep: pd.DataFrame, df_360: pd.DataFrame) -> pd.DataFrame:
        """VAEP + 360 데이터를 (match_id, event_uuid) 기준으로 inner join한다."""
        vaep_cols = [
            "game_id", "original_event_id", "period_id", "time_seconds",
            "team_id", "player_id", "type_name", "result_name", "bodypart_name",
            "start_x", "start_y", "end_x", "end_y",
            "offensive_value", "defensive_value", "vaep_value",
            "is_ltr",
        ]
        df_merged = df_360.merge(
            df_vaep[vaep_cols],
            left_on=["match_id", "event_uuid"],
            right_on=["game_id", "original_event_id"],
            how="inner",
        ).drop(columns=["game_id", "original_event_id"])
        return df_merged.rename(columns={"event_uuid": "event_id"})

    def _group_by_event(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """
        이벤트당 1행으로 그룹화한다.

        - actor==True 행 → 이벤트 메타 + carrier_x/y 추출
        - actor==False 행 → freeze_frame JSON 문자열로 직렬화
        - is_ltr==False 이벤트 → 좌표 반전(x=120-x, y=80-y)으로 공격 방향 정규화
        """
        # 캐리어(actor==True) 행에서 이벤트 메타 추출
        actor_cols = [
            "event_id", "match_id", "period_id", "time_seconds", "team_id", "player_id",
            "type_name", "result_name", "bodypart_name",
            "start_x", "start_y", "end_x", "end_y",
            "offensive_value", "defensive_value", "vaep_value",
            "location_x", "location_y",
            "is_ltr",
        ]
        carrier = (
            df_merged[df_merged["actor"] == True]
            .drop_duplicates("event_id")[actor_cols]
            .rename(columns={"location_x": "carrier_x", "location_y": "carrier_y"})
        )

        # RTL 이벤트 좌표 반전 — 공격 방향 정규화 (모두 왼→오로)
        rtl = ~carrier["is_ltr"]
        for col_x, col_y in [("start_x", "start_y"), ("end_x", "end_y"), ("carrier_x", "carrier_y")]:
            carrier.loc[rtl, col_x] = 120 - carrier.loc[rtl, col_x]
            carrier.loc[rtl, col_y] = 80  - carrier.loc[rtl, col_y]

        # 비캐리어 선수 → role 인코딩 → freeze_frame JSON
        others = df_merged[df_merged["actor"] == False].copy()
        others["role"] = np.where(
            others["keeper"], 2,
            np.where(others["teammate"], 1, -1),
        )

        # RTL 이벤트의 freeze_frame 좌표도 반전
        rtl_event_ids = set(carrier.loc[rtl, "event_id"])
        def make_entry(r):
            x, y = r["location_x"], r["location_y"]
            if r["event_id"] in rtl_event_ids:
                x = 120 - x
                y = 80  - y
            return {"x": x, "y": y, "role": int(r["role"])}

        others["_entry"] = others.apply(make_entry, axis=1)
        freeze = (
            others.groupby("event_id")["_entry"]
            .apply(list)
            .reset_index()
            .rename(columns={"_entry": "freeze_frame"})
        )
        freeze["freeze_frame"] = freeze["freeze_frame"].apply(json.dumps)

        col_order = [
            "event_id", "match_id", "period_id", "time_seconds", "team_id", "player_id",
            "type_name", "result_name", "bodypart_name",
            "start_x", "start_y", "end_x", "end_y",
            "carrier_x", "carrier_y",
            "offensive_value", "defensive_value", "vaep_value",
            "freeze_frame",
        ]
        return carrier.merge(freeze, on="event_id", how="left")[col_order]
