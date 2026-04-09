import os
import pandas as pd
from tqdm import tqdm
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader


class DataPreprocessor:
    """
    StatsBomb 데이터를 로드하고 SPADL 형식으로 변환하여 H5 파일로 저장하는 클래스.

    사용 예시:
        preprocessor = DataPreprocessor(data_dir="data/vaep", source="remote")

        # 사용 가능한 대회 목록 확인
        competitions = preprocessor.load_competitions()

        # 원하는 대회/시즌 선택 후 변환
        selected = {"FIFA World Cup": [2018]}
        data = preprocessor.convert(selected)

        # H5 파일로 저장
        preprocessor.save(data, filename="spadl-statsbomb.h5")
    """

    def __init__(self, data_dir: str = "data/vaep", source: str = "remote"):
        """
        Args:
            data_dir: 데이터 저장 루트 경로
            source:   "remote"만 허용 (StatsBomb 공개 API 사용)
        """
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

        if source != "remote":
            raise ValueError("source must be 'remote'. Local file loading is not implemented.")

        self.loader = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def load_competitions(self) -> pd.DataFrame:
        """StatsBomb에서 사용 가능한 대회 목록을 반환한다."""
        return self.loader.competitions()

    def convert(self, selected: dict) -> dict:
        """
        선택한 대회/시즌 데이터를 SPADL로 변환하여 반환한다.

        Args:
            selected: {"대회명": [시즌_연도, ...], ...}
                      예) {"FIFA World Cup": [2018], "Premier League": [2015, 2016]}

        Returns:
            {
                "competitions": DataFrame,
                "games":        DataFrame,
                "teams":        DataFrame,
                "players":      DataFrame,
                "player_games": DataFrame,
                "actions":      {game_id: DataFrame, ...}
            }
        """
        competitions = self.loader.competitions()
        target_games = self._filter_games(competitions, selected)

        teams, players, player_games, actions = [], [], [], {}

        for _, game in tqdm(target_games.iterrows(), total=len(target_games), desc="Converting games"):
            game_id = game["game_id"]

            teams.append(self.loader.teams(game_id))
            players.append(self.loader.players(game_id))

            events = self.loader.events(game_id)
            actions[game_id] = spadl.statsbomb.convert_to_actions(
                events, game["home_team_id"],
                xy_fidelity_version=1,
                shot_fidelity_version=1,
            )

            player_games.append(
                self.loader.players(game_id)[
                    ["player_id", "player_name", "nickname",
                     "is_starter", "starting_position_id", "minutes_played"]
                ].assign(game_id=game_id)
            )

        selected_competitions = competitions[
            competitions["competition_name"].isin(selected.keys())
        ]

        return {
            "competitions": selected_competitions.reset_index(drop=True),
            "games":        target_games.reset_index(drop=True),
            "teams":        pd.concat(teams).drop_duplicates("team_id").reset_index(drop=True),
            "players":      pd.concat(players).drop_duplicates("player_id")[["player_id", "player_name", "nickname"]].reset_index(drop=True),
            "player_games": pd.concat(player_games).reset_index(drop=True),
            "actions":      actions,
        }

    def save(self, data: dict, filename: str = "spadl-statsbomb.h5") -> None:
        """
        convert()의 결과를 HDF5 파일로 저장한다.

        Args:
            data:     convert()가 반환한 딕셔너리
            filename: 저장 파일명 (processed/ 디렉토리 기준)
        """
        output_path = os.path.join(self.processed_dir, filename)

        with pd.HDFStore(output_path, mode="w") as store:
            store["competitions"] = data["competitions"]
            store["games"]        = data["games"]
            store["teams"]        = data["teams"]
            store["players"]      = data["players"]
            store["player_games"] = data["player_games"]

            for game_id, action_df in tqdm(data["actions"].items(), desc="Saving actions"):
                store[f"actions/game_{game_id}"] = action_df

        print(f"Saved → {output_path}")

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _filter_games(self, competitions: pd.DataFrame, selected: dict) -> pd.DataFrame:
        """대회명/시즌 딕셔너리를 기반으로 게임 목록을 필터링한다."""
        all_games = []

        for competition_name, seasons in selected.items():
            comp_rows = competitions[competitions["competition_name"] == competition_name]

            if comp_rows.empty:
                print(f"[경고] '{competition_name}' 대회를 찾을 수 없습니다.")
                continue

            for season_year in seasons:
                season_rows = comp_rows[
                    comp_rows["season_name"].str.contains(str(season_year), na=False)
                ]

                if season_rows.empty:
                    print(f"[경고] '{competition_name}' {season_year} 시즌을 찾을 수 없습니다.")
                    continue

                for _, row in season_rows.iterrows():
                    games = self.loader.games(row["competition_id"], row["season_id"])
                    all_games.append(games)

        if not all_games:
            raise ValueError("선택한 조건에 맞는 게임이 없습니다.")

        return pd.concat(all_games).reset_index(drop=True)
