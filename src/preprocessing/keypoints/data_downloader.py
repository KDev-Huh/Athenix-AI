"""
SoccerNet calibration 데이터 다운로더.

SoccerNet 공식 API를 통해 calibration / calibration-2023 데이터를
로컬 디렉토리로 다운로드한다.
"""


class SoccerNetDataDownloader:
    """
    SoccerNet calibration 데이터를 다운로드하는 클래스.

    SoccerNet API를 통해 지정한 splits / tasks 의 calibration 데이터를
    로컬 디렉토리에 저장한다.

    사용 예시:
        downloader = SoccerNetDataDownloader(
            local_directory=r"C:\\Datasets\\SoccerNet\\Data",
            password="your_soccernet_password",
        )
        downloader.download()
    """

    DEFAULT_SPLITS = ["train", "valid", "test"]
    DEFAULT_TASKS  = ["calibration", "calibration-2023"]

    def __init__(
        self,
        local_directory: str = r"C:\Datasets\SoccerNet\Data",
        password: str = "",
        splits: list[str] = None,
        tasks: list[str] = None,
    ):
        """
        Args:
            local_directory: 데이터를 저장할 로컬 경로
            password:        SoccerNet API 비밀번호
            splits:          다운로드할 split 목록. None이면 기본값 사용.
                             기본값: ["train", "valid", "test"]
            tasks:           다운로드할 task 목록. None이면 기본값 사용.
                             기본값: ["calibration", "calibration-2023"]
        """
        self.local_directory = local_directory
        self.password        = password
        self.splits          = splits or self.DEFAULT_SPLITS
        self.tasks           = tasks  or self.DEFAULT_TASKS

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def download(self) -> None:
        """
        SoccerNet calibration 데이터를 모두 다운로드한다.

        tasks × splits 조합으로 순차 다운로드하며, 각 task 별 성공/실패를 출력한다.
        """
        from SoccerNet.Downloader import SoccerNetDownloader

        print("SoccerNet 다운로더 초기화...")
        downloader          = SoccerNetDownloader(LocalDirectory=self.local_directory)
        downloader.password = self.password

        print(f"  저장 경로 : {self.local_directory}")
        print(f"  Splits    : {self.splits}")
        print(f"  Tasks     : {self.tasks}")

        for task in self.tasks:
            print(f"\n[{task}] 다운로드 중...")
            try:
                downloader.downloadDataTask(task=task, split=self.splits)
                print(f"  ✓ {task} 완료")
            except Exception as e:
                print(f"  ✗ {task} 실패: {e}")

        print("\n다운로드 완료!")
